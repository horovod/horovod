# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import collections
import logging
import threading
import socket

from six.moves import socketserver, BaseHTTPServer, SimpleHTTPServer

from horovod.run.util.network import find_port

# Timeout for reading from a single request
SINGLE_REQUEST_TIMEOUT = 3

BAD_REQUEST = 400
TIMEOUT = 408
OK = 200


class KVStoreHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    # Set timeout
    timeout = SINGLE_REQUEST_TIMEOUT

    # Override GET handler
    def do_GET(self):
        paths = self.path.split('/')
        if len(paths) < 3:
            logging.error(
                'KVStore ERROR: Invalid request path: {path}.'.format(
                    path=self.path))
            self.send_status_code(BAD_REQUEST)
            return

        _, scope, key = paths
        value = self._get_value(scope, key)

        if value is None:
            self.send_status_code(404)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)

    # Override PUT handler
    def do_PUT(self):
        paths = self.path.split('/')
        if len(paths) < 3:
            logging.error(
                'KVStore ERROR: Invalid request path: {path}.'.format(
                    path=self.path))
            self.send_status_code(BAD_REQUEST)
            return

        _, scope, key = paths

        # Get body length
        content_length = int(self.headers['Content-Length'])
        try:
            value = self.rfile.read(content_length)
        except socket.timeout:
            if self.server.verbose:
                logging.error(
                    'KVStore ERROR: Timeout when receiving {content_bytes} '
                    'bytes, aborting this incomplete request.' .format(
                        content_bytes=content_length))

            # If timeout, abort this request
            self.send_status_code(TIMEOUT)
            return

        self._put_value(scope, key, value)
        self.send_status_code(OK)

    def send_status_code(self, status_code):
        self.send_response(status_code)
        self.send_header("Content-Length", 0)
        self.end_headers()

    # Override this function to prevent SimpleHTTPServer printing every
    # request out.
    def log_message(self, format, *args):
        pass

    def _get_value(self, scope, key):
        with self.server.cache_lock:
            return self.server.cache.get(scope, {}).get(key)

    def _put_value(self, scope, key, value):
        with self.server.cache_lock:
            scope_dict = self.server.cache.setdefault(scope, {})
            scope_dict[key] = value
            if self.server.verbose:
                logging.info(scope, self.server.cache[scope].keys())


class RendezvousHandler(KVStoreHandler):
    # Override DELETE handler
    def do_DELETE(self):
        paths = self.path.split('/')
        if len(paths) < 3:
            logging.error(
                'Rendezvous ERROR: Invalid request path: {path}.'.format(
                    path=self.path))
            self.send_status_code(BAD_REQUEST)
            return

        _, scope, key = paths

        with self.server.finished_list_lock:
            self.server.finished_list[scope].append(key)
            if self.server.scope_size[scope] == len(self.server.finished_list[scope]):
                with self.server.cache_lock:
                    self.server.cache.get(scope, {}).clear()

        self.send_status_code(OK)


class RendezvousHTTPServer(socketserver.ThreadingMixIn, BaseHTTPServer.HTTPServer, object):
    def __init__(self, addr, handler, verbose):
        # This class has to inherit from object since HTTPServer is an old-style
        # class that does not inherit from object.
        super(RendezvousHTTPServer, self).__init__(addr, handler)

        # Cache that provides the store
        self.cache_lock = threading.Lock()
        self.cache = {}

        self.verbose = verbose

        # Lists for finished rendezvous workers
        self.finished_list_lock = threading.Lock()
        self.finished_list = collections.defaultdict(list)

        # Total size for scopes
        self.scope_size = {}

    def init(self, host_alloc_plan):
        with self.cache_lock:
            self.cache.clear()

        with self.finished_list_lock:
            self.finished_list.clear()

        self.scope_size.clear()
        self._extract_scope_size(host_alloc_plan)

    def _extract_scope_size(self, host_alloc_plan):
        for slot_info in host_alloc_plan:
            self.scope_size['global_'] = slot_info.size
            cross_rank = slot_info.cross_rank
            self.scope_size['local_' + str(cross_rank)] = slot_info.local_size
            local_rank = slot_info.local_rank
            self.scope_size['cross_' + str(local_rank)] = slot_info.cross_size

    def should_continue(self):
        return True


class RendezvousServer:
    def __init__(self, verbose=0):
        self.httpd = None
        self.listen_thread = None
        self.verbose = verbose

    # Rendezvous function finds a available port, create http socket,
    # and start listening loop to handle request
    # self.httpd.init needs to be called after server start
    def start_server(self, handler_cls=RendezvousHandler):
        self.httpd, port = find_port(
            lambda addr: RendezvousHTTPServer(
                addr, handler_cls, self.verbose))
        if self.verbose:
            logging.info('Rendezvous INFO: HTTP rendezvous server started.')

        # start the listening loop
        self.listen_thread = threading.Thread(target=self.httpd.serve_forever)
        self.listen_thread.daemon = True
        self.listen_thread.start()

        return port

    def stop_server(self):
        self.httpd.shutdown()
        self.listen_thread.join()


class KVStoreHTTPServer(BaseHTTPServer.HTTPServer, object):
    def __init__(self, addr, handler, verbose):
        super(KVStoreHTTPServer, self).__init__(addr, handler)

        # Cache that provides the store
        self.cache_lock = threading.Lock()
        self.cache = {}

        self.verbose = verbose


class KVStoreServer:
    def __init__(self, verbose):
        self.httpd = None
        self.listen_thread = None
        self.verbose = verbose

    # KVStore server finds a available port, create http socket,
    # and start listening loop to handle request
    def start_server(self):
        self.httpd, port = find_port(
            lambda addr: KVStoreHTTPServer(
                addr, KVStoreHandler, self.verbose))

        self.listen_thread = threading.Thread(target=self.httpd.serve_forever)
        self.listen_thread.daemon = True
        self.listen_thread.start()

        if self.verbose:
            logging.info('KVStoreServer INFO: KVStore server started. Listen on port ' + str(port))

        return port

    def shutdown_server(self):
        self.httpd.shutdown()

        self.httpd.server_close()

        if self.verbose:
            logging.info('KVStoreServer INFO: KVStore server finishes.')
        # Because this thread is daemonized, no need to join.
