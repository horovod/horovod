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
import socket
import threading

from six.moves import BaseHTTPServer, SimpleHTTPServer

from horovod.run.util.threads import in_thread
from horovod.run.util.network import find_port

# Timeout for reading from a single request
SINGLE_REQUEST_TIMEOUT = 3

# Timeout for accepting new request
TOTAL_TIMEOUT = 60

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
            print(
                'KVStore ERROR: Invalid request path: {path}.'.format(
                    path=self.path))
            self.send_status_code(BAD_REQUEST)
            return

        _, scope, key = paths
        with self.server.cache_lock:
            value = self.server.cache.get(scope, {}).get(key)

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
            print(
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
                print(
                    'KVStore ERROR: Timeout when receiving {content_bytes} '
                    'bytes, aborting this incomplete request.' .format(
                        content_bytes=content_length))

            # If timeout, abort this request
            self.send_status_code(TIMEOUT)
            return

        with self.server.cache_lock:
            scope_dict = self.server.cache.setdefault(scope, {})
            scope_dict[key] = value
            if self.server.verbose:
                print(scope, self.server.cache[scope].keys())

        self.send_status_code(OK)

    def send_status_code(self, status_code):
        self.send_response(status_code)
        self.send_header("Content-Length", 0)
        self.end_headers()

    # Override this function to prevent SimpleHTTPServer printing every
    # request out.
    def log_message(self, format, *args):
        pass


class RendezvousHandler(KVStoreHandler):
    # Override DELETE handler
    def do_DELETE(self):
        paths = self.path.split('/')
        if len(paths) < 3:
            print(
                'Rendezvous ERROR: Invalid request path: {path}.'.format(
                    path=self.path))
            self.send_status_code(BAD_REQUEST)
            return

        _, scope, key = paths

        with self.server.finished_list_lock:
            self.server.finished_list[scope].append(key)

        self.send_status_code(OK)


class RendezvousHTTPServer(BaseHTTPServer.HTTPServer, object):
    def __init__(self, addr, handler, verbose):
        # This class has to inherit from object since HTTPServer is an old-style
        # class that does not inherit from object.
        super(RendezvousHTTPServer, self).__init__(addr, handler)

        # Lists for finished rendezvous workers
        self.finished_list_lock = threading.Lock()
        self.finished_list = collections.defaultdict(list)

        # Total size for scopes
        self.scope_size = {}

        # Cache that provides the store
        self.cache_lock = threading.Lock()
        self.cache = {}

        self.verbose = verbose

    def extract_scope_size(self, host_alloc_plan):
        for slot_info in host_alloc_plan:
            self.scope_size['global'] = slot_info.size
            cross_rank = slot_info.cross_rank
            self.scope_size['local_' + str(cross_rank)] = slot_info.local_size
            local_rank = slot_info.local_rank
            self.scope_size['cross_' + str(local_rank)] = slot_info.cross_size

    # Decide whether all ranks have confirmed rendezvous completion.
    def should_continue(self):
        should_continue = False
        with self.finished_list_lock:
            for scope, cnt in self.scope_size.items():
                if cnt > len(self.finished_list[scope]):
                    should_continue = True
        return should_continue

    def handle_timeout(self):
        error_msg = 'Rendezvous ERROR: Rendezvous server timeout after ' \
                    '{time} seconds while waiting for all the ranks to send finalize ' \
                    'messages.\n'.format(time=TOTAL_TIMEOUT)

        for scope, finished_list in self.finished_list:
            if self.scope_size[scope] > len(finished_list):
                error_msg += 'Scope {scope} expects {size} workers, only received' \
                    'finalized message from [{ranks}].\n'.format(
                        scope=scope,
                        size=self.scope_size[scope],
                        ranks=' '.join(finished_list))

        raise RuntimeError(error_msg)


class RendezvousServer:
    def __init__(self, verbose):
        self.httpd = None
        self.listen_thread = None
        self.verbose = verbose

    # Rendezvous function finds a available port, create http socket,
    # and start listening loop to handle request
    def start_server(self, host_alloc_plan):
        self.httpd, port = find_port(
            lambda addr: RendezvousHTTPServer(
                addr, RendezvousHandler, self.verbose))
        self.httpd.extract_scope_size(host_alloc_plan)
        if self.verbose:
            print('Rendezvous INFO: HTTP rendezvous server started.')

        # start the listening loop
        self.listen_thread = in_thread(target=self.listen_loop)

        return port

    # Listening loop for handle request
    def listen_loop(self):
        while self.httpd.should_continue():
            self.httpd.handle_request()

        self.httpd.server_close()

        if self.verbose:
            print('Rendezvous INFO: Rendezvous finishes.')
        # Because this thread is daemonized, no need to join.


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

        self.listen_thread = in_thread(target=self.httpd.serve_forever)

        if self.verbose:
            print('KVStoreServer INFO: KVStore server started. Listen on port ' + str(port))

        return port

    def shutdown_server(self):
        self.httpd.shutdown()

        self.httpd.server_close()

        if self.verbose:
            print('KVStoreServer INFO: KVStore server finishes.')
        # Because this thread is daemonized, no need to join.
