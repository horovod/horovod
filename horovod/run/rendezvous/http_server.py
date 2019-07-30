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

from six.moves import BaseHTTPServer, SimpleHTTPServer
import random
import threading
import socket

# Timeout for reading from a single request
SINGLE_REQUEST_TIMEOUT = 3

# Timeout for accepting new request
TOTAL_TIMEOUT = 60


class MyHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

    # Set timeout
    timeout = SINGLE_REQUEST_TIMEOUT

    # Override POST HTTP request handler
    def do_POST(self):
        paths = self.path.split('/')

        if len(paths) < 3:
            print(
                'Rendezvous ERROR: Invalid request path: {path}.'.format(
                    path=self.path))
            self.send_status_code(400)
            return

        if self.server.verbose >= 5:
            print('Rendezvous TEACE: Receive reqeust to url ' + self.path)

        if paths[2] == 'get':
            self.handle_get(paths[1])
        elif paths[2] == 'set':
            self.handle_set(paths[1])
        elif paths[2] == 'finalize':
            self.handle_finalize(paths[1])
        else:
            print(
                'Rendezvous ERROR: Unsupported request type: {type}.'.format(
                    type=paths[2]))
            self.send_status_code(400)

    def read_from_socket(self, sock_input, content_bytes):
        try:
            value = sock_input.read(content_bytes)
            return value
        except socket.timeout:
            if self.server.verbose >= 2:
                print(
                    'Rendezvous WARNING: Timeout when receiving {content_bytes} '
                    'bytes, aborting this incomplete request.' .format(
                        content_bytes=content_bytes))
            self.send_status_code(408)
            return bytes('')

    def send_status_code(self, status_code):
        self.send_response(status_code)
        self.send_header("Content-Length", 0)
        self.end_headers()

    def handle_get(self, namespace):
        key_len = int(self.headers.get('Key-Length', '-1'))
        if key_len <= -1:
            if self.server.verbose >= 2:
                print(
                    'Rendezvous WARNING: Invalid get request with no Key-Length '
                    'header line, aborting this incomplete request.')
            self.send_status_code(400)
            return

        key = self.read_from_socket(self.rfile, key_len)
        if len(key) != key_len:
            if self.server.verbose >= 5:
                print(
                    'Rendezvous TRACE: Key-Length is specified as {key_len}, '
                    'but read {true_len} from socket without timeout, '
                    'aborting this incomplete request.' .format(
                        key_len=key_len, true_len=len(key)))
            self.send_status_code(400)
            return

        if self.server.verbose >= 5:
            rank = self.headers.get('Rank')
            print('Rendezvous TEACE: Receive a get request from '
                  '{namespace}:{rank} with key: {key}.' .format(
                      namespace=namespace,
                      rank=rank,
                      key=key))

        with self.server.cache_lock:
            value = self.server.cache.get(namespace, {}).get(key, '')

        self.send_response(200)
        self.send_header("Content-Length", str(len(value)))
        self.end_headers()
        self.wfile.write(value)
        self.wfile.flush()

        if self.server.verbose >= 5:
            rank = self.headers.get('Rank')
            print(
                'Rendezvous TRACE: Respond a get request from '
                '{namespace}:{rank} with value length {val_len}.' .format(
                    namespace=namespace,
                    rank=rank,
                    val_len=len(value)))

    def handle_set(self, namespace):
        key_len = int(self.headers.get('Key-Length', '-1'))
        if key_len <= -1:
            if self.server.verbose >= 2:
                print(
                    'Rendezvous WARNING: Invalid Set request with no Key-Length '
                    'header line, aborting this incomplete request.')
            self.send_status_code(400)
            return

        key = self.read_from_socket(self.rfile, key_len)

        value_len = int(self.headers.get('Value-Length', '-1'))
        if value_len <= -1:
            if self.server.verbose >= 2:
                print(
                    'Rendezvous WARNING: Invalid Set request with no Value-Length '
                    'header line, aborting this incomplete request.')
            self.send_status_code(400)
            return

        if self.server.verbose >= 5:
            rank = self.headers.get('Rank')
            print(
                'Rendezvous TRACE: Receive a set request from '
                '{namespace}:{rank} with key: {key}, '
                'value length: {val_len}.' .format(
                    namespace=namespace,
                    rank=rank,
                    key=key,
                    val_len=value_len))

        value = self.read_from_socket(self.rfile, value_len)

        if len(key) < key_len:
            if self.server.verbose >= 2:
                print(
                    'Rendezvous WARNING: Key-Length is specified as {len}, '
                    'but read {true_len} from socket without timeout, '
                    'aborting this incomplete request.' .format(
                        len=key_len, true_len=len(key)))
            self.send_status_code(400)
            return

        if len(value) < value_len:
            if self.server.verbose >= 2:
                print(
                    'Rendezvous WARNING: Value-Length is specified as {len}, '
                    'but read {true_len} from socket without timeout, '
                    'aborting this incomplete request.' .format(
                        len=value_len, true_len=len(value)))
            self.send_status_code(400)
            return

        self.send_status_code(200)

        with self.server.cache_lock:
            namespace_dict = self.server.cache.setdefault(namespace, {})
            namespace_dict[key] = value
            if self.server.verbose >= 5:
                print(self.server.cache.keys())

    def handle_finalize(self, namespace):
        if self.server.verbose >= 4:
            rank = self.headers.get('Rank')
            print(
                'Rendezvous DEBUG: Receive a finalize request from '
                '{namespace}:{rank}.' .format(
                    namespace=namespace, rank=rank))
        self.send_status_code(200)

        with self.server.finished_cnt_lock:
            self.server.finished_cnt[namespace] += 1

    # Override this function to prevent SimpleHTTPServer printing every
    # request out.
    def log_message(self, format, *args):
        pass


class MyHTTPServer(BaseHTTPServer.HTTPServer, object):
    def __init__(self, addr, handler, alloc_plan, verbose):
        super(MyHTTPServer, self).__init__(addr, handler)
        self.httpd = None

        # Count for finished rendezvous workers
        self.finished_cnt_lock = threading.Lock()
        self.finished_cnt = collections.defaultdict(int)
        self.comm_size = {}
        self.extract_comm_size(alloc_plan)

        # Cache that provides the store
        self.cache_lock = threading.Lock()
        self.cache = {}

        self.verbose = verbose

    def extract_comm_size(self, alloc_plan):
        # print(alloc_plan)
        for rank, info in enumerate(alloc_plan):
            # print(rank, info)
            self.comm_size['global'] = info['Size']
            cross_rank = info['Cross_rank']
            self.comm_size['local' + str(cross_rank)] = info['Local_size']
            local_rank = info['Local_rank']
            self.comm_size['cross' + str(local_rank)] = info['Cross_size']

    # Decide whether all ranks have confirmed rendezvous completion.
    def should_continue(self):
        should_continue = False
        with self.finished_cnt_lock:
            for scope, cnt in self.comm_size.items():
                if cnt > self.finished_cnt[scope]:
                    should_continue = True
        return should_continue

    def handle_timeout(self):
        raise Exception(
            'Rendezvous ERROR: Rendezvous server timeout after '
            '{time} seconds while waiting for all the ranks to send finalize '
            'messages. Received {recv}, expecting {expect}. '
            'Please make sure number of process set by -np is correct.' .format(
                time=TOTAL_TIMEOUT, recv=self.finished_cnt, expect=self.comm_size))


class RendezvousServer:
    def __init__(self, alloc_plan, verbose):
        self.httpd = None
        self.listen_thread = None
        self.alloc_plan = alloc_plan
        self.verbose = verbose

    def _find_port(self):
        # Create a HTTP socket
        min_port = 1024
        max_port = 65536
        num_ports = max_port - min_port
        start_port = random.randrange(0, num_ports)
        for port_offset in range(num_ports):
            try:
                port = min_port + (start_port + port_offset) % num_ports
                addr = ('', port)
                self.httpd = MyHTTPServer(
                    addr, MyHandler, self.alloc_plan, self.verbose)
                return port
            except Exception as e:
                pass

        raise Exception('Rendezvous ERROR: Unable to find a port to bind to.')

    # Rendezvous function finds a available port, create http socket,
    # and start listening loop to handle request
    def rendezvous(self):
        port = self._find_port()

        if self.verbose >= 3:
            print('Rendezvous INFO: HTTP rendezvous server started.')

        # start the listening loop
        self.listen_thread = threading.Thread(target=self.listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()

        return port

    # Listening loop for handle request
    def listen_loop(self):
        while self.httpd.should_continue():
            self.httpd.handle_request()

        if self.verbose >= 3:
            print('Rendezvous INFO: Rendezvous finishes.')
        # Because this thread is daemonized, it can destroy itself

    # Finalize rendezvous server
    def finalize(self):
        while self.listen_thread.is_alive():
            # wait for the listening loop thread to join
            self.listen_thread.join(1)
