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

# Timeout for the server listening for new request
TOTAL_TIMEOUT = 60


class MyHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

    # Set timeout
    timeout = SINGLE_REQUEST_TIMEOUT

    # Override POST HTTP request handler
    def do_POST(self):
        paths = self.path.split('/')

        if len(paths) < 3:
            raise Exception('Invalid request path: {path}.'.format(path=self.path))
        if self.server.verbose >= 5:
            print('Receive reqeust to url ' + self.path)

        if paths[2] == 'get':
            self.handle_get(paths[1])
        elif paths[2] == 'set':
            self.handle_set(paths[1])
        else:
            raise Exception('Unsupported request type: {type}.'.format(type=paths[2]))

    def read_from_socket(self, sock_input, content_bytes):
        try:
            value = sock_input.read(content_bytes)
            return value
        except socket.timeout:
            if self.server.verbose >= 2:
                print('Rendezvous WARNING: Timeout when receiving {content_bytes} '
                      'bytes, aborting this incomplete request.'
                      .format(content_bytes=content_bytes))
            return bytes('')

    def handle_get(self, namespace):
        key_len = int(self.headers.get('Key-Length', '-1'))
        if key_len < 0:
            raise Exception('Invalid Get request with no Key-Length header line.')
        key = self.read_from_socket(self.rfile, key_len)
        # if not key:
        #     print('Request does not have Key-Length header line, response 408.')
        #     self.send_response(408)
        #     self.end_headers()
        #     return

        if self.server.verbose >= 5:
            rank = self.headers.get('Rank')
            print('Receive a get request from {namespace}:{rank} with key: {key}.'
                  .format(namespace=namespace, rank=rank, key=key))

        with self.server.cache_lock:
            value = self.server.cache.get(namespace, {}).get(key, '')

        self.send_response(200)
        self.send_header("Content-Length", str(len(value)))
        self.end_headers()
        self.wfile.write(value)
        self.wfile.flush()

        if self.server.verbose >= 5:
            rank = self.headers.get('Rank')
            print('Respond a get request from {namespace}:{rank} with value length {val_len}.'
                  .format(namespace=namespace, rank=rank, val_len=len(value)))

    def handle_set(self, namespace):
        key_len = int(self.headers.get('Key-Length', '-1'))
        if key_len < 0:
            raise Exception('Invalid Set request with no Key-Length header line.')

        key = self.read_from_socket(self.rfile, key_len)

        value_len = int(self.headers.get('Value-Length', '-1'))
        if value_len < 0:
            raise Exception('Invalid Set request with no Value-Length header line.')
        if self.server.verbose >= 5:
            rank = self.headers.get('Rank')
            print('Receive a set request from {namespace}:{rank} with key: {key}, '
                  'value length: {val_len}.'
                  .format(namespace=namespace, rank=rank, key=key, val_len=value_len))

        value = self.read_from_socket(self.rfile, value_len)

        # if not key or not value:
        #     print()
        #     self.send_response(408)
        #     self.end_headers()
        #     return

        self.send_response(200)
        self.send_header("Content-Length", 0)
        self.end_headers()

        with self.server.cache_lock:
            namespace_dict = self.server.cache.setdefault(namespace, {})
            namespace_dict[key] = value
            if self.server.verbose >= 5:
                print(self.server.cache.keys())

    # TODO: do we need finish here?
    # def handle_finish(self, namespace):
    #     if self.server.verbose >= 5:
    #         rank = self.headers.get('Rank')
    #         print('Receive a set request from {namespace}:{rank} with key: {key}, '
    #               'value length: {val_len}.'
    #               .format(namespace=namespace, rank=rank, key=key, val_len=value_len))
    #     self.send_response(200)
    #     self.end_headers()
    #
    #     with self.server.cache_lock:
    #         namespace_dict = self.server.cache.setdefault(namespace, {})
    #         namespace_dict[key] = value

    # Override this function to prevent SimpleHTTPServer printing every request out.
    def log_message(self, format, *args):
        pass


class MyHTTPServer(BaseHTTPServer.HTTPServer, object):

    def __init__(self, addr, handler, size, verbose):
        super(MyHTTPServer, self).__init__(addr, handler)
        # Total size of the context
        self.size = size
        self.httpd = None

        # Count for finished rendezvous workers
        self.finished_cnt_lock = threading.Lock()
        self.finished_cnt = collections.defaultdict(int)

        # Cache that provides the store
        self.cache_lock = threading.Lock()
        self.cache = {}

        self.verbose = verbose

    # Decide whether all ranks have confirmed rendezvous completion.
    def should_continue(self):
        with self.finished_cnt_lock:
            finished_cnt = self.finished_cnt
        return finished_cnt < self.size

    def handle_timeout(self):
        raise Exception('Rendezvous TIMEOUT: Rendezvous server timeout after '
                        '{time} seconds while waiting for all the ranks to send finished '
                        'messages. Received {recv}, expecting {expect}. '
                        'Please make sure number of process set by -np is correct.'
                        .format(time=TOTAL_TIMEOUT, recv=self.finished_cnt, expect=self.size))


class RendezvousServer:
    def __init__(self, size, verbose):
        self.httpd = None
        self.listen_thread = None
        self.size = size
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
                self.httpd = MyHTTPServer(addr, MyHandler, self.size, self.verbose)
                return port
            except Exception as e:
                pass

        raise Exception('Rendezvous ERROR: Unable to find a port to bind to.')

    # Rendezvous function finds a available port, create http socket,
    # and start listening loop to handle request
    def rendezvous(self):
        port = self._find_port()
        self.httpd.timeout = TOTAL_TIMEOUT

        if self.verbose >= 3:
            print('Rendezvous INFO: HTTP rendezvous server started.')

        # start the listening loop
        self.listen_thread = threading.Thread(target=self.listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()

        return port

    # Listening loop for handle request
    def listen_loop(self):
        # while self.httpd.should_continue():
        while True:
            self.httpd.handle_request()
        if self.verbose >= 3:
            print('Rendezvous INFO: Rendezvous finishes.')

    # Finalize rendezvous server
    def finalize(self):
        while self.listen_thread.is_alive():
            # wait for the listening loop thread to join
            self.listen_thread.join(1)


# Test
if __name__ == '__main__':

    import requests
    rendez = RendezvousServer(2, 5)
    port = rendez.rendezvous()
    print(port)

    headers = {'Key-Length': str(6), 'Value-Length':str(6), 'Finished':str(0)}
    requests.post(url='http://localhost:'+str(port)+'/global/set',
                        data='asdada123456', headers=headers)
    headers = {'Key-Length': str(6), 'Value-Length':str(0), 'Finished':str(0)}
    res = requests.post(url='http://localhost:'+str(port)+'/global/get',
                        data='asdada', headers=headers)
    assert(res.content == '123456')
    headers = {'Key-Length': str(6), 'Value-Length':str(6), 'Finished':str(0)}
    requests.post(url='http://localhost:'+str(port)+'/local/set',
                  data='asdada654321', headers=headers)
    headers = {'Key-Length': str(6), 'Value-Length':str(0), 'Finished':str(0)}
    res = requests.post(url='http://localhost:'+str(port)+'/local/get',
                        data='asdada', headers=headers)
    assert(res.content == '654321')
    headers = {'Key-Length': str(6), 'Value-Length':str(0), 'Finished':str(0)}
    res = requests.post(url='http://localhost:'+str(port)+'/local/get',
                        data='asdada', headers=headers)
    assert(res.content == '654321')
    headers = {'Key-Length': str(6), 'Value-Length':str(0), 'Finished':str(0)}
    res = requests.post(url='http://localhost:'+str(port)+'/local/get',
                        data='asdada', headers=headers)
    assert(res.content == '654321')
    headers = {'Key-Length': str(6), 'Value-Length':str(0), 'Finished':str(0)}
    res = requests.post(url='http://localhost:'+str(port)+'/global/get',
                        data='asdada', headers=headers)
    assert(res.content == '123456')
    # headers = {'Key-Length': str(6), 'Value-Length':str(0), 'Finished':str(1)}
    # res = requests.post(url='http://localhost:'+str(port),
    #                     data='asdbda', headers=headers)
    # print('Got response:', res.content)
    # headers = {'Key-Length': str(6), 'Value-Length':str(0), 'Finished':str(0)}
    # res = requests.post(url='http://localhost:'+str(port),
    #                     data='asdbda', headers=headers)
    # print('Got response:', res.content)
    # headers = {'Key-Length': str(6), 'Value-Length':str(0), 'Finished':str(1)}
    # res = requests.post(url='http://localhost:'+str(port),
    #                     data='asdbda', headers=headers)
    # print('Got response:', res.content)

