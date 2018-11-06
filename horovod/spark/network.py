# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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
# ==============================================================================

import cloudpickle
from six.moves import queue
import psutil
import random
import socket
import SocketServer
import threading


class PingRequest(object):
    pass


class PingResponse(object):
    def __init__(self, service_name):
        self.service_name = service_name


class BasicService(object):
    def __init__(self, service_name):
        self._service_name = service_name
        self._server = self._make_server()
        self._port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True
        self._thread.start()

    # TODO: make port range configurable
    def _make_server(self):
        min_port = 1024
        max_port = 65536
        num_ports = max_port - min_port
        start_port = random.randrange(0, num_ports)
        for port_offset in range(num_ports):
            try:
                port = min_port + (start_port + port_offset) % num_ports
                return SocketServer.ThreadingTCPServer(('0.0.0.0', port), self._make_handler())
            except:
                pass

        raise Exception('Unable to find a port to bind to.')

    # TODO: add TLS with cert based auth
    def _make_handler(self):
        server = self

        class _Handler(SocketServer.StreamRequestHandler):
            def handle(self):
                try:
                    req = cloudpickle.load(self.rfile)
                    resp = server._handle(req, self.client_address)
                    if not resp:
                        raise Exception('Handler did not return a response.')
                    cloudpickle.dump(resp, self.wfile)
                except EOFError:
                    # Happens when client is abruptly terminated, don't want to pollute the logs.
                    # TODO: consider putting all these in the debug logs.
                    pass

        return _Handler

    def _handle(self, req, client_address):
        if isinstance(req, PingRequest):
            return PingResponse(self._service_name)

        raise NotImplementedError(req)

    def addresses(self):
        result = {}
        for intf, intf_addresses in psutil.net_if_addrs().items():
            for addr in intf_addresses:
                if addr.family == socket.AF_INET:
                    if intf not in result:
                        result[intf] = []
                    result[intf].append((addr.address, self._port))
        return result

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


class BasicClient(object):
    def __init__(self, service_name, addresses, probe_timeout=20, retries=3):
        # Note: because of retry logic, ALL RPC calls are REQUIRED to be idempotent.
        self._service_name = service_name
        self._probe_timeout = probe_timeout
        self._retries = retries
        self._addresses = self._probe(addresses)
        if not self._addresses:
            raise Exception('Unable to connect to the %s on any of the addresses: %s'
                            % (service_name, addresses))

    def _probe(self, addresses):
        result_queue = queue.Queue()
        threads = []
        for intf, intf_addresses in addresses.items():
            for addr in intf_addresses:
                thread = threading.Thread(target=self._probe_one,
                                          args=(intf, addr, result_queue))
                thread.daemon = True
                thread.start()
                threads.append(thread)
        for t in threads:
            t.join()

        result = {}
        while not result_queue.empty():
            intf, addr = result_queue.get()
            if intf not in result:
                result[intf] = []
            result[intf].append(addr)
        return result

    def _probe_one(self, intf, addr, result_queue):
        for iter in range(self._retries):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._probe_timeout)
            try:
                sock.connect(addr)
                rfile = sock.makefile('rb', -1)
                wfile = sock.makefile('wb', 0)
                try:
                    cloudpickle.dump(PingRequest(), wfile)
                    wfile.flush()
                    resp = cloudpickle.load(rfile)
                    if resp.service_name == self._service_name:
                        result_queue.put((intf, addr))
                    return
                finally:
                    rfile.close()
                    wfile.close()
            except:
                pass
            finally:
                sock.close()

    def _send_one(self, addr, req):
        for iter in range(self._retries):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect(addr)
                rfile = sock.makefile('rb', -1)
                wfile = sock.makefile('wb', 0)
                try:
                    cloudpickle.dump(req, wfile)
                    wfile.flush()
                    resp = cloudpickle.load(rfile)
                    return resp
                finally:
                    rfile.close()
                    wfile.close()
            except:
                if iter == self._retries - 1:
                    # Raise exception on the last retry.
                    raise
            finally:
                sock.close()

    def _send(self, req):
        # Since all the addresses were vetted, use the first one.
        addr = self._addresses.values()[0][0]
        return self._send_one(addr, req)

    def addresses(self):
        return self._addresses
