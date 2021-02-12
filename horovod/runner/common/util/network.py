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
# ==============================================================================
import pickle
import psutil
import queue
import socket
import socketserver
import struct
import shutil

import cloudpickle

from horovod.runner.util.threads import in_thread
from horovod.runner.common.util import secret
from horovod.runner.util.network import find_port


class PingRequest(object):
    pass


class NoValidAddressesFound(Exception):
    pass


class PingResponse(object):
    def __init__(self, service_name, source_address):
        self.service_name = service_name
        """Service name that responded to this ping."""
        self.source_address = source_address
        """Source IP address that was visible to the service."""


class AckResponse(object):
    """Used for situations when the response does not carry any data."""
    pass


class AckStreamResponse(object):
    """Used to indicate that stream data follow."""
    pass


class Wire(object):
    """
    Used for serialization/deserialization of objects over the wire.

    We use HMAC to protect services from unauthorized use. The key used for
    the HMAC digest is distributed by Open MPI and Spark.

    The objects are serialized using cloudpickle. Serialized objects become
    the body of the message.

    Structure of the message is as follows:
    - HMAC digest of the body (32 bytes)
    - length of the body (4 bytes)
    - body
    """
    def __init__(self, key):
        self._key = key

    def write(self, obj, wfile):
        message = cloudpickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        digest = secret.compute_digest(self._key, message)
        wfile.write(digest)
        # Pack message length into 4-byte integer.
        wfile.write(struct.pack('i', len(message)))
        wfile.write(message)
        wfile.flush()

    def stream(self, stream, wfile):
        """Transfers data from the utf8 text stream to the wfile until stream exhausts."""
        # we assume utf8 encoding of the stream
        from encodings.utf_8 import StreamWriter
        w = StreamWriter(wfile)
        # do not handle exceptions or close the stream here
        shutil.copyfileobj(stream, w)
        wfile.flush()

    def read(self, rfile):
        digest = rfile.read(secret.DIGEST_LENGTH)
        # Unpack message length into 4-byte integer.
        message_len = struct.unpack('i', rfile.read(4))[0]
        message = rfile.read(message_len)
        if not secret.check_digest(self._key, message, digest):
            raise Exception('Security error: digest did not match the message.')
        return cloudpickle.loads(message)


class BasicService(object):
    def __init__(self, service_name, key, nics):
        self._service_name = service_name
        self._wire = Wire(key)
        self._nics = nics
        self._server, _ = find_port(
            lambda addr: socketserver.ThreadingTCPServer(
                addr, self._make_handler()))
        self._server._block_on_close = True
        self._port = self._server.socket.getsockname()[1]
        self._addresses = self._get_local_addresses()
        self._thread = in_thread(target=self._server.serve_forever)

    def _make_handler(self):
        server = self

        class _Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    req = server._wire.read(self.rfile)
                    resp = server._handle(req, self.client_address)
                    if not resp:
                        raise Exception('Handler did not return a response.')
                    # A tuple is the usual response object followed by a utf8 text stream
                    if type(resp) == tuple:
                        (resp, stream) = resp
                        server._wire.write(resp, self.wfile)
                        server._wire.stream(stream, self.wfile)
                    else:
                        server._wire.write(resp, self.wfile)
                except (EOFError, BrokenPipeError):
                    # Happens when client is abruptly terminated, don't want to pollute the logs.
                    pass

        return _Handler

    def _handle(self, req, client_address):
        """
        Returns the response to be sent to the client.
        Can be a single pickle-able object, or a tuple of such an object and a utf8 text stream.
        """
        if isinstance(req, PingRequest):
            return PingResponse(self._service_name, client_address[0])

        raise NotImplementedError(req)

    def _get_local_addresses(self):
        result = {}
        for intf, intf_addresses in psutil.net_if_addrs().items():
            if self._nics and intf not in self._nics:
                continue
            for addr in intf_addresses:
                if addr.family == socket.AF_INET:
                    if intf not in result:
                        result[intf] = []
                    result[intf].append((addr.address, self._port))
        if not result and self._nics:
            raise NoValidAddressesFound(
                'No available network interface found matching user provided interface: {}'.format(self._nics))
        return result

    def addresses(self):
        return self._addresses.copy()

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def get_port(self):
        return self._port


class BasicClient(object):
    def __init__(self, service_name, addresses, key, verbose, match_intf=False,
                 probe_timeout=20, attempts=3):
        # Note: because of retry logic, ALL RPC calls are REQUIRED to be idempotent.
        self._verbose = verbose
        self._service_name = service_name
        self._wire = Wire(key)
        self._match_intf = match_intf
        self._probe_timeout = probe_timeout
        self._attempts = attempts
        self._addresses = self._probe(addresses)
        if not self._addresses:
            raise NoValidAddressesFound(
                'Horovod was unable to connect to {service_name} on any '
                'of the following addresses: {addresses}.\n\n'
                'One possible cause of this problem is that '
                'horovod currently requires every host to have at '
                'least one routable network interface with the same '
                'name across all of the hosts. '
                'You can run \"ifconfig -a\" '
                'on every host and check for the common '
                'routable interface. '
                'To fix the problem, you can rename interfaces on '
                'Linux.'.format(service_name=service_name, addresses=addresses))

    def _probe(self, addresses):
        result_queue = queue.Queue()
        threads = []
        for intf, intf_addresses in addresses.items():
            for addr in intf_addresses:
                thread = in_thread(target=self._probe_one, args=(intf, addr, result_queue))
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
        for iter in range(self._attempts):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._probe_timeout)
            try:
                sock.connect(addr)
                rfile = sock.makefile('rb')
                wfile = sock.makefile('wb')
                try:
                    self._wire.write(PingRequest(), wfile)
                    resp = self._wire.read(rfile)
                    if resp.service_name != self._service_name:
                        return
                    if self._match_intf:
                        # Interface name of destination and source must match
                        # since `match_intf` is requested.
                        client_intf_addrs = [x.address
                                             for x in psutil.net_if_addrs().get(intf, [])
                                             if x.family == socket.AF_INET]
                        if resp.source_address not in client_intf_addrs:
                            if self._verbose >= 2:
                                # Need to find the local interface name whose
                                # address was visible to the target host's server.
                                resp_intf = ''
                                for key in psutil.net_if_addrs().keys():
                                    key_intf_addrs = [x.address
                                                      for x in psutil.net_if_addrs().get(key, [])]
                                    if resp.source_address in key_intf_addrs:
                                        resp_intf = key
                                        break
                                print('WARNING: Expected to connect the host '
                                      '{addr} using interface '
                                      '{intf}, but reached it on interface '
                                      '{resp_intf}.'.format(
                                    addr=str(addr[0])+':'+str(addr[1]),
                                    intf=intf,
                                    resp_intf=resp_intf))
                            return
                    result_queue.put((intf, addr))
                    return
                finally:
                    rfile.close()
                    wfile.close()
            except:
                pass
            finally:
                sock.close()

    def _send_one(self, addr, req, stream=None):
        """
        Send the request to the server and retry on errors.
        Streams data that follow a AckStreamResponse to the given utf8 text stream.
        """
        for iter in range(self._attempts):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect(addr)
                rfile = sock.makefile('rb')
                wfile = sock.makefile('wb')
                try:
                    self._wire.write(req, wfile)
                    resp = self._wire.read(rfile)
                    if stream and isinstance(resp, AckStreamResponse):
                        # stream and byte content in rfile are expected to be utf8 text
                        from encodings.utf_8 import StreamReader
                        r = StreamReader(rfile)
                        shutil.copyfileobj(r, stream)
                    return resp
                finally:
                    rfile.close()
                    wfile.close()
            except:
                if iter == self._attempts - 1:
                    # Raise exception on the last retry.
                    raise
            finally:
                sock.close()

    def _send(self, req, stream=None):
        """
        Sends the request and returns the response object.
        Streaming data response is transferred to the optional stream parameter.
        """
        # Since all the addresses were vetted, use the first one.
        addr = list(self._addresses.values())[0][0]
        return self._send_one(addr, req, stream)

    def addresses(self):
        return self._addresses
