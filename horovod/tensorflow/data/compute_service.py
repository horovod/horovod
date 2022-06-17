# Copyright 2022 G-Research. All Rights Reserved.
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

import binascii
import dataclasses
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping, Sequence, Tuple, Any, Optional

import tensorflow as tf

import horovod.tensorflow as hvd
from horovod.runner.common.service.compute_service import ComputeClient


@dataclasses.dataclass(frozen=True)
class TfDataServiceConfig:
    dispatchers: int
    workers_per_dispatcher: int
    dispatcher_side: str
    addresses: Mapping[str, Sequence[Tuple[str, int]]]
    key: bytes
    timeout: int = 60

    def compute_client(self, verbose=1) -> ComputeClient:
        return ComputeClient(self.addresses, self.key, verbose=verbose)

    def to_dict(self) -> Mapping[str, Any]:
        config = self.__dict__.copy()
        config['key'] = binascii.hexlify(config.get('key')).decode()
        return config

    @staticmethod
    def from_dict(config: Mapping[str, Any]) -> 'TfDataServiceConfig':
        config = dict(**config)
        config['key'] = binascii.unhexlify(config.get('key'))
        config['addresses'] = {intf: [(addr[0], addr[1]) for addr in addrs]
                               for intf, addrs in config.get('addresses').items()}

        return TfDataServiceConfig(
            dispatchers=config.get('dispatchers'),
            workers_per_dispatcher=config.get('workers_per_dispatcher'),
            dispatcher_side=config.get('dispatcher_side'),
            addresses=config.get('addresses'),
            key=config.get('key'),
            timeout=config.get('timeout')
        )

    def write(self, filename: str):
        path = Path(filename)
        with NamedTemporaryFile('w', dir=str(path.parent), prefix=str(path.name), delete=False) as w:
            # write the complete config into a different file first
            logging.info(f'Writing config to {w.name}')
            w.write(json.dumps(self.to_dict()))

            # move the finished config file into place, this happens inside the same directory so it should be quick
            logging.info(f'Renaming config from {w.name} to {filename}')
            os.rename(w.name, filename)

    @staticmethod
    def read(filename: str, wait_for_file_creation: bool = False) -> 'TfDataServiceConfig':
        while wait_for_file_creation:
            if os.path.exists(filename):
                break
            time.sleep(1)

        with open(filename, 'r') as r:
            return TfDataServiceConfig.from_dict(json.load(r))


@contextmanager
def tf_data_service(compute_config: TfDataServiceConfig, rank: int) -> str:
    """
    Provides the address of the TF Dispatcher.
    """

    compute = compute_config.compute_client(verbose=2)

    dispatcher_server = None
    if compute_config.dispatcher_side == 'training':
        if compute_config.dispatchers > 1 or compute_config.dispatchers == 1 and rank == 0:
            if compute_config.dispatchers == 1:
                logging.info(f"Setting up Dispatcher for all tasks")
            else:
                logging.info(f"Setting up Dispatcher for task {rank}")

            dispatcher_server = tf.data.experimental.service.DispatchServer()
            logging.debug(f"Registering Dispatcher {rank} at {dispatcher_server.target}")
            compute.register_dispatcher(rank, dispatcher_server.target)
            logging.info(f"Registered Dispatcher {rank} at {dispatcher_server.target}")

    dispatcher_id = rank if compute_config.dispatchers > 1 else 0
    dispatcher_address = compute.wait_for_dispatcher_registration(dispatcher_id, compute_config.timeout)
    compute.wait_for_dispatcher_worker_registration(dispatcher_id, compute_config.timeout)

    # let the caller use the dispatcher
    try:
        yield dispatcher_address
    finally:
        if dispatcher_server:
            # there is currently no other way to stop the dispatch server
            logging.debug(f"Shuting down dispatcher")
            dispatcher_server._stop()
            dispatcher_server.join()
            logging.info(f"Dispatcher shut down")


def send_to_data_service(dataset: tf.data.Dataset,
                         compute_config: TfDataServiceConfig,
                         rank: int,
                         size: Optional[int] = None,
                         processing_mode: str = 'distributed_epoch',
                         reuse_dataset: bool = False,
                         round_robin: bool = False) -> tf.data.Dataset:
    if compute_config.dispatcher_side == 'training':
        raise RuntimeError('training side dispatcher not supported, use tf_data_service context manager instead')

    with tf_data_service(compute_config, rank) as dispatcher_address:
        return dataset.apply(tf.data.experimental.service.distribute(
            processing_mode=processing_mode,
            service=dispatcher_address,
            job_name='job' if reuse_dataset else None,
            consumer_index=rank if reuse_dataset and round_robin else None,
            num_consumers=size if reuse_dataset and round_robin else None))


tf.data.Dataset.send_to_data_service = send_to_data_service


def compute_worker_fn(compute_config: TfDataServiceConfig):
    """ Function run on the compute tasks providing tf dispatcher and worker server. """
    hvd.init()
    index, size = hvd.rank(), hvd.size()
    dispatcher_index = index // compute_config.workers_per_dispatcher

    compute = compute_config.compute_client(verbose=2)

    import tensorflow as tf

    # Create dispatcher for train task
    dispatcher_server = None
    if compute_config.dispatcher_side == 'compute' and index % compute_config.workers_per_dispatcher == 0:
        if compute_config.dispatchers == 1:
            logging.info(f"Setting up Dispatcher for all tasks")
        else:
            logging.info(f"Setting up Dispatcher for task {dispatcher_index}")

        dispatcher_server = tf.data.experimental.service.DispatchServer()
        logging.debug(f"Registering Dispatcher {dispatcher_index} at {dispatcher_server.target}")
        compute.register_dispatcher(dispatcher_index, dispatcher_server.target)
        logging.info(f"Registered Dispatcher {dispatcher_index} at {dispatcher_server.target}")

    # Get dispatcher for the worker
    logging.debug(f'Waiting for dispatcher {dispatcher_index} for worker {index}')
    dispatcher_address = compute.wait_for_dispatcher_registration(dispatcher_index, compute_config.timeout)
    logging.debug(f'Dispatcher {dispatcher_index} for worker {index} available')

    # Create worker
    logging.debug(f"Setting up worker for dispatcher {dispatcher_index}")
    worker_config = tf.data.experimental.service.WorkerConfig(
        dispatcher_address=dispatcher_address.split("://")[1],
        heartbeat_interval_ms=1000,
        dispatcher_timeout_ms=compute_config.timeout * 1000)
    worker_server = tf.data.experimental.service.WorkerServer(worker_config)
    logging.debug(f"Starting worker for dispatcher {dispatcher_index}")
    worker_server.start()
    logging.debug(f"Started worker for dispatcher {dispatcher_index}")

    # Tell the compute service that we are ready
    logging.debug(f"Registering worker for dispatcher {dispatcher_index}")
    compute.register_worker_for_dispatcher(dispatcher_index, index)
    logging.info(f"Worker for dispatcher {dispatcher_index} registered")

    # Wait until the compute service shuts down
    logging.debug(f"Waiting for shutdown request")
    compute.wait_for_shutdown()
    logging.debug(f"Shutdown requested")

    # stop the servers
    # there is currently no other way to stop the worker server
    logging.debug(f"Shuting down worker")
    worker_server._stop()
    worker_server.join()
    logging.info(f"Worker shut down")
    if dispatcher_server:
        # there is currently no other way to stop the dispatch server
        logging.debug(f"Shuting down dispatcher")
        dispatcher_server._stop()
        dispatcher_server.join()
        logging.info(f"Dispatcher shut down")
