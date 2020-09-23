"""Ray-Horovod Elastic training unit tests.

This is currently not run on the Ray CI.
"""
import logging
logging.basicConfig(level=logging.DEBUG)
import os

import mock
import pytest
import ray
import torch

from horovod.common.util import gloo_built
from horovod.runner.elastic.discovery import HostDiscovery
from horovod.ray.elastic import ElasticRayExecutor, RayHostDiscovery



@pytest.fixture
def ray_shutdown():
    yield
    # The code after the yield will run as teardown code.
    ray.shutdown()

@pytest.fixture
def ray_8_cpus():
    ray.init(num_cpus=8)
    yield
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_8_cpus_gpus():
    ray.init(num_cpus=8, num_gpus=8)
    yield
    # The code after the yield will run as teardown code.
    ray.shutdown()

class TestRayDiscoverySuite:
    def test_cpu_discovery(self, ray_shutdown):
        ray.init(num_cpus=4, num_gpus=1)
        discovery = RayHostDiscovery(cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 1
        assert list(mapping.values()) == [4]

    def test_gpu_discovery(self, ray_shutdown):
        ray.init(num_cpus=4, num_gpus=1)
        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 1
        assert list(mapping.values()) == [1]

    def test_multinode(self, monkeypatch):
        def create_multi_node_mock():
            host_names = ["host-1", "host-2", "host-3"]
            resources = {"GPU": 2, "CPU": 8}
            def create_node_entry(hostname):
                return {
                    "NodeManagerAddress": hostname,
                    "Resources": resources.copy(),
                    "alive": True
                }
            return map(create_node_entry, host_names)
        monkeypatch.setattr(ray, "nodes", create_multi_node_mock)
        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 3
        assert list(mapping.values()) == [2, 2, 2]

    def test_multinode_mismatch(self, monkeypatch):
        def create_multi_node_mock():
            host_names = ["host-1", "host-2", "host-3"]
            resources = {"CPU": 8}
            def create_node_entry(hostname):
                return {
                    "NodeManagerAddress": hostname,
                    "Resources": resources.copy(),
                    "alive": True
                }
            return map(create_node_entry, host_names)
        monkeypatch.setattr(ray, "nodes", create_multi_node_mock)
        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert sum(mapping.values()) == 0


# def test_auto_scale_up(mock_get_min_start_hosts):
#     discovery_schedule = [
#         (0, ['host-1:1']),
#         (1, ['host-1:1', 'host-2:1', 'host-3:1']),
#     ]
#     executor = ElasticRayExecutor()
#     executor.start()
#     results = executor.run()
#     assert len(results) ==

class TestDiscovery(HostDiscovery):
    def __init__(self, schedule):
        self._schedule = schedule
        self._generator = self.host_generator()

    def host_generator(self):
        for iters, hosts in self._schedule:
            iters = iters or 500  # max
            for i in range(iters):
                yield hosts

    def find_available_hosts_and_slots(self):
        hostlist = next(self._generator)
        from ray.experimental.dynamic_resources import set_resource
        hosts = {}
        for item in hostlist:
            host, slots = item.split(":")
            slots = int(slots)
            set_resource(f"node:{host}", 1)
            hosts[host] = slots
        return hosts


@mock.patch('horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.1)
@mock.patch("horovod.runner.util.network.get_driver_ip", return_value=ray.services.get_node_ip_address())
def run_test_fault_tolerance_hosts_added_and_removed(driver_ip_mock):
    logging.basicConfig(level="DEBUG")
    discovery_schedule = [
        (20, ['host-1:2']),
        (100, ['host-1:2', 'host-2:1', 'host-3:1']),
        (None, ['host-2:1']),
    ]
    @ray.remote(num_cpus=0)
    class Logger:
        def __init__(self):
            self._journal = []

        def log(self, info):
            print(f"Got {info}")
            self._journal.append(info)

        def fetch(self):
            return self._journal

    logger = Logger.remote()

    def training_fn():
        import time
        import torch
        import horovod.torch as hvd

        hvd.init()

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        logger.log.remote(("started", os.getpid()))

        @hvd.elastic.run
        def train(state):
            for i in range(100):
                logger.log.remote(("training", os.getpid()))
                time.sleep(1)
            logger.log.remote(("finished", os.getpid()))


        state = hvd.elastic.TorchState(model, optimizer, batch=0, epoch=0, commits=0, rendezvous=0)
        train(state)
        return True

    settings = ElasticRayExecutor.create_settings(min_np=4, verbose=2, nics={"ens3"})  # todo: determine nic later
    settings.discovery = TestDiscovery(discovery_schedule)
    executor = ElasticRayExecutor(settings, cpus_per_slot=1, override_discovery=False)
    executor.start()
    results = executor.run(training_fn)
    assert len(results) == 1

    events = ray.get(logger.fetch.remote())
    assert len([e for e in events if "started" in e]) == 3

def test_fault_tolerance_hosts_added_and_removed(ray_8_cpus):
    run_test_fault_tolerance_hosts_added_and_removed()

if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(sys.argv[1:] + ["-v", "-x", __file__]))