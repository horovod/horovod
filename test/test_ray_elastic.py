"""Ray-Horovod Elastic training unit tests.

This is currently not run on the Ray CI.
"""
import os

import pytest
import ray
import torch

from horovod.common.util import gloo_built
from horovod.ray.elastic import (
    ElasticRayExecutor, RayDiscovery)



@pytest.fixture
def ray_shutdown():
    yield
    # The code after the yield will run as teardown code.
    ray.shutdown()

class RayDiscoveryTestSuite:

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
        monkeypatch.setattr(ray, "nodes", self.create_multi_node_mock)
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
        monkeypatch.setattr(ray, "nodes", self.create_multi_node_mock)
        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert sum(mapping.values()) == 0


def test_auto_scale_up(mock_get_min_start_hosts):
    discovery_schedule = [
        (0, ['host-1:1']),
        (1, ['host-1:1', 'host-2:1', 'host-3:1']),
    ]
    executor = ElasticRayExecutor()
    executor.start()
    results = executor.run()
    assert len(results) == 3

def test_fault_tolerance_hosts_added_and_removed(nit):
    discovery_schedule = [
        (0, ['host-1:1']),
        (1, ['host-1:1', 'host-2:1', 'host-3:1']),
        (None, ['host-2:1']),
    ]

    @ray.remote(num_cpus=0)
    class Logger:
        def __init__(self):
            self._journal = []

        def log(self, info):
            self._journal.append(info)

        def fetch(self):
            return self._journal

    logger = Logger.remote()

    def training_fn():
        logger.log.remote(("started", hvd.rank()))
        return True
    executor = ElasticRayExecutor()
    executor.start()
    results = executor.run(training_fn)
    assert len(results) == 1

    events = ray.get(logger.fetch.remote())
    assert len([e for e in events if "started" in e]) == 3
