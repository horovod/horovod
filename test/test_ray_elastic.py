"""Ray-Horovod Elastic training unit tests.

This is currently not run on the Ray CI.
"""
from contextlib import contextmanager
import psutil
import os

import mock
import pytest
import ray

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
    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_cpu_discovery(self, ray_shutdown):
        ray.init(num_cpus=4, num_gpus=1)
        discovery = RayHostDiscovery(cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 1
        assert list(mapping.values()) == [4]

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_gpu_discovery(self, ray_shutdown):
        ray.init(num_cpus=4, num_gpus=1)
        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 1
        assert list(mapping.values()) == [1]

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_gpu_slot_discovery(self, ray_shutdown):
        ray.init(num_cpus=4, num_gpus=4)
        discovery = RayHostDiscovery(
            use_gpu=True, cpus_per_slot=1, gpus_per_slot=2)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 1
        assert list(mapping.values()) == [2]

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
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

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_multinode_gpus_per_slot(self, monkeypatch):
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
        discovery = RayHostDiscovery(use_gpu=True, gpus_per_slot=2)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 3
        assert list(mapping.values()) == [1, 1, 1]

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
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


class SimpleTestDiscovery(HostDiscovery):
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


def _create_training_function(iterations):
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
        import time
        import torch
        import horovod.torch as hvd

        hvd.init()

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        logger.log.remote(("started", os.getpid()))

        @hvd.elastic.run
        def train(state):
            for state.epoch in range(state.epoch, iterations):
                logger.log.remote(("training", os.getpid()))
                time.sleep(0.1)
                state.commit()  # triggers scale-up, scale-down
            logger.log.remote(("finished", os.getpid()))

        state = hvd.elastic.TorchState(
            model, optimizer, batch=0, epoch=0, commits=0, rendezvous=0)
        train(state)
        return True

    return logger, training_fn


@contextmanager
def fault_tolerance_patches():
    with mock.patch(
            'horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS',
            0.1):
        with mock.patch(
                "horovod.runner.util.network.get_driver_ip",
                return_value=ray.services.get_node_ip_address()):
            yield


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_fault_tolerance_hosts_added_and_removed(ray_8_cpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (20, ['host-1:2']),
            (60, ['host-1:2', 'host-2:1', 'host-3:1']),
            (None, ['host-2:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(min_np=1, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, cpus_per_slot=1, override_discovery=False)

        logger, training_fn = _create_training_function(iterations=100)
        executor.start()
        results = executor.run(training_fn)
        assert len(results) == 1

        events = ray.get(logger.fetch.remote())
        assert sum(int("started" in e) for e in events) == 4, events
        assert sum(int("finished" in e) for e in events) == 1, events


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_fault_tolerance_hosts_remove_and_add(ray_8_cpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (60, ['host-1:2', 'host-2:1', 'host-3:2']),
            (20, ['host-1:2']),
            (None, ['host-1:2', 'host-4:1', 'host-5:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(min_np=1, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, cpus_per_slot=1, override_discovery=False)

        logger, training_fn = _create_training_function(iterations=100)
        executor.start()
        results = executor.run(training_fn)
        assert len(results) == 4

        events = ray.get(logger.fetch.remote())
        assert sum(int("started" in e) for e in events) == 7, events
        assert sum(int("finished" in e) for e in events) == 4, events


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_max_np(ray_8_cpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (20, ['host-1:2']),
            (None, ['host-1:2', 'host-4:1', 'host-5:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(
            min_np=1, max_np=2, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, cpus_per_slot=1, override_discovery=False)

        logger, training_fn = _create_training_function(iterations=100)
        executor.start()
        results = executor.run(training_fn)
        assert len(results) == 2

        events = ray.get(logger.fetch.remote())
        assert sum(int("started" in e) for e in events) == 2, events
        assert sum(int("finished" in e) for e in events) == 2, events


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_min_np(ray_8_cpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (20, ['host-1:1']),
            (60, ['host-1:1', 'host-4:1', 'host-5:1']),
            (None, ['host-1:1', 'host-4:1', 'host-5:1', 'host-6:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(
            min_np=4, max_np=4, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, cpus_per_slot=1, override_discovery=False)

        logger, training_fn = _create_training_function(iterations=100)
        executor.start()
        results = executor.run(training_fn)
        assert len(results) == 4

        events = ray.get(logger.fetch.remote())
        assert sum(int("started" in e) for e in events) == 4, events
        assert sum(int("finished" in e) for e in events) == 4, events


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_gpu_e2e(ray_8_cpus_gpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (20, ['host-1:1']),
            (60, ['host-1:1', 'host-4:1', 'host-5:1']),
            (None, ['host-1:1', 'host-4:1', 'host-5:1', 'host-6:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(
            min_np=4, max_np=4, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, gpus_per_slot=1, use_gpu=True, override_discovery=False)

        logger, training_fn = _create_training_function(iterations=100)
        executor.start()
        results = executor.run(training_fn)
        assert len(results) == 4

        events = ray.get(logger.fetch.remote())
        assert sum(int("started" in e) for e in events) == 4, events
        assert sum(int("finished" in e) for e in events) == 4, events


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(sys.argv[1:] + ["-v", "-x", __file__]))
