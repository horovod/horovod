"""Ray-Horovod Job unit tests.

This is currently not run on the Ray CI.
"""
import os
import sys

import socket
import pytest
import ray
from ray import services
import torch

from horovod.common.util import gloo_built
from horovod.ray.runner import (BaseHorovodWorker, NodeColocator, Coordinator,
                                MiniSettings, RayExecutor)

sys.path.append(os.path.dirname(__file__))


@pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_start_6_cpus():
    address_info = ray.init(num_cpus=6)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_start_4_cpus_4_gpus():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    address_info = ray.init(num_cpus=4, num_gpus=4)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()
    del os.environ["CUDA_VISIBLE_DEVICES"]


def check_resources(original_resources):
    for i in reversed(range(10)):
        if original_resources == ray.available_resources():
            return True
        else:
            print(ray.available_resources())
            import time
            time.sleep(0.5)
    return False


def test_coordinator_registration():
    settings = MiniSettings()
    coord = Coordinator(settings)
    assert coord.world_size == 0
    assert coord.hoststring == ""
    ranks = list(range(12))

    for i, hostname in enumerate(["a", "b", "c"]):
        for r in ranks:
            if r % 3 == i:
                coord.register(hostname, world_rank=r)

    rank_to_info = coord.finalize_registration()
    assert len(rank_to_info) == len(ranks)
    assert all(
        info["HOROVOD_CROSS_SIZE"] == 3 for info in rank_to_info.values())
    assert {info["HOROVOD_CROSS_RANK"]
            for info in rank_to_info.values()} == {0, 1, 2}
    assert all(
        info["HOROVOD_LOCAL_SIZE"] == 4 for info in rank_to_info.values())
    assert {info["HOROVOD_LOCAL_RANK"]
            for info in rank_to_info.values()} == {0, 1, 2, 3}


def test_colocator(tmpdir, ray_start_6_cpus):
    SetColocator = NodeColocator.options(num_cpus=4)
    colocator = SetColocator.remote(
        node_rank=4, num_slots=4, world_size=5, use_gpu=False)
    colocator.create_workers.remote()
    worker_handles = ray.get(colocator.get_workers.remote())
    assert len(set(ray.get(
        [h.hostname.remote() for h in worker_handles]))) == 1

    resources = ray.available_resources()
    ip_address = services.get_node_ip_address()
    assert resources.get("CPU", 0) == 2, resources

    # TODO: https://github.com/horovod/horovod/issues/2438
    # assert resources.get(f"node:{ip_address}", 0) == 1 - 4 * 0.01


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason='GPU colocator test requires 4 GPUs')
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='GPU colocator test requires CUDA')
def test_colocator_gpu(tmpdir, ray_start_4_cpus_4_gpus):
    SetColocator = NodeColocator.options(num_cpus=4, num_gpus=4)
    colocator = SetColocator.remote(
        node_rank=0, num_slots=4, world_size=4, use_gpu=True)
    colocator.create_workers.remote()
    worker_handles = ray.get(colocator.get_workers.remote())
    assert len(set(ray.get(
        [h.hostname.remote() for h in worker_handles]))) == 1
    resources = ray.available_resources()
    ip_address = services.get_node_ip_address()
    assert resources.get("CPU", 0) == 0, resources
    assert resources.get("GPU", 0) == 0, resources

    # TODO: https://github.com/horovod/horovod/issues/2438
    # assert resources.get(f"node:{ip_address}", 0) == 1 - 4 * 0.01, resources

    all_envs = ray.get([h.env_vars.remote() for h in worker_handles])
    all_cudas = {ev["CUDA_VISIBLE_DEVICES"] for ev in all_envs}
    assert len(all_cudas) == 1, all_cudas
    assert len(all_envs[0]["CUDA_VISIBLE_DEVICES"].split(",")) == 4


def test_horovod_mixin(ray_start_2_cpus):
    class Test(BaseHorovodWorker):
        pass

    assert Test().hostname() == socket.gethostname()
    actor = ray.remote(BaseHorovodWorker).remote()
    DUMMY_VALUE = 1123123
    actor.update_env_vars.remote({"TEST": DUMMY_VALUE})
    assert ray.get(actor.env_vars.remote())["TEST"] == str(DUMMY_VALUE)


def test_local(ray_start_4_cpus):
    original_resources = ray.available_resources()
    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(setting, num_hosts=1, num_slots=4)
    hjob.start()
    hostnames = hjob.execute(lambda _: socket.gethostname())
    assert len(set(hostnames)) == 1, hostnames
    hjob.shutdown()
    assert check_resources(original_resources)


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_ray_init(ray_start_4_cpus):
    original_resources = ray.available_resources()

    def simple_fn(worker):
        import horovod.torch as hvd
        hvd.init()
        return hvd.rank()

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting, num_hosts=1, num_slots=4, use_gpu=torch.cuda.is_available())
    hjob.start()
    result = hjob.execute(simple_fn)
    assert len(set(result)) == 4
    hjob.shutdown()
    assert check_resources(original_resources)


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_ray_exec_func(ray_start_4_cpus):
    def simple_fn(num_epochs):
        import horovod.torch as hvd
        hvd.init()
        return hvd.rank() * num_epochs

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting, num_hosts=1, num_slots=4, use_gpu=torch.cuda.is_available())
    hjob.start()
    result = hjob.run(simple_fn, args=[0])
    assert len(set(result)) == 1
    hjob.shutdown()


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_ray_exec_remote_func(ray_start_4_cpus):
    def simple_fn(num_epochs):
        import horovod.torch as hvd
        hvd.init()
        return hvd.rank() * num_epochs

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting, num_hosts=1, num_slots=4, use_gpu=torch.cuda.is_available())
    hjob.start()
    object_refs = hjob.run_remote(simple_fn, args=[0])
    result = ray.get(object_refs)
    assert len(set(result)) == 1
    hjob.shutdown()


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_ray_executable(ray_start_4_cpus):
    class Executable:
        def __init__(self, epochs):
            import horovod.torch as hvd
            self.hvd = hvd
            self.epochs = epochs
            self.hvd.init()

        def rank_epoch(self):
            return self.hvd.rank() * self.epochs

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting, num_hosts=1, num_slots=4, use_gpu=torch.cuda.is_available())
    hjob.start(executable_cls=Executable, executable_args=[2])
    result = hjob.execute(lambda w: w.rank_epoch())
    assert set(result) == {0, 2, 4, 6}
    hjob.shutdown()


def _train(batch_size=32, batch_per_iter=10):
    import torch.backends.cudnn as cudnn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.utils.data.distributed
    from torchvision import models
    import horovod.torch as hvd
    import timeit

    hvd.init()

    # Set up fixed fake data
    data = torch.randn(batch_size, 2)
    target = torch.LongTensor(batch_size).random_() % 2

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters())

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    def benchmark_step():
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    time = timeit.timeit(benchmark_step, number=batch_per_iter)
    return hvd.local_rank()


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_horovod_train(ray_start_4_cpus):
    def simple_fn(worker):
        local_rank = _train()
        return local_rank

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting, num_hosts=1, num_slots=4, use_gpu=torch.cuda.is_available())
    hjob.start()
    result = hjob.execute(simple_fn)
    assert set(result) == {0, 1, 2, 3}
    hjob.shutdown()


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__] + sys.argv[1:]))
