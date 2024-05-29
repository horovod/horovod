"""Ray-Horovod Job unit tests.

This is currently not run on the Ray CI.
"""
import os
import socket
import sys

import pytest
import ray
import torch
from ray.util.client.ray_client_helpers import ray_start_client_server

from horovod.common.util import gloo_built
from horovod.ray.runner import (Coordinator, MiniSettings, RayExecutor)
from horovod.ray.strategy import create_placement_group
from horovod.ray.worker import BaseHorovodWorker

sys.path.append(os.path.dirname(__file__))


@pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4, _redis_max_memory=1024 * 1024 * 1024)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_start_6_cpus():
    address_info = ray.init(num_cpus=6)
    try:
        yield address_info
    finally:
        # The code after the yield will run as teardown code.
        ray.shutdown()


@pytest.fixture
def ray_start_4_cpus_4_gpus():
    orig_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    address_info = ray.init(num_cpus=4, num_gpus=4)
    try:
        yield address_info
        # The code after the yield will run as teardown code.
        ray.shutdown()
    finally:
        if orig_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig_devices
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]


@pytest.fixture
def ray_start_client():
    def ray_connect_handler(job_config=None, *vargs, **kwargs):
        # Ray client will disconnect from ray when
        # num_clients == 0.
        if ray.is_initialized():
            return
        else:
            return ray.init(job_config=job_config, num_cpus=4)

    assert not ray.util.client.ray.is_connected()
    with ray_start_client_server(ray_connect_handler=ray_connect_handler):
        yield


def test_coordinator_registration():
    settings = MiniSettings()
    coord = Coordinator(settings)
    assert coord.world_size == 0
    assert coord.node_id_string == ""
    ranks = list(range(12))

    for i, hostname in enumerate(["a", "b", "c"]):
        for r in ranks:
            if r % 3 == i:
                coord.register(hostname, node_id=str(i), world_rank=r)

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


@pytest.mark.parametrize("use_same_host", [True, False])
def test_cross_rank(use_same_host):
    settings = MiniSettings()
    coord = Coordinator(settings)
    assert coord.world_size == 0
    assert coord.node_id_string == ""
    ranks = list(range(12))

    for r in ranks:
        if r < 5:
            coord.register("host1", node_id="host1", world_rank=r)
        elif r < 9:
            coord.register(
                "host1" if use_same_host else "host2",
                node_id="host2",
                world_rank=r)
        else:
            coord.register(
                "host1" if use_same_host else "host3",
                node_id="host3",
                world_rank=r)

    rank_to_info = coord.finalize_registration()
    assert len(rank_to_info) == len(ranks)
    # check that there is only 1 rank with cross_size == 1, cross_rank == 0
    cross_size_1 = list(info for info in rank_to_info.values()
                        if info["HOROVOD_CROSS_SIZE"] == 1)
    assert len(cross_size_1) == 1
    assert cross_size_1[0]["HOROVOD_CROSS_RANK"] == 0
    # check that there is only 2 rank with cross_size == 2
    cross_size_2 = list(info for info in rank_to_info.values()
                        if info["HOROVOD_CROSS_SIZE"] == 2)
    assert len(cross_size_2) == 2

    # check that if cross_size == 2, set(cross_rank) == 0,1
    assert set(d["HOROVOD_CROSS_RANK"] for d in cross_size_2) == {0, 1}

    # check that there is 9 rank with cross_size = 3
    cross_size_3 = list(info for info in rank_to_info.values()
                        if info["HOROVOD_CROSS_SIZE"] == 3)
    assert len(cross_size_3) == 9


# Used for Pytest parametrization.
parameter_str = "num_workers,num_hosts,num_workers_per_host"
ray_executor_parametrized = [(4, None, None), (None, 1, 4)]


@pytest.mark.parametrize(parameter_str, ray_executor_parametrized)
def test_infeasible_placement(ray_start_2_cpus, num_workers, num_hosts,
                              num_workers_per_host):
    setting = RayExecutor.create_settings(
        timeout_s=30, placement_group_timeout_s=5)
    hjob = RayExecutor(
        setting,
        num_workers=num_workers,
        num_hosts=num_hosts,
        num_workers_per_host=num_workers_per_host)
    with pytest.raises(TimeoutError):
        hjob.start()
    hjob.shutdown()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="GPU test requires 4 GPUs")
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU test requires CUDA.")
def test_gpu_ids(ray_start_4_cpus_4_gpus):
    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting, num_hosts=1, num_workers_per_host=4, use_gpu=True)
    hjob.start()
    all_envs = hjob.execute(lambda _: os.environ.copy())
    all_cudas = {ev["CUDA_VISIBLE_DEVICES"] for ev in all_envs}
    assert len(all_cudas) == 1, all_cudas
    assert len(all_envs[0]["CUDA_VISIBLE_DEVICES"].split(",")) == 4, all_envs[0]["CUDA_VISIBLE_DEVICES"]
    hjob.shutdown()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="GPU test requires 4 GPUs")
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU test requires CUDA.")
def test_gpu_ids_num_workers(ray_start_4_cpus_4_gpus):
    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(setting, num_workers=4, use_gpu=True)
    hjob.start()
    all_envs = hjob.execute(lambda _: os.environ.copy())
    all_cudas = {ev["CUDA_VISIBLE_DEVICES"] for ev in all_envs}

    assert len(all_cudas) == 1, all_cudas
    assert len(all_envs[0]["CUDA_VISIBLE_DEVICES"].split(",")) == 4, all_envs[0]["CUDA_VISIBLE_DEVICES"]

    def _test(worker):
        import horovod.torch as hvd
        hvd.init()
        local_rank = str(hvd.local_rank())
        return local_rank in os.environ["CUDA_VISIBLE_DEVICES"]

    all_valid_local_rank = hjob.execute(_test)
    assert all(all_valid_local_rank)
    hjob.shutdown()


def test_horovod_mixin(ray_start_2_cpus):
    class Test(BaseHorovodWorker):
        pass

    assert Test().hostname() == socket.gethostname()
    actor = ray.remote(BaseHorovodWorker).remote()
    DUMMY_VALUE = 1123123
    actor.update_env_vars.remote({"TEST": DUMMY_VALUE})
    assert ray.get(actor.env_vars.remote())["TEST"] == str(DUMMY_VALUE)


@pytest.mark.parametrize(parameter_str, ray_executor_parametrized)
def test_local(ray_start_4_cpus, num_workers, num_hosts, num_workers_per_host):
    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting,
        num_workers=num_workers,
        num_hosts=num_hosts,
        num_workers_per_host=num_workers_per_host)
    hjob.start()
    hostnames = hjob.execute(lambda _: socket.gethostname())
    assert len(set(hostnames)) == 1, hostnames
    hjob.shutdown()


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
@pytest.mark.parametrize(parameter_str, ray_executor_parametrized)
def test_ray_init(ray_start_4_cpus, num_workers, num_hosts,
                  num_workers_per_host):
    def simple_fn(worker):
        import horovod.torch as hvd
        hvd.init()
        return hvd.rank()

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting,
        num_workers=num_workers,
        num_hosts=num_hosts,
        num_workers_per_host=num_workers_per_host,
        use_gpu=torch.cuda.is_available())
    hjob.start()
    result = hjob.execute(simple_fn)
    assert len(set(result)) == 4
    hjob.shutdown()


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
@pytest.mark.parametrize(parameter_str, ray_executor_parametrized)
def test_ray_exec_func(ray_start_4_cpus, num_workers, num_hosts,
                       num_workers_per_host):
    def simple_fn(num_epochs):
        import horovod.torch as hvd
        hvd.init()
        return hvd.rank() * num_epochs

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting,
        num_workers=num_workers,
        num_hosts=num_hosts,
        num_workers_per_host=num_workers_per_host,
        use_gpu=torch.cuda.is_available())
    hjob.start()
    result = hjob.run(simple_fn, args=[0])
    assert len(set(result)) == 1
    hjob.shutdown()


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
@pytest.mark.parametrize(parameter_str, ray_executor_parametrized)
def test_ray_exec_remote_func(ray_start_4_cpus, num_workers, num_hosts,
                              num_workers_per_host):
    def simple_fn(num_epochs):
        import horovod.torch as hvd
        hvd.init()
        return hvd.rank() * num_epochs

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting,
        num_workers=num_workers,
        num_hosts=num_hosts,
        num_workers_per_host=num_workers_per_host,
        use_gpu=torch.cuda.is_available())
    hjob.start()
    object_refs = hjob.run_remote(simple_fn, args=[0])
    result = ray.get(object_refs)
    assert len(set(result)) == 1
    hjob.shutdown()


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
@pytest.mark.parametrize(parameter_str, ray_executor_parametrized)
def test_ray_executable(ray_start_4_cpus, num_workers, num_hosts,
                        num_workers_per_host):
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
        setting,
        num_workers=num_workers,
        num_hosts=num_hosts,
        num_workers_per_host=num_workers_per_host,
        use_gpu=torch.cuda.is_available())
    hjob.start(executable_cls=Executable, executable_args=[2])
    result = hjob.execute(lambda w: w.rank_epoch())
    assert set(result) == {0, 2, 4, 6}
    hjob.shutdown()


def _train(batch_size=32, batch_per_iter=10):
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.utils.data.distributed
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

    timeit.timeit(benchmark_step, number=batch_per_iter)
    return hvd.local_rank()


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
@pytest.mark.parametrize(parameter_str, ray_executor_parametrized)
def test_horovod_train(ray_start_4_cpus, num_workers, num_hosts,
                       num_workers_per_host):
    def simple_fn(worker):
        local_rank = _train()
        return local_rank

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting,
        num_workers=num_workers,
        num_hosts=num_hosts,
        num_workers_per_host=num_workers_per_host,
        use_gpu=torch.cuda.is_available())
    hjob.start()
    result = hjob.execute(simple_fn)
    assert set(result) == {0, 1, 2, 3}
    hjob.shutdown()


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_horovod_train_in_pg(ray_start_4_cpus):
    pg, _ = create_placement_group(
        {"CPU": 1, "GPU": int(torch.cuda.is_available())}, 4, 30, "PACK")

    @ray.remote
    class _Actor():
        def run(self):
            def simple_fn(worker):
                local_rank = _train()
                return local_rank

            setting = RayExecutor.create_settings(timeout_s=30)
            hjob = RayExecutor(
                setting,
                num_workers=4,
                num_hosts=None,
                num_workers_per_host=None,
                cpus_per_worker=1,
                gpus_per_worker=int(torch.cuda.is_available()) or None,
                use_gpu=torch.cuda.is_available())
            hjob.start()
            assert not hjob.adapter.strategy._created_placement_group
            result = hjob.execute(simple_fn)
            assert set(result) == {0, 1, 2, 3}
            hjob.shutdown()
    actor = _Actor.options(
        num_cpus=0, num_gpus=0, placement_group_capture_child_tasks=True, placement_group=pg).remote()
    ray.get(actor.run.remote())


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_remote_client_train(ray_start_client):
    def simple_fn(worker):
        local_rank = _train()
        return local_rank

    assert ray.util.client.ray.is_connected()

    setting = RayExecutor.create_settings(timeout_s=30)
    hjob = RayExecutor(
        setting, num_workers=3, use_gpu=torch.cuda.is_available())
    hjob.start()
    result = hjob.execute(simple_fn)
    assert set(result) == {0, 1, 2}
    result = ray.get(hjob.run_remote(simple_fn, args=[None]))
    assert set(result) == {0, 1, 2}
    hjob.shutdown()


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__] + sys.argv[1:]))
