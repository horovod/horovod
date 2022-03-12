from typing import Callable, List, Any, Dict, Optional, Tuple
import logging
import ray.exceptions
import socket

import time
import os
import random
import math
import threading
from dataclasses import dataclass

from horovod.ray.adapter import Adapter, BaseParams
from horovod.runner.http.http_server import RendezvousServer
from horovod.ray.utils import detect_nics
from horovod.runner.elastic.rendezvous import create_rendezvous_handler
from horovod.runner.gloo_run import (create_slot_env_vars, create_run_env_vars,
                                     _get_min_start_hosts)
from horovod.ray.worker import BaseHorovodWorker
from horovod.runner.elastic.discovery import HostDiscovery
from horovod.runner.elastic.driver import ElasticDriver

import ray
import ray.exceptions
from horovod.ray.worker import BaseHorovodWorker
from horovod.ray.utils import detect_nics
logger = logging.getLogger(__name__)

if hasattr(ray.exceptions, "GetTimeoutError"):
    GetTimeoutError = ray.exceptions.GetTimeoutError
elif hasattr(ray.exceptions, "RayTimeoutError"):
    GetTimeoutError = ray.exceptions.RayTimeoutError
else:
    raise ImportError("Unable to find Ray Timeout Error class "
                      "(GetTimeoutError, RayTimeoutError). "
                      "This is likely due to the Ray version not "
                      "compatible with Horovod-Ray.")


class RayHostDiscovery(HostDiscovery):
    """Uses Ray global state to obtain host mapping.

    Assumes that the whole global state is available for usage."""

    def __init__(self, use_gpu=False, cpus_per_worker=1, gpus_per_worker=1):
        self.use_gpu = use_gpu
        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker
        logger.debug(f"Discovery started with {cpus_per_worker} CPU / "
                     f"{gpus_per_worker} GPU per slot.")

    def find_available_hosts_and_slots(self) -> Dict[str, int]:
        """Returns a dict mapping <hostname> -> <number of slots>."""
        alive_nodes = [k for k in ray.nodes() if k["alive"]]
        host_mapping = {}
        for node in alive_nodes:
            hostname = node["NodeManagerAddress"]
            resources = node["Resources"]
            slots = resources.get("CPU", 0) // self.cpus_per_worker
            if self.use_gpu:
                gpu_slots = resources.get("GPU", 0) // self.gpus_per_worker
                slots = min(slots, gpu_slots)
            slots = int(math.ceil(slots))
            if slots:
                host_mapping[hostname] = slots

        if host_mapping and sum(host_mapping.values()) == 0:
            logger.info(f"Detected {len(host_mapping)} hosts, but no hosts "
                        "have available slots.")
            logger.debug(f"Alive nodes: {alive_nodes}")
        return host_mapping


class TestDiscovery(RayHostDiscovery):
    def __init__(self,
                 min_hosts,
                 max_hosts,
                 change_frequency_s,
                 use_gpu=False,
                 cpus_per_worker=1,
                 gpus_per_worker=1,
                 verbose=True,
                 _graceful=True):
        super().__init__(
            use_gpu=use_gpu,
            cpus_per_worker=cpus_per_worker,
            gpus_per_worker=gpus_per_worker)
        self._min_hosts = min_hosts
        self._graceful = _graceful
        self._max_hosts = max_hosts
        self._change_frequency_s = change_frequency_s
        self._last_reset_t = None
        self.verbose = verbose
        self._removed_hosts = set()

    def add_host(self, hosts):
        available_hosts = self._removed_hosts & hosts.keys()
        if available_hosts:
            host = random.choice(list(available_hosts))
            self._removed_hosts.remove(host)
        else:
            print("No hosts to add.")

    def remove_host(self, hosts):
        good_hosts = [k for k in hosts if k not in self._removed_hosts]

        from ray.autoscaler._private.commands import kill_node
        if good_hosts:
            if self._graceful:
                host = random.choice(good_hosts)
            else:
                host = kill_node(
                    os.path.expanduser("~/ray_bootstrap_config.yaml"), True,
                    False, None)
        self._removed_hosts.add(host)

    def change_hosts(self, hosts):
        for host in self._removed_hosts:
            if host not in hosts:
                self._removed_hosts.remove(host)
        current_hosts = len(hosts) - len(self._removed_hosts)
        if current_hosts <= self._min_hosts:
            self.add_host(hosts)
        elif current_hosts >= self._max_hosts:
            self.remove_host(hosts)
        else:
            if random.random() < 0.5:
                self.add_host(hosts)
            else:
                self.remove_host(hosts)

    def find_available_hosts_and_slots(self):
        t = time.time()
        if self._last_reset_t is None:
            self._last_reset_t = t
        hosts = super().find_available_hosts_and_slots()
        if t - self._last_reset_t >= self._change_frequency_s:
            self.change_hosts(hosts)
            self._last_reset_t = t
        if self.verbose:
            print(f"Total hosts: {len(hosts)}")
        remaining = {
            k: v
            for k, v in hosts.items() if k not in self._removed_hosts
        }
        if self.verbose:
            print(f"Remaining hosts: {len(remaining)} -- {remaining}")
        return remaining

@dataclass
class ElasticParams(BaseParams):
    """Parameters for elastic jobs.

    Args:
        min_workers (int): Minimum number of processes running for
            training to continue. If number of available processes dips
            below this threshold, then training will wait for
            more instances to become available.
        max_workers (int): Maximum number of training processes,
            beyond which no additional processes will be created.
            If not specified, then will be unbounded.
        reset_limit (int): Maximum number of times that the training
            job can scale up or down the number of workers after
            which the job is terminated.
        cooldown_range(Tuple[int, int]): Range(in seconds) a failing
            host will remain in blacklist.
            Example: cooldown_range=(10, 100)
            This sets the minimum cooldown period to 10 seconds,
            and the maximum cooldown period to 100 seconds.
        elastic_timeout (int): Timeout for elastic initialisation after
            re-scaling the cluster. The default value is 600 seconds.
            Alternatively, the environment variable
            HOROVOD_ELASTIC_TIMEOUT can also be used.
        cpus_per_worker (int): Number of CPU resources to allocate to
            each worker.
        use_gpu (bool): Whether to use GPU for allocation. TODO: this
            can be removed.
        gpus_per_worker (int): Number of GPU resources to allocate to
            each worker.

    """
    min_workers: int = 1
    max_workers: int = None
    reset_limit: int = None
    cooldown_range: Optional[Tuple[int, int]] = None
    elastic_timeout: int = 600
    override_discovery: bool = True

    @property
    def elastic(self):
        return True

    @property
    def adapter(self):
        return ElasticAdapter

class ElasticAdapter(Adapter):
    """Adapter for executing Ray calls for elastic Horovod jobs.

    Args:
        settings (horovod.Settings): Configuration for job setup. You can
            use a standard Horovod Settings object or create one directly
            from RayExecutor.create_settings.
        min_workers (int): Minimum number of processes running for
            training to continue. If number of available processes dips
            below this threshold, then training will wait for
            more instances to become available.
        max_workers (int): Maximum number of training processes,
            beyond which no additional processes will be created.
            If not specified, then will be unbounded.
        reset_limit (int): Maximum number of times that the training
            job can scale up or down the number of workers after
            which the job is terminated.
        cooldown_range (Tuple[int, int]): Range(in seconds) a failing
            host will remain in blacklist.
            Example: cooldown_range=(10, 100)
            This sets the minimum cooldown period to 10 seconds,
            and the maximum cooldown period to 100 seconds.
        elastic_timeout (int): Timeout for elastic initialisation after
            re-scaling the cluster. The default value is 600 seconds.
            Alternatively, the environment variable
            HOROVOD_ELASTIC_TIMEOUT can also be used.'
        cpus_per_worker (int): Number of CPU resources to allocate to
            each worker.
        use_gpu (bool): Whether to use GPU for allocation. TODO: this
            can be removed.
        gpus_per_worker (int): Number of GPU resources to allocate to
            each worker.
        override_discovery (bool): Whether for the ElasticRayExecutor to
            automatically provide a discovery mechanism for ElasticSettings.

    """
    def __init__(self,
                settings,
                min_workers: int,
                max_workers: Optional[int] = None,
                use_gpu: bool = False,
                cpus_per_worker: int = 1,
                gpus_per_worker: Optional[int] = None,
                override_discovery: bool=True,
                reset_limit: int = None,
                cooldown_range: Optional[Tuple[int, int]] = None,
                elastic_timeout: int = 600):
        self.settings = settings
        if override_discovery:
            settings.discovery = RayHostDiscovery(
                use_gpu=use_gpu,
                cpus_per_worker=cpus_per_worker,
                gpus_per_worker=gpus_per_worker)
        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker
        self.use_gpu = use_gpu
        # moved from settings
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.num_workers = min_workers
        self.reset_limit = reset_limit
        self.cooldown_range = cooldown_range
        self.elastic_timeout = elastic_timeout
        self.driver = None
        self.rendezvous = None

    def start(self,
              executable_cls: type = None,
              executable_args: Optional[List] = None,
              executable_kwargs: Optional[Dict] = None,
              extra_env_vars: Optional[Dict] = None):
        """Starts the Horovod driver and services.

        Args:
            executable_cls (type): The class that will be created within
                an actor (BaseHorovodWorker). This will allow Horovod
                to establish its connections and set env vars.
            executable_args (List): Arguments to be passed into the
                worker class upon initialization.
            executable_kwargs (Dict): Keyword arguments to be passed into the
                worker class upon initialization.
            extra_env_vars (Dict): Environment variables to be set
                on the actors (worker processes) before initialization.

        """

        self.rendezvous = RendezvousServer(self.settings.verbose)
        self.driver = ElasticDriver(
            rendezvous=self.rendezvous,
            discovery=self.settings.discovery,
            min_num_proc=self.min_workers,
            max_num_proc=self.max_workers,
            timeout=self.elastic_timeout,
            reset_limit=self.reset_limit,
            cooldown_range=self.cooldown_range,
            verbose=self.settings.verbose)
        handler = create_rendezvous_handler(self.driver)
        logger.debug("[ray] starting rendezvous")
        global_rendezv_port = self.rendezvous.start(handler)

        logger.debug(f"[ray] waiting for {self.num_workers} to start.")
        self.driver.wait_for_available_slots(self.num_workers)

        # Host-to-host common interface detection
        # requires at least 2 hosts in an elastic job.
        min_hosts = _get_min_start_hosts(self.settings)
        current_hosts = self.driver.wait_for_available_slots(
            self.num_workers, min_hosts=min_hosts)
        logger.debug("[ray] getting common interfaces")
        nics = detect_nics(
            self.settings,
            all_host_names=current_hosts.host_assignment_order,
        )
        logger.debug("[ray] getting driver IP")
        server_ip = socket.gethostbyname(socket.gethostname())
        self.run_env_vars = create_run_env_vars(
            server_ip, nics, global_rendezv_port, elastic=True)

        self.executable_cls = executable_cls
        self.executable_args = executable_args
        self.executable_kwargs = executable_kwargs
        self.env_vars = extra_env_vars or {}


    def _create_resources(self, hostname: str):
        resources = dict(
            num_cpus=self.cpus_per_worker,
            num_gpus=int(self.use_gpu) * self.gpus_per_worker,
            resources={f"node:{hostname}": 0.01})
        return resources

    def _create_remote_worker(self, slot_info, worker_env_vars):
        hostname = slot_info.hostname
        loaded_worker_cls = self.remote_worker_cls.options(
            **self._create_resources(hostname))

        worker = loaded_worker_cls.remote()
        worker.update_env_vars.remote(worker_env_vars)
        worker.update_env_vars.remote(create_slot_env_vars(slot_info))
        if self.use_gpu:
            visible_devices = ",".join(
                [str(i) for i in range(slot_info.local_size)])
            worker.update_env_vars.remote({
                "CUDA_VISIBLE_DEVICES":
                visible_devices
            })
        return worker

    def _create_spawn_worker_fn(self, return_results: List,
                                worker_fn: Callable,
                                queue: "ray.util.Queue") -> Callable:
        self.remote_worker_cls = ray.remote(BaseHorovodWorker)
        # event = register_shutdown_event()
        worker_env_vars = {}
        worker_env_vars.update(self.run_env_vars.copy())
        worker_env_vars.update(self.env_vars.copy())
        worker_env_vars.update({"PYTHONUNBUFFERED": "1"})

        def worker_loop(slot_info, events):
            def ping_worker(worker):
                # There is an odd edge case where a node can be removed
                # before the remote worker is started, leading to a failure
                # in trying to create the horovod mesh.
                try:
                    ping = worker.execute.remote(lambda _: 1)
                    ray.get(ping, timeout=10)
                except Exception as e:
                    logger.error(f"{slot_info.hostname}: Ping failed - {e}")
                    return False
                return True

            worker = self._create_remote_worker(slot_info, worker_env_vars)
            if not ping_worker(worker):
                return 1, time.time()

            ray.get(worker.set_queue.remote(queue))
            future = worker.execute.remote(worker_fn)

            result = None
            while result is None:
                try:
                    #  TODO: make this event driven at some point.
                    retval = ray.get(future, timeout=0.1)
                    return_results.append((slot_info.rank, retval))
                    # Success
                    result = 0, time.time()
                except GetTimeoutError:
                    # Timeout
                    if any(e.is_set() for e in events):
                        ray.kill(worker)
                        result = 1, time.time()
                except Exception as e:
                    logger.error(f"{slot_info.hostname}[{slot_info.rank}]:{e}")
                    ray.kill(worker)
                    result = 1, time.time()
            logger.debug(f"Worker ({slot_info}) routine is done!")
            return result

        return worker_loop


    def run(self,
            fn: Callable[[Any], Any],
            args: Optional[List] = None,
            kwargs: Optional[Dict] = None,
            callbacks: Optional[List[Callable]] = None) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            fn: Target function that can be executed with arbitrary
                args and keyword arguments.
            args: List of arguments to be passed into the target function.
            kwargs: Dictionary of keyword arguments to be
                passed into the target function.
            callbacks: List of callables. Each callback must either
                be a callable function or a class that implements __call__.
                Every callback will be invoked on every value logged
                by the rank 0 worker.

        Returns:
            Deserialized return values from the target function.
        """
        args = args or []
        kwargs = kwargs or {}
        f = lambda _: fn(*args, **kwargs)
        return self._run_remote(f, callbacks=callbacks)

    def _run_remote(self,
            worker_fn: Callable,
            callbacks: Optional[List[Callable]] = None) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            worker_fn: Target elastic function that can be executed.
            callbacks: List of callables. Each callback must either
                be a callable function or a class that implements __call__.
                Every callback will be invoked on every value logged
                by the rank 0 worker.

        Returns:
            List of return values from every completed worker.
        """
        return_values = []
        from ray.util.queue import Queue
        import inspect
        args = inspect.getfullargspec(Queue).args
        if "actor_options" not in args:
            # Ray 1.1 and less
            _queue = Queue()
        else:
            _queue = Queue(actor_options={
                "num_cpus": 0,
                "resources": {
                    ray.state.current_node_id(): 0.001
                }
            })
        self.driver.start(
            self.num_workers,
            self._create_spawn_worker_fn(return_values, worker_fn, _queue))

        def _process_calls(queue, callbacks, event):
            if not callbacks:
                return
            while queue.actor:
                if not queue.empty():
                    result = queue.get_nowait()
                    for c in callbacks:
                        c(result)
                    # avoid slamming the CI
                elif event.is_set():
                    break
                time.sleep(0.1)

        try:
            event = threading.Event()
            _callback_thread = threading.Thread(
                target=_process_calls,
                args=(_queue, callbacks, event),
                daemon=True)
            _callback_thread.start()
            res = self.driver.get_results()
            event.set()
            if _callback_thread:
                _callback_thread.join(timeout=60)
        finally:
            if hasattr(_queue, "shutdown"):
                _queue.shutdown()
            else:
                done_ref = _queue.actor.__ray_terminate__.remote()
                done, not_done = ray.wait([done_ref], timeout=5)
                if not_done:
                    ray.kill(_queue.actor)
        self.driver.stop()

        if res.error_message is not None:
            raise RuntimeError(res.error_message)

        for name, value in sorted(
                res.worker_results.items(), key=lambda item: item[1][1]):
            exit_code, timestamp = value
            if exit_code != 0:
                raise RuntimeError(
                    'Horovod detected that one or more processes '
                    'exited with non-zero '
                    'status, thus causing the job to be terminated. '
                    'The first process '
                    'to do so was:\nProcess name: {name}\nExit code: {code}\n'
                    .format(name=name, code=exit_code))

        return_values = [
            value for k, value in sorted(return_values, key=lambda kv: kv[0])
        ]
        return return_values

    def run_remote(self,
                fn: Callable[[Any], Any]) -> List[Any]:
        raise NotImplementedError("ObjectRefs cannot be returned from Elastic runs as the workers are ephemeral")

    def execute(self, fn: Callable[["executable_cls"], Any],
                callbacks: Optional[List[Callable]] = None) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            fn: Target function to be invoked on every object.
            callbacks: List of callables. Each callback must either
                be a callable function or a class that implements __call__.
                Every callback will be invoked on every value logged
                by the rank 0 worker.
        Returns:
            Deserialized return values from the target function.
        """
        return ray.get(self._run_remote(fn, callbacks=callbacks))

    def execute_single(self,
                       fn: Callable[["executable_cls"], Any]) -> List[Any]:
        """Executes the provided function on the rank 0 worker (chief).

        Args:
            fn: Target function to be invoked on the chief object.

        Returns:
            Deserialized return values from the target function.
        """
        raise NotImplementedError("Elastic mode does not support execute_single. Please use the execute method instead")

    def shutdown(self):
        """Destroys the driver."""
        if not self.driver:
            return
        assert self.driver.finished()
        self.driver = None
