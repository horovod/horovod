import logging
import time
import os
import math
from typing import Callable, List, Any, Dict, Optional

from horovod.runner.common.util import timeout, secret

from horovod.runner.http.http_server import RendezvousServer
from horovod.runner.util import network
from horovod.runner.gloo_run import (create_slot_env_vars, create_run_env_vars,
                                     _get_min_start_hosts)
from horovod.runner.driver import driver_service
from horovod.runner.elastic.settings import ElasticSettings
from horovod.runner.elastic.rendezvous import create_rendezvous_handler
from horovod.runner.elastic.discovery import HostDiscovery
from horovod.runner.elastic.driver import ElasticDriver

import ray
import ray.exceptions
from horovod.ray.runner import BaseHorovodWorker

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

    def __init__(self, use_gpu=False, cpus_per_slot=1, gpus_per_slot=1):
        self.use_gpu = use_gpu
        self.cpus_per_slot = cpus_per_slot
        self.gpus_per_slot = gpus_per_slot

    def find_available_hosts_and_slots(self) -> Dict[str, int]:
        """Returns a dict mapping <hostname> -> <number of slots>."""
        alive_nodes = [k for k in ray.nodes() if k["alive"]]
        host_mapping = {}
        for node in alive_nodes:
            hostname = node["NodeManagerAddress"]
            resources = node["Resources"]
            slots = resources.get("CPU", 0) // self.cpus_per_slot
            if self.use_gpu:
                gpu_slots = resources.get("GPU", 0) // self.gpus_per_slot
                slots = min(slots, gpu_slots)
            host_mapping[hostname] = int(math.ceil(slots))
        return host_mapping


class ElasticRayExecutor:
    """Executor for elastic jobs using Ray.

    Leverages the Ray global state to detect available hosts and
    slots. Assumes that the entire Ray cluster is available for
    the Executor to use.

    Args:
        settings: Configuration for the elastic job
            setup. You can use a standard Horovod ElasticSettings
            object or create one directly from
            ElasticRayExecutor.create_settings.
        use_gpu (bool): Whether to use GPU for allocation.
        cpus_per_slot (int): Number of CPU resources to allocate to
            each worker.
        gpus_per_slot (int): Number of GPU resources to allocate to
            each worker.
        env_vars (Dict): Environment variables to be set
            on the actors (worker processes) before initialization.
        override_discovery (bool): Whether for the ElasticRayExecutor to
            automatically provide a discovery mechanism for ElasticSettings.

    Example:

    .. code-block:: python

        import ray
        ray.init(address="auto")
        settings = ElasticRayExecutor.create_settings(verbose=True)
        executor = ElasticRayExecutor(
            settings, use_gpu=True, cpus_per_slot=2)
        executor.start()
        executor.run(train_fn)
    """

    @staticmethod
    def create_settings(min_np: int = 1,
                        max_np: int = None,
                        reset_limit: int = None,
                        elastic_timeout: int = 600,
                        timeout_s: int = 30,
                        ssh_identity_file: str = None,
                        nics: str = None,
                        **kwargs):
        """Returns a Settings object for ElasticRayExecutor.

        Note that the `discovery` property will be set at runtime.

        Args:
            min_np (int): Minimum number of processes running for
                training to continue. If number of available processes dips
                below this threshold, then training will wait for
                more instances to become available.
            max_np (int): Maximum number of training processes,
                beyond which no additional processes will be created.
                If not specified, then will be unbounded.
            reset_limit (int): Maximum number of times that the training
                job can scale up or down the number of workers after
                which the job is terminated.
            elastic_timeout (int): Timeout for elastic initialisation after
                re-scaling the cluster. The default value is 600 seconds.
                Alternatively, the environment variable
                HOROVOD_ELASTIC_TIMEOUT can also be used.'
            timeout_s (int): Horovod performs all the checks and starts the
                processes before the specified timeout.
                The default value is 30 seconds.
            ssh_identity_file (str): File on the driver from which
                the identity (private key) is read.
            nics (set): Network interfaces that can be used for communication.
        """
        start_timeout = timeout.Timeout(
            timeout_s,
            message="Timed out waiting for {activity}. Please "
            "check connectivity between servers. You "
            "may need to increase the --start-timeout "
            "parameter if you have too many servers.")
        ssh_identity_file = ssh_identity_file or os.path.expanduser(
            "~/ray_bootstrap_key.pem")
        settings = ElasticSettings(
            discovery=None,
            min_np=min_np,
            max_np=max_np,
            elastic_timeout=elastic_timeout,
            reset_limit=reset_limit,
            num_proc=min_np,
            ssh_identity_file=ssh_identity_file,
            nics=nics,
            start_timeout=start_timeout,
            key=secret.make_secret_key() if secret else None,
            **kwargs)
        return settings

    def __init__(self,
                 settings: ElasticSettings,
                 use_gpu: bool = False,
                 cpus_per_slot: int = 1,
                 gpus_per_slot: Optional[int] = None,
                 env_vars: dict = None,
                 override_discovery=True):
        if gpus_per_slot and not use_gpu:
            raise ValueError("gpus_per_slot is set, but use_gpu is False. "
                             "use_gpu must be True if gpus_per_slot is set. ")
        if use_gpu and gpus_per_slot < 1:
            raise ValueError(
                f"gpus_per_slot must be >= 1: Got {gpus_per_slot}.")

        gpus_per_slot = gpus_per_slot or 1
        if override_discovery:
            settings.discovery = RayHostDiscovery(
                use_gpu=use_gpu,
                cpus_per_slot=cpus_per_slot,
                gpus_per_slot=gpus_per_slot)
        self.cpus_per_slot = cpus_per_slot
        self.gpus_per_slot = gpus_per_slot
        self.use_gpu = use_gpu
        self.settings = settings
        self.driver = None
        self.rendezvous = None
        self.env_vars = env_vars or {}

    def start(self):
        """Starts the Horovod driver and services."""
        self.rendezvous = RendezvousServer(self.settings.verbose)
        self.driver = ElasticDriver(
            rendezvous=self.rendezvous,
            discovery=self.settings.discovery,
            min_np=self.settings.min_np,
            max_np=self.settings.max_np,
            timeout=self.settings.elastic_timeout,
            reset_limit=self.settings.reset_limit,
            verbose=self.settings.verbose)
        handler = create_rendezvous_handler(self.driver)
        global_rendezv_port = self.rendezvous.start(handler)
        self.driver.wait_for_available_slots(self.settings.num_proc)

        # Host-to-host common interface detection
        # requires at least 2 hosts in an elastic job.
        min_hosts = _get_min_start_hosts(self.settings)
        current_hosts = self.driver.wait_for_available_slots(
            self.settings.num_proc, min_hosts=min_hosts)
        nics = driver_service.get_common_interfaces(
            self.settings, current_hosts.host_assignment_order)

        server_ip = network.get_driver_ip(nics)
        self.run_env_vars = create_run_env_vars(
            server_ip, nics, global_rendezv_port, elastic=True)

    def _create_resources(self, hostname: str):
        resources = dict(
            num_cpus=self.cpus_per_slot,
            num_gpus=int(self.use_gpu) * self.gpus_per_slot,
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
                                worker_fn: Callable) -> Callable:
        self.remote_worker_cls = ray.remote(BaseHorovodWorker)
        # event = register_shutdown_event()
        worker_env_vars = {}
        worker_env_vars.update(self.run_env_vars.copy())
        worker_env_vars.update(self.env_vars.copy())
        worker_env_vars.update({"PYTHONUNBUFFERED": "1"})

        def worker_loop(slot_info, events):
            worker = self._create_remote_worker(slot_info, worker_env_vars)
            future = worker.execute.remote(lambda _: worker_fn())

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
                    logger.exception(str(e))
                    # Fail
                    result = 1, time.time()
            return result

        return worker_loop

    def run(self, worker_fn: Callable) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            worker_fn: Target elastic function that can be executed.

        Returns:
            List of return values from every completed worker.
        """
        return_values = []
        self.driver.start(
            self.settings.num_proc,
            self._create_spawn_worker_fn(return_values, worker_fn))
        res = self.driver.get_results()
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
