import ray
from ray import services

from collections import defaultdict
from dataclasses import dataclass
import os
import socket
from typing import Dict, Callable, Any, Optional, List
import logging

from horovod.runner.common.util import secret, timeout, hosts
from horovod.runner.http.http_server import RendezvousServer
from horovod.ray import ray_logger
from horovod.ray.utils import detect_nics, nics_to_env_var

logger = logging.getLogger(__name__)


@dataclass
class MiniSettings:
    """Minimal settings necessary for Ray to work.

    Can be replaced with a proper Horovod Settings object.
    """
    nics: set = None
    verbose: int = 1
    key: str = secret.make_secret_key() if secret else None
    ssh_port: int = None
    ssh_identity_file: str = None
    timeout_s: int = 300
    placement_group_timeout_s = 100

    @property
    def start_timeout(self):
        return timeout.Timeout(
            self.timeout_s,
            message="Timed out waiting for {activity}. Please "
            "check connectivity between servers. You "
            "may need to increase the --start-timeout "
            "parameter if you have too many servers.")


def map_blocking(fn, collection):
    return ray.get([fn(w) for w in collection])


class BaseHorovodWorker:
    executable = None

    def __init__(self, world_rank=0, world_size=1):
        os.environ["HOROVOD_HOSTNAME"] = self.hostname()
        os.environ["HOROVOD_RANK"] = str(world_rank)
        os.environ["HOROVOD_SIZE"] = str(world_size)

    def hostname(self) -> str:
        # TODO: This is probably not the right way to retrieve
        # the intended hostname.
        return socket.gethostname()

    def get_gpu_ids(self) -> List[int]:
        """Return list of CUDA device IDs available to this worker."""
        return ray.get_gpu_ids()

    def update_env_vars(self, env_vars: Dict[str, str]):
        """Update the env vars in the actor process."""
        sanitized = {k: str(v) for k, v in env_vars.items()}
        os.environ.update(sanitized)

    def env_vars(self):
        """Check the env vars in the actor process."""
        return dict(os.environ)

    def start_executable(self,
                         executable_cls: type = None,
                         executable_args: list = None,
                         executable_kwargs: dict = None):
        """Instantiates the executable class with provided args.

        If none, self.executable = None.

        Args:
            executable_cls (type): Class of object to be created on all
                workers.
            executable_args (list): Initialization arguments for the
                executable_cls.
            executable_kwargs (dict): Initialization arguments for the
                executable_cls.
        """
        executable_args = executable_args or []
        executable_kwargs = executable_kwargs or {}
        if executable_cls:
            self.executable = executable_cls(*executable_args,
                                             **executable_kwargs)

    def execute(self, func):
        """Executes an arbitrary function on self."""
        return func(self.executable)

    def set_queue(self, queue):
        """Sets the queue for multi-node logging."""
        ray_logger.configure(queue=queue)


class Coordinator:
    """Responsible for instantiating the Rendezvous server.

    Args:
        settings: Horovod Settings object."""
    rendezvous = None
    global_rendezv_port = None
    nics = None
    hostnames = None

    def __init__(
            self,
            settings,
    ):
        self.settings = settings
        self.hostnames_by_rank = defaultdict(list)

    @property
    def world_size(self) -> int:
        return sum(len(ranks) for ranks in self.hostnames_by_rank.values())

    @property
    def hoststring(self) -> str:
        return ",".join([
            f"{host}:{len(ranks)}"
            for host, ranks in self.hostnames_by_rank.items()
        ])

    def register(self, hostname: str, world_rank: int):
        self.hostnames_by_rank[hostname].append(world_rank)

    def finalize_registration(self) -> dict:
        """Return a dictionary for all ranks."""
        rank_to_info = {}
        for node_world_rank, (hostname, ranks) in enumerate(
                self.hostnames_by_rank.items()):
            for local_rank, world_rank in enumerate(ranks):
                rank_to_info[world_rank] = dict(
                    HOROVOD_CROSS_RANK=node_world_rank,
                    HOROVOD_CROSS_SIZE=len(self.hostnames_by_rank),
                    HOROVOD_LOCAL_RANK=local_rank,
                    HOROVOD_LOCAL_SIZE=len(ranks))
        return rank_to_info

    def establish_rendezvous(self) -> Dict[str, str]:
        """Creates the rendezvous server and identifies the nics to be used.

        Returns:
            Environment variables for each worker.
        """

        # start global rendezvous server and get port that it is listening on
        self.rendezvous = RendezvousServer(self.settings.verbose)

        # allocate processes into slots
        # hosts = parse_hosts(hosts_string="10.11.11.11:4,10.11.11.12:4")
        parsed_hosts = hosts.parse_hosts(hosts_string=self.hoststring)
        host_alloc_plan = hosts.get_host_assignments(parsed_hosts,
                                                     self.world_size)

        # start global rendezvous server and get port that it is listening on
        self.global_rendezv_port = self.rendezvous.start()
        self.rendezvous.init(host_alloc_plan)

        return {
            "HOROVOD_GLOO_RENDEZVOUS_ADDR": services.get_node_ip_address(),
            "HOROVOD_GLOO_RENDEZVOUS_PORT": str(self.global_rendezv_port),
            "HOROVOD_CONTROLLER": "gloo",
            "HOROVOD_CPU_OPERATIONS": "gloo",
        }


class RayExecutor:
    """Job class for Horovod + Ray integration.

    Args:
        settings (horovod.Settings): Configuration for job setup. You can
            use a standard Horovod Settings object or create one directly
            from RayExecutor.create_settings.
        num_hosts (int): Number of machines to execute the job on.
        num_slots (int): Number of workers to be placed on each machine.
        cpus_per_slot (int): Number of CPU resources to allocate to
            each worker.
        use_gpu (bool): Whether to use GPU for allocation. TODO: this
            can be removed.
        gpus_per_slot (int): Number of GPU resources to allocate to
            each worker.
    """

    @classmethod
    def create_settings(cls,
                        timeout_s,
                        ssh_identity_file=None,
                        ssh_str=None,
                        placement_group_timeout_s=100):
        """Create a mini setting object.

        Args:
            timeout_s (int): Timeout parameter for Gloo rendezvous.
            ssh_identity_file (str): Path to the identity file to
                ssh into different hosts on the cluster.
            ssh_str (str): CAUTION WHEN USING THIS. Private key
                file contents. Writes the private key to ssh_identity_file.
            placement_group_timeout_s (int): Timeout parameter for Ray
                Placement Group creation.

        Returns:
            MiniSettings object.
        """
        if ssh_str and not os.path.exists(ssh_identity_file):
            with open(ssh_identity_file, "w") as f:
                os.chmod(ssh_identity_file, 0o600)
                f.write(ssh_str)
        return MiniSettings(
            ssh_identity_file=ssh_identity_file,
            timeout_s=timeout_s,
            placement_group_timeout_s=placement_group_timeout_s)

    def __init__(self,
                 settings,
                 num_hosts: int = 1,
                 num_slots: int = 1,
                 cpus_per_slot: int = 1,
                 use_gpu: bool = False,
                 gpus_per_slot: Optional[int] = None):

        if gpus_per_slot and not use_gpu:
            raise ValueError("gpus_per_slot is set, but use_gpu is False. "
                             "use_gpu must be True if gpus_per_slot is set. ")
        if use_gpu and isinstance(gpus_per_slot, int) and gpus_per_slot < 1:
            raise ValueError(
                f"gpus_per_slot must be >= 1: Got {gpus_per_slot}.")

        self.settings = settings
        self.num_hosts = num_hosts
        self.num_slots = num_slots
        self.cpus_per_slot = cpus_per_slot
        self.use_gpu = use_gpu
        self.gpus_per_slot = gpus_per_slot or 1

        self.workers = []
        self.placement_group = None

    @property
    def num_workers(self):
        return self.num_hosts * self.num_slots

    def _create_workers(self, host_resources):
        bundles = [host_resources.copy() for _ in range(self.num_hosts)]
        pg = ray.util.placement_group(bundles, strategy="STRICT_SPREAD")
        self.placement_group = pg
        logger.debug("Waiting for placement group to start.")
        ready, _ = ray.wait(
            [pg.ready()], timeout=self.settings.placement_group_timeout_s)
        if ready:
            logger.debug("Placement group has started.")
        else:
            raise TimeoutError(
                "Placement group creation timed out. Make sure "
                "your cluster either has enough resources or use "
                "an autoscaling cluster. Current resources "
                "available: {}, resources requested by the "
                "placement group: {}".format(ray.available_resources(),
                                             pg.bundle_specs))

        # Placement group has started. Now create the workers.
        self.workers = []
        # Keep ref of one worker per node for NIC detection.
        node_workers = []
        # STRICT_SPREAD guarantees each bundle is on a different node.
        # Create num_slots workers per bundle, i.e. per machine.
        for bundle_index in range(len(bundles)):
            gpu_id_futures = []
            curr_node_workers = []
            remote_cls = ray.remote(BaseHorovodWorker)
            for i in range(self.num_slots):
                remote_cls = remote_cls.options(
                    num_cpus=self.cpus_per_slot,
                    num_gpus=self.gpus_per_slot * int(self.use_gpu),
                    placement_group=pg,
                    placement_group_bundle_index=bundle_index)
                worker = remote_cls.remote(
                    world_rank=self.num_slots * bundle_index + i,
                    world_size=self.num_workers)
                if self.use_gpu:
                    gpu_id_futures.append(worker.get_gpu_ids.remote())
                self.workers.append(worker)
                curr_node_workers.append(worker)
            if len(gpu_id_futures) > 0:
                # By setting CUDA VISIBLE DEVICES to ALL GPUs,
                # CUDA will be able to detect adjacent devices and use IPC
                # allowing for better performance.
                gpu_ids = sum(ray.get(gpu_id_futures), [])
                # Make sure that each worker on the node has unique device.
                assert len(gpu_ids) == len(
                    set(gpu_ids)) == self.num_slots, gpu_ids
                all_ids = ",".join([str(gpu_id) for gpu_id in gpu_ids])
                futures = []
                for worker in curr_node_workers:
                    futures.append(
                        worker.update_env_vars.remote({
                            "CUDA_VISIBLE_DEVICES":
                            all_ids
                        }))
                ray.get(futures)
            node_workers.append(curr_node_workers[0])

        return self.workers, node_workers

    def _start_executables(self, executable_cls, executable_args,
                           executable_kwargs):
        def _start_exec(worker):
            return worker.start_executable.remote(
                executable_cls, executable_args, executable_kwargs)

        map_blocking(_start_exec, self.workers)

    def start(self,
              executable_cls: type = None,
              executable_args: Optional[List] = None,
              executable_kwargs: Optional[Dict] = None,
              extra_env_vars: Optional[Dict] = None):
        """Starts the workers and colocates them on all machines.

        We implement a node grouping because it seems like
        our implementation doesn't quite work for imbalanced nodes.
        Also, colocation performance is typically much better than
        non-colocated workers.

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
        extra_env_vars = extra_env_vars or {}

        def resources_per_host():
            num_cpus = self.cpus_per_slot * self.num_slots
            num_gpus = self.gpus_per_slot * self.num_slots * int(self.use_gpu)
            return dict(CPU=num_cpus, GPU=num_gpus)

        self.coordinator = Coordinator(self.settings)
        executable_args = executable_args or []
        self.workers, node_workers = self._create_workers(resources_per_host())
        # Get all the hostnames of all workers
        hostnames = map_blocking(lambda w: w.hostname.remote(), self.workers)
        # Register each hostname to the coordinator. assumes the hostname
        # ordering is the same.
        for rank, hostname in enumerate(hostnames):
            self.coordinator.register(hostname, rank)
        all_info = self.coordinator.finalize_registration()

        indexed_runners = dict(enumerate(self.workers))
        for rank, local_cross_env_var in all_info.items():
            indexed_runners[rank].update_env_vars.remote(local_cross_env_var)

        coordinator_envs = self.coordinator.establish_rendezvous()
        coordinator_envs.update(extra_env_vars)
        nics = detect_nics(
            self.settings,
            all_host_names=list(self.coordinator.hostnames_by_rank),
            node_workers=node_workers)
        coordinator_envs.update(nics_to_env_var(nics))

        map_blocking(lambda w: w.update_env_vars.remote(coordinator_envs),
                     self.workers)

        self._start_executables(executable_cls, executable_args,
                                executable_kwargs)

    def execute(self, fn: Callable[["executable_cls"], Any]) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            fn: Target function to be invoked on every object.

        Returns:
            Deserialized return values from the target function.
        """
        return ray.get([worker.execute.remote(fn) for worker in self.workers])

    def run(self,
            fn: Callable[[Any], Any],
            args: Optional[List] = None,
            kwargs: Optional[Dict] = None) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            fn: Target function that can be executed with arbitrary
                args and keyword arguments.
            args: List of arguments to be passed into the target function.
            kwargs: Dictionary of keyword arguments to be
                passed into the target function.

        Returns:
            Deserialized return values from the target function.
        """
        return ray.get(self.run_remote(fn, args, kwargs))

    def run_remote(self,
                   fn: Callable[[Any], Any],
                   args: Optional[List] = None,
                   kwargs: Optional[Dict] = None) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            fn: Target function that can be executed with arbitrary
                args and keyword arguments.
            args: List of arguments to be passed into the target function.
            kwargs: Dictionary of keyword arguments to be
                passed into the target function.

        Returns:
            list: List of ObjectRefs that you can run `ray.get` on to
                retrieve values.
        """
        args = args or []
        kwargs = kwargs or {}
        return [
            worker.execute.remote(lambda w: fn(*args, **kwargs))
            for worker in self.workers
        ]

    def execute_single(self,
                       fn: Callable[["executable_cls"], Any]) -> List[Any]:
        """Executes the provided function on the rank 0 worker (chief).

        Args:
            fn: Target function to be invoked on the chief object.

        Returns:
            Deserialized return values from the target function.
        """
        return ray.get(self.workers[0].execute.remote(fn))

    def shutdown(self):
        """Destroys the provided workers."""
        for worker in self.workers:
            del worker

        if self.placement_group:
            ray.util.remove_placement_group(self.placement_group)

        self.workers = []
        self.placement_group = None
