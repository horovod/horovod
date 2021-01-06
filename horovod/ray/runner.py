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


@ray.remote
class NodeColocator:
    """Responsible for colocation of child actors.

    These actors are given resources equal to the sum of resources
    to be effectively allocated to their children. The child
    workers are currently allocated 0 resources in this implementation.

    This is a mechanism for gang-scheduling and could be replaced
    later on with placement groups. Gang-scheduling must occur because
    otherwise another concurrent group could be placed on this node.

    Right now, the only resources that are explicitly propogated to
    underlying colocated workers are cuda visible devices.

    Args:
        node_rank (int): Rank of the node that this colocator is placed on.
        num_slots (int): Total number of slots on this machine.
        world_size (int): Total number of workers (slots) participating
            in the job across all nodes.
        use_gpu (bool): Whether to utilize the GPUs on the node.
    """

    def __init__(self, *, node_rank: int, num_slots: int, world_size: int,
                 use_gpu: bool):
        self.node_rank = node_rank
        self.num_slots = num_slots
        self.world_size = world_size
        if use_gpu:
            gpu_ids = ray.get_gpu_ids()
            assert len(gpu_ids) == num_slots, gpu_ids
        self.workers = []

    def create_workers(self):
        """Colocates a number of workers.

        Also passes on the CUDA_VISIBLE_DEVICES to each worker.
        """
        # Create a node ip resource label so that we can pin
        # all of the child actors to the same node. This ensures
        # colocation and balanced training.
        node_id = f"node:{services.get_node_ip_address()}"
        remote_cls = ray.remote(BaseHorovodWorker)
        remote_cls = remote_cls.options(
            num_cpus=0, num_gpus=0, resources={node_id: 0.01})

        rank_start = self.num_slots * self.node_rank
        self.workers = [
            remote_cls.remote(world_rank=rank, world_size=self.world_size)
            for rank in range(rank_start, rank_start + self.num_slots)
        ]

        # Propogate cuda visible devices to the underlying
        # colocated workers.
        gpu_ids = ray.get_gpu_ids()
        all_ids = ",".join([str(gpu_id) for gpu_id in gpu_ids])
        # By setting CUDA VISIBLE DEVICES to ALL GPUs,
        # CUDA will be able to detect adjacent devices and use IPC
        # allowing for better performance.
        futures = []
        for worker in self.workers:
            futures.append(
                worker.update_env_vars.remote({
                    "CUDA_VISIBLE_DEVICES": all_ids
                }))
        # Avoid asynchrony for tests
        ray.get(futures)
        return node_id

    def get_workers(self) -> List:
        return self.workers

    def execute(self, func):
        """Executes an arbitrary function on self."""
        return func(self)


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
                    NODE_WORLD_RANK=node_world_rank,
                    NODE_WORLD_SIZE=len(self.hostnames_by_rank),
                    LOCAL_RANK=local_rank,
                    LOCAL_SIZE=len(ranks))
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
        num_slots (int): Humber of workers to be placed on each machine.
        cpus_per_slot (int): Number of CPU resources to allocate to
            each worker.
        use_gpu (bool): Whether to use GPU for allocation. TODO: this
            can be removed.
        gpus_per_slot (int): Number of GPU resources to allocate to
            each worker.
    """

    @classmethod
    def create_settings(cls, timeout_s, ssh_identity_file=None, ssh_str=None):
        """Create a mini setting object.

        Args:
            timeout_s (int): Tiemout parameter for Gloo rendezvous.
            ssh_identity_file (str): Path to the identity file to
                ssh into different hosts on the cluster.
            ssh_str (str): CAUTION WHEN USING THIS. Private key
                file contents. Writes the private key to ssh_identity_file.

        Returns:
            MiniSettings object.
        """
        if ssh_str and not os.path.exists(ssh_identity_file):
            with open(ssh_identity_file, "w") as f:
                os.chmod(ssh_identity_file, 0o600)
                f.write(ssh_str)
        return MiniSettings(
            ssh_identity_file=ssh_identity_file, timeout_s=timeout_s)

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

    @property
    def num_workers(self):
        return self.num_hosts * self.num_slots

    def _create_workers(self, host_resources):
        colocator_cls = NodeColocator.options(**host_resources)
        # Create a number of coordinators.
        colocators = [
            colocator_cls.remote(
                node_rank=node_rank,
                num_slots=self.num_slots,
                world_size=self.num_workers,
                use_gpu=host_resources["num_gpus"] > 0)
            for node_rank in range(self.num_hosts)
        ]
        # We must save a pointer to each colocator to prevent their resource
        # allocation from being released, along with their children from
        # going out of scope.
        self.colocators = colocators

        node_ids = map_blocking(lambda a: a.create_workers.remote(),
                                colocators)
        if not len(set(node_ids)) == len(node_ids):
            raise RuntimeError("Colocator actors must "
                               f"be placed on unique nodes! Got: {node_ids}")

        # Obtain handles to the workers
        workers = map_blocking(lambda w: w.get_workers.remote(), colocators)
        return sum(workers, [])

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
            return dict(num_cpus=num_cpus, num_gpus=num_gpus)

        self.coordinator = Coordinator(self.settings)
        executable_args = executable_args or []
        self.workers = self._create_workers(resources_per_host())
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
            node_workers=self.colocators)
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
        args = args or []
        kwargs = kwargs or {}
        return ray.get([
            worker.execute.remote(lambda w: fn(*args, **kwargs))
            for worker in self.workers
        ])

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
        for colocator in self.colocators:
            del colocator

        for worker in self.workers:
            del worker

        self.colocators = []
        self.workers = []
