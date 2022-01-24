import ray
from ray.util.placement_group import get_current_placement_group

from collections import defaultdict
from dataclasses import dataclass, asdict
import os
from typing import Dict, Callable, Any, Optional, List
import logging
import ray.exceptions
from horovod.ray.adapter import Adapter, BaseParams

from horovod.runner.common.util import secret, timeout, hosts
from horovod.runner.http.http_server import RendezvousServer
from horovod.ray.utils import detect_nics, nics_to_env_var, map_blocking
from horovod.ray.strategy import ColocatedStrategy, PGStrategy
from horovod.ray.elastic_v2 import ElasticParams

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
    placement_group_timeout_s: int = 100
    elastic: bool = False

    @property
    def start_timeout(self):
        return timeout.Timeout(
            self.timeout_s,
            message="Timed out waiting for {activity}. Please "
            "check connectivity between servers. You "
            "may need to increase the --start-timeout "
            "parameter if you have too many servers.")


class Coordinator:
    """Responsible for instantiating the Rendezvous server.

    Args:
        settings: Horovod Settings object."""
    rendezvous = None
    global_rendezv_port = None
    nics = None
    node_id_by_rank = None

    def __init__(
            self,
            settings,
    ):
        self.settings = settings
        self.node_id_by_rank = defaultdict(list)
        self._hostnames = set()

    @property
    def world_size(self) -> int:
        return sum(len(ranks) for ranks in self.node_id_by_rank.values())

    @property
    def hostnames(self):
        return self._hostnames

    @property
    def node_id_string(self) -> str:
        return ",".join([
            f"{node_id}:{len(ranks)}"
            for node_id, ranks in self.node_id_by_rank.items()
        ])

    def register(self, hostname: str, node_id: str, world_rank: int):
        self._hostnames.add(hostname)
        self.node_id_by_rank[node_id].append(world_rank)

    def finalize_registration(self) -> dict:
        """Return a dictionary for all ranks."""
        rank_to_info = {}

        cross_sizes = defaultdict(int)
        cross_ranks = {}
        for rank_list in self.node_id_by_rank.values():
            for local_rank, world_rank in enumerate(rank_list):
                cross_ranks[world_rank] = cross_sizes[local_rank]
                cross_sizes[local_rank] += 1

        for node_world_rank, (node_id, ranks) in enumerate(
                self.node_id_by_rank.items()):
            for local_rank, world_rank in enumerate(ranks):
                rank_to_info[world_rank] = dict(
                    HOROVOD_CROSS_RANK=cross_ranks[world_rank],
                    HOROVOD_CROSS_SIZE=cross_sizes[local_rank],
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
        parsed_node_ids = hosts.parse_hosts(hosts_string=self.node_id_string)
        host_alloc_plan = hosts.get_host_assignments(parsed_node_ids,
                                                     self.world_size)

        # start global rendezvous server and get port that it is listening on
        self.global_rendezv_port = self.rendezvous.start()
        self.rendezvous.init(host_alloc_plan)

        return {
            # needs to be real address
            "HOROVOD_GLOO_RENDEZVOUS_ADDR": ray.util.get_node_ip_address(),
            "HOROVOD_GLOO_RENDEZVOUS_PORT": str(self.global_rendezv_port),
            "HOROVOD_CONTROLLER": "gloo",
            "HOROVOD_CPU_OPERATIONS": "gloo",
        }


@dataclass
class StaticParams(BaseParams):
    """Parameters for non-elastic jobs.

    Args:
        num_workers (int): Number of workers to use for training.
        cpus_per_worker (int): Number of CPU resources to allocate to
            each worker.
        use_gpu (bool): Whether to use GPU for allocation. TODO: this
            can be removed.
        gpus_per_worker (int): Number of GPU resources to allocate to
            each worker.
        num_hosts (int): Alternative API to ``num_workers``. Number of
            machines to execute the job on. Used to enforce equal number of
            workers on each machine.
        num_workers_per_host (int): Alternative API to
            ``num_workers``. Number of workers to be placed on each machine.
            Used to enforce equal number of workers on each machine. Only
            used in conjunction with `num_hosts`.
        use_current_placement_group (bool): Whether to use the current
            placement group instead of creating a new one. Defaults to True.

    """
    num_workers: Optional[int] = None
    num_hosts: Optional[int] = None
    num_workers_per_host: int = 1
    use_current_placement_group: bool = True

    @property
    def elastic(self):
        return False

    @property
    def adapter(self):
        return StaticAdapter

class RayExecutor:
    """Job class for Horovod + Ray integration.

    Args:
        settings (horovod.Settings): Configuration for job setup. You can
            use a standard Horovod Settings object or create one directly
            from RayExecutor.create_settings.
        num_workers (int): Number of workers to use for training.
        cpus_per_worker (int): Number of CPU resources to allocate to
            each worker.
        use_gpu (bool): Whether to use GPU for allocation. TODO: this
            can be removed.
        gpus_per_worker (int): Number of GPU resources to allocate to
            each worker.
        num_hosts (int): Alternative API to ``num_workers``. Number of
            machines to execute the job on. Used to enforce equal number of
            workers on each machine.
        num_workers_per_host (int): Alternative API to
            ``num_workers``. Number of workers to be placed on each machine.
            Used to enforce equal number of workers on each machine. Only
            used in conjunction with `num_hosts`.
        use_current_placement_group (bool): Whether to use the current
            placement group instead of creating a new one. Defaults to True.
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
        elastic_timeout (int): Timeout for elastic initialisation after
            re-scaling the cluster. The default value is 600 seconds.
            Alternatively, the environment variable
            HOROVOD_ELASTIC_TIMEOUT can also be used.
        override_discovery (bool): Whether for the ElasticRayExecutor to
            automatically provide a discovery mechanism for ElasticSettings.

    """

    @classmethod
    def create_settings(cls,
                        timeout_s=30,
                        ssh_identity_file=None,
                        ssh_str=None,
                        placement_group_timeout_s=100,
                        nics=None):
        """Create a mini setting object.

        Args:
            timeout_s (int): Timeout parameter for Gloo rendezvous.
            ssh_identity_file (str): Path to the identity file to
                ssh into different hosts on the cluster.
            ssh_str (str): CAUTION WHEN USING THIS. Private key
                file contents. Writes the private key to ssh_identity_file.
            placement_group_timeout_s (int): Timeout parameter for Ray
                Placement Group creation.
            nics (set): Network interfaces that can be used for communication.

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
            placement_group_timeout_s=placement_group_timeout_s,
            nics=nics)

    def __init__(
            self,
            settings,
            num_workers: Optional[int] = None,
            num_hosts: Optional[int] = None,
            num_workers_per_host: int = 1,
            cpus_per_worker: int = 1,
            use_gpu: bool = False,
            gpus_per_worker: Optional[int] = None,
            use_current_placement_group: bool = True,

            min_workers: int = None,
            max_workers: int = None,
            reset_limit: int = None,
            cooldown_range: List[int] = None,
            elastic_timeout: int = 600,
            override_discovery: bool = True
            ):
        if max_workers and (not min_workers or min_workers <= 0):
            raise ValueError("`max_workers` provided without any positive `min_workers`"
                            "Elastic workloads require a positive `min_workers`")
        if min_workers and num_workers:
            raise ValueError("Both `min_workers` and `num_workers` provided."
                             "Only one of the above is allowed as workloads cannot be elastic and non-elastic.")

        if min_workers is not None:
            self.params = ElasticParams(
                min_workers=min_workers,
                max_workers=max_workers,
                reset_limit=reset_limit,
                cooldown_range=cooldown_range,
                elastic_timeout=elastic_timeout,
                override_discovery=override_discovery,
                cpus_per_worker=cpus_per_worker,
                use_gpu=use_gpu,
                gpus_per_worker=gpus_per_worker
            )
        else:
            self.params = StaticParams(
                num_workers=num_workers,
                num_hosts=num_hosts,
                num_workers_per_host=num_workers_per_host,
                cpus_per_worker=cpus_per_worker,
                use_gpu=use_gpu,
                gpus_per_worker=gpus_per_worker,
                use_current_placement_group=use_current_placement_group
            )
        self.settings = settings
        self.settings.elastic = self.params.elastic

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
        self._initialize_adapter()

        kwargs_ = dict(
            executable_cls=executable_cls,
            executable_args=executable_args,
            executable_kwargs=executable_kwargs,
            extra_env_vars=extra_env_vars)
        return self._maybe_call_ray(self.adapter.start, **kwargs_)

    def _initialize_adapter(self):
        kwargs = asdict(self.params)
        logger.debug(f"Kwargs: {kwargs}")
        Adapter = self.params.adapter
        self._is_remote = False
        if ray.util.client.ray.is_connected():
            RemoteAdapter = ray.remote(Adapter)
            self.adapter = RemoteAdapter.remote(self.settings, **kwargs)
            self._is_remote = True
        else:
            self.adapter= Adapter(self.settings, **kwargs)

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
        kwargs_ = dict(fn=fn, callbacks=callbacks)
        # invoke run_remote
        return self._maybe_call_ray(self.adapter.execute, **kwargs_)

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
        kwargs_ = dict(fn=fn, args=args, kwargs=kwargs, callbacks=callbacks)
        return self._maybe_call_ray(self.adapter.run, **kwargs_)

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
        kwargs_ = dict(fn=fn, args=args, kwargs=kwargs)
        return self._maybe_call_ray(self.adapter.run_remote, **kwargs_)

    def execute_single(self,
                       fn: Callable[["executable_cls"], Any]) -> List[Any]:
        """Executes the provided function on the rank 0 worker (chief).

        Args:
            fn: Target function to be invoked on the chief object.

        Returns:
            Deserialized return values from the target function.
        """
        kwargs = dict(fn=fn)
        return self._maybe_call_ray(self.adapter.execute_single, **kwargs)

    def shutdown(self):
        """Destroys the provided workers."""
        result = self._maybe_call_ray(self.adapter.shutdown)
        del self.adapter
        return result

    def _maybe_call_ray(self, driver_func, *args, **kwargs):
        if self._is_remote:
            return ray.get(driver_func.remote(*args, **kwargs))
        else:
            return driver_func(**kwargs)


class StaticAdapter(Adapter):
    """Adapter for executing Ray calls for non-elastic Horovod jobs.

    Args:
        settings (horovod.Settings): Configuration for job setup. You can
            use a standard Horovod Settings object or create one directly
            from RayExecutor.create_settings.
        num_workers (int): Number of workers to use for training.
        cpus_per_worker (int): Number of CPU resources to allocate to
            each worker.
        use_gpu (bool): Whether to use GPU for allocation. TODO: this
            can be removed.
        gpus_per_worker (int): Number of GPU resources to allocate to
            each worker.
        num_hosts (int): Alternative API to ``num_workers``. Number of
            machines to execute the job on. Used to enforce equal number of
            workers on each machine.
        num_workers_per_host (int): Alternative API to
            ``num_workers``. Number of workers to be placed on each machine.
            Used to enforce equal number of workers on each machine. Only
            used in conjunction with `num_hosts`.
        use_current_placement_group (bool): Whether to use the current
            placement group instead of creating a new one. Defaults to True.

    """
    def __init__(self,
                 settings,
                 num_workers: Optional[int] = None,
                 num_hosts: Optional[int] = None,
                 num_workers_per_host: int = 1,
                 cpus_per_worker: int = 1,
                 use_gpu: bool = False,
                 gpus_per_worker: Optional[int] = None,
                 use_current_placement_group: bool = True):

        self.settings = settings
        self.num_workers = num_workers
        self.num_hosts = num_hosts
        self.num_workers_per_host = num_workers_per_host
        self.cpus_per_worker = cpus_per_worker
        self.use_gpu = use_gpu
        self.gpus_per_worker = gpus_per_worker or 1
        self.use_current_placement_group = use_current_placement_group

        self.workers = []
        self.strategy = None

    def _start_executables(self, executable_cls, executable_args,
                           executable_kwargs):
        def _start_exec(worker):
            return worker.start_executable.remote(
                executable_cls, executable_args, executable_kwargs)

        map_blocking(_start_exec, self.workers)

    def _create_strategy(self):
        assert self.num_workers is None or self.num_hosts is None
        use_pg = self.use_current_placement_group and get_current_placement_group()
        if self.num_workers or use_pg:
            if use_pg:
                logger.info(
                    "Found an existing placement group, inheriting. "
                    "You can disable this behavior by setting "
                    "`use_current_placement_group=False`."
                )
            num_workers = self.num_workers or self.num_workers_per_host * self.num_hosts
            return PGStrategy(
                settings=self.settings,
                num_workers=num_workers,
                use_gpu=self.use_gpu,
                cpus_per_worker=self.cpus_per_worker,
                gpus_per_worker=self.gpus_per_worker,
                force_create_placement_group=not self.use_current_placement_group)
        else:
            return ColocatedStrategy(
                settings=self.settings,
                num_hosts=self.num_hosts,
                num_workers_per_host=self.num_workers_per_host,
                use_gpu=self.use_gpu,
                cpus_per_worker=self.cpus_per_worker,
                gpus_per_worker=self.gpus_per_worker)

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

        self.strategy = self._create_strategy()
        self.coordinator = Coordinator(self.settings)
        executable_args = executable_args or []
        self.workers, node_workers = self.strategy.create_workers()
        # Get all the hostnames of all workers
        node_ids = map_blocking(lambda w: w.node_id.remote(), self.workers)
        hostnames = map_blocking(lambda w: w.hostname.remote(), self.workers)
        # Register each hostname to the coordinator. assumes the hostname
        # ordering is the same.
        for rank, (hostname, node_id) in enumerate(zip(hostnames, node_ids)):
            self.coordinator.register(hostname, node_id, rank)
        all_info = self.coordinator.finalize_registration()

        indexed_runners = dict(enumerate(self.workers))
        for rank, local_cross_env_var in all_info.items():
            indexed_runners[rank].update_env_vars.remote(local_cross_env_var)

        coordinator_envs = self.coordinator.establish_rendezvous()
        coordinator_envs.update(extra_env_vars)
        nics = detect_nics(
            self.settings,
            all_host_names=list(self.coordinator.hostnames),
            node_workers=node_workers)
        coordinator_envs.update(nics_to_env_var(nics))

        map_blocking(lambda w: w.update_env_vars.remote(coordinator_envs),
                     self.workers)

        self._start_executables(executable_cls, executable_args,
                                executable_kwargs)

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
        return ray.get(self._run_remote(fn))

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

        Returns:
            Deserialized return values from the target function.
        """
        args = args or []
        kwargs = kwargs or {}
        f = lambda w: fn(*args, **kwargs)
        return ray.get(self._run_remote(fn=f))

    def run_remote(self,
                   fn: Callable[[Any], Any],
                   args: Optional[List] = None,
                   kwargs: Optional[Dict] = None,
                   callbacks: Optional[List[Callable]] = None):

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
        f = lambda w: fn(*args, **kwargs)
        return self._run_remote(fn=f)

    def _run_remote(self,
                   fn: Callable[[Any], Any]) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            fn: Target function that can be executed with arbitrary
                args and keyword arguments.

        Returns:
            list: List of ObjectRefs that you can run `ray.get` on to
                retrieve values.
        """
        # Use run_remote for all calls
        # for elastic, start the driver and launch the job
        return [
            worker.execute.remote(fn) for worker in self.workers
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
        """Destroys the workers."""
        for worker in self.workers:
            del worker

        if self.strategy:
            self.strategy.shutdown()
