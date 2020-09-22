import logging
import ray
import time
import os

from horovod.ray.runner import BaseHorovodWorker
from horovod.runner.common.util import timeout, hosts

from horovod.runner.http.http_server import RendezvousServer
from horovod.runner.util import network
from horovod.runner.gloo_run import (create_slot_env_vars, create_run_envs,
                                     register_shutdown_event, _get_min_start_hosts)
from horovod.runner.driver import driver_service
from horovod.runner.elastic import settings as elastic_settings
from horovod.runner.elastic.rendezvous import create_rendezvous_handler
from horovod.runner.elastic.discovery import HostDiscovery
from horovod.runner.elastic.driver import ElasticDriver

logger = logging.getLogger(__name__)


class RayHostDiscovery(HostDiscovery):
    def __init__(self, use_gpu=False, cpus_per_slot=1):
        self.use_gpu = use_gpu
        self.cpus_per_slot = cpus_per_slot

    def find_available_hosts_and_slots(self) -> dict:
        """Returns a dict mapping <hostname> -> <number of slots>."""
        alive_nodes = [k for k in ray.nodes() if k["alive"]]
        host_mapping = {}
        for node in alive_nodes:
            hostname = node["NodeManagerAddress"]
            resources = node["Resources"]
            slots = resources["CPU"] // self.cpus_per_slot
            if self.use_gpu:
                slots = min(slots, resources["GPU"])
            host_mapping[hostname] = int(slots)
        return host_mapping


class ElasticRayExecutor:
    @staticmethod
    def create_settings(
            min_np=1, max_np=None, elastic_timeout=600, timeout_s=60, ssh_identity_file=None, nics=None, **kwargs):
        start_timeout = timeout.Timeout(
            timeout_s,
            message="Timed out waiting for {activity}. Please "
            "check connectivity between servers. You "
            "may need to increase the --start-timeout "
            "parameter if you have too many servers.")
        ssh_identity_file = ssh_identity_file or os.path.expanduser("~/ray_bootstrap_key.pem")
        settings = elastic_settings.ElasticSettings(
            discovery=None,
            min_np=min_np,
            max_np=max_np,
            elastic_timeout=elastic_timeout,
            reset_limit=3,
            num_proc=min_np,
            ssh_identity_file=ssh_identity_file,
            nics=nics,
            start_timeout=start_timeout,
            **kwargs
            # ssh_port=args.ssh_port,
            # key=secret.make_secret_key(),
        )
        return settings

    def __init__(self, settings, cpus_per_slot: int = 1,
                 use_gpu: bool = False):
        if not isinstance(settings.discovery, RayHostDiscovery):
            settings.discovery = RayHostDiscovery(use_gpu, cpus_per_slot)
        self.cpus_per_slot = cpus_per_slot
        self.use_gpu = use_gpu
        self.settings = settings
        self.rendezvous = RendezvousServer(self.settings.verbose)
        self.driver = ElasticDriver(
            rendezvous=self.rendezvous,
            discovery=settings.discovery,
            min_np=settings.min_np,
            max_np=settings.max_np,
            timeout=settings.elastic_timeout,
            reset_limit=settings.reset_limit,
            verbose=settings.verbose)

    def start(self, envs=None):
        settings = self.settings
        def get_common_interfaces(driver):
            # Host-to-host common interface detection requires at least 2 hosts in an elastic job.
            min_hosts = _get_min_start_hosts(settings)
            current_hosts = driver.wait_for_available_slots(settings.num_proc, min_hosts=min_hosts)
            return driver_service.get_common_interfaces(settings, current_hosts.host_assignment_order)

        self.envs = envs or {}
        handler = create_rendezvous_handler(self.driver)
        global_rendezv_port = self.rendezvous.start(handler)
        self.driver.wait_for_available_slots(self.settings.num_proc)

        nics = get_common_interfaces(self.driver)
        server_ip = network.get_driver_ip(nics)
        self.run_envs = create_run_envs(
            server_ip, nics, global_rendezv_port, elastic=True)

    def create_resources(self, hostname):
        resources = dict(
            num_cpus=self.cpus_per_slot,
            num_gpus=int(self.use_gpu),
            resources={f"node:{hostname}": 0.01}
        )
        return resources

    def create_spawn_worker_fn(self, worker_fn):
        self.remote_worker_cls = ray.remote(BaseHorovodWorker)
        # event = register_shutdown_event()
        worker_envs = {}
        worker_envs.update(self.run_envs.copy())
        worker_envs.update(self.envs.copy())
        worker_envs.update(PYTHONUNBUFFERED="1")

        def create_worker(slot_info, events):
            hostname = slot_info.hostname
            loaded_worker_cls = self.remote_worker_cls.options(
                **self.create_resources(hostname))

            worker = loaded_worker_cls.remote()
            worker.update_env_vars.remote(worker_envs)
            worker.update_env_vars.remote(create_slot_env_vars(slot_info))
            visible_devices = ",".join(
                [str(i) for i in range(slot_info.local_size)])
            worker.update_env_vars.remote({"CUDA_VISIBLE_DEVICES": visible_devices})
            future = worker.execute.remote(worker_fn)
            def get_or_fail():
                try:
                    ray.get(future, timeout=0.1)
                    # Success
                    return 0, time.time()
                except ray.exceptions.GetTimeoutError:
                    # Timeout
                    if any(e.is_set() for e in events):
                        ray.kill(worker)
                        return 1, time.time()
                except Exception:
                    # Fail
                    return 1, time.time()
            result = None
            while not result:
                result = get_or_fail()
            return result



        return create_worker

    def run(self, worker_fn):
        self.driver.start(self.settings.num_proc,
                          self.create_spawn_worker_fn(worker_fn))
        res = self.driver.get_results()
        self.driver.stop()

        if res.error_message is not None:
            raise RuntimeError(res.error_message)

        for name, value in sorted(
                res.worker_results.items(), key=lambda item: item[1][1]):
            exit_code, timestamp = value
            if exit_code != 0:
                raise RuntimeError(
                    'Horovod detected that one or more processes exited with non-zero '
                    'status, thus causing the job to be terminated. The first process '
                    'to do so was:\nProcess name: {name}\nExit code: {code}\n'
                    .format(name=name, code=exit_code))
