import logging
import ray
import time

from horovod.ray.runner import BaseHorovodWorker
from horovod.runner.http.http_server import RendezvousServer
from horovod.runner.util import network
from horovod.runner.gloo_run import (create_slot_env_vars, create_run_envs,
                                     register_shutdown_event)
from horovod.runner.elastic.rendezvous import create_rendezvous_handler
from horovod.runner.elastic.discovery import HostDiscovery
from horovod.runner.elastic.driver import ElasticDriver

logger = logging.getLogger(__name__)


class RayHostDiscovery(HostDiscovery):
    def __init__(self, use_gpu=False, cpus_per_slot=1):
        self.use_gpu = False
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
            host_mapping[hostname] = slots
        return host_mapping


class ElasticRayExecutor:
    def __init__(self, settings, cpus_per_slot: int = 1,
                 use_gpu: bool = False):
        if not isinstance(settings.discovery, RayHostDiscovery):
            settings.discovery = RayHostDiscovery(use_gpu, cpus_per_slot)
        self.settings = settings
        self.rendezvous = RendezvousServer(self.settings.verbose)
        self.driver = ElasticDriver(
            self.rendezvous,
            settings.discovery,
            settings.min_np,
            settings.max_np,
            timeout=settings.elastic_timeout,
            reset_limit=settings.reset_limit,
            verbose=settings.verbose)

    def start(self, get_common_interfaces, envs):
        self.envs = envs
        handler = create_rendezvous_handler(self.driver)
        global_rendezv_port = self.rendezvous.start(handler)
        self.driver.wait_for_available_slots(self.settings.num_proc)

        nics = get_common_interfaces(self.driver)
        server_ip = network.get_driver_ip(nics)
        self.run_envs = create_run_envs(
            server_ip, nics, global_rendezv_port, elastic=True)

    def create_resources(self, hostname):
        resources = {f"node:{hostname}": 0.01}
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
            future = worker.execute.remote(worker_fn)
            while not ray.get(future, timeout=0.1):
                if any(e.is_set() for e in events):
                    ray.kill(worker)
                    return 1, time.time()
            return 0, time.time()

        return create_worker

    def run(self, worker_fn):
        self.driver.start(self.settings.num_proc,
                          self.create_spawn_worker(worker_fn))
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
