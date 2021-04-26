import logging
import ray
from horovod.ray.utils import map_blocking
from horovod.ray.worker import BaseHorovodWorker

logger = logging.getLogger(__name__)


class StrategyInterface:
    def create_workers(self):
        raise NotImplementedError

    @property
    def num_workers(self):
        raise NotImplementedError

    def shutdown(self):
        if self.placement_group:
            ray.util.remove_placement_group(self.placement_group)

        self.workers = []
        self.placement_group = None


class ColocatedStrategy(StrategyInterface):
    def __init__(self, *, settings, num_hosts, num_workers_per_host, use_gpu,
                 cpus_per_worker, gpus_per_worker):
        self.settings = settings
        self.num_hosts = num_hosts
        self.num_workers_per_host = num_workers_per_host
        self.use_gpu = use_gpu
        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker or 1

    def _resources_per_host(self):
        num_cpus = self.cpus_per_worker * self.num_workers_per_host
        num_gpus = self.gpus_per_worker * self.num_workers_per_host * int(
            self.use_gpu)
        return dict(CPU=num_cpus, GPU=num_gpus)

    def create_workers(self):
        bundles = [self._resources_per_host() for _ in range(self.num_hosts)]
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
        # Create num_workers_per_host workers per bundle, i.e. per machine.
        for bundle_index in range(len(bundles)):
            gpu_id_futures = []
            curr_node_workers = []
            remote_cls = ray.remote(BaseHorovodWorker)
            for i in range(self.num_workers_per_host):
                remote_cls_with_options = remote_cls.options(
                    num_cpus=self.cpus_per_worker,
                    num_gpus=self.gpus_per_worker * int(self.use_gpu),
                    placement_group=pg,
                    placement_group_bundle_index=bundle_index)
                worker = remote_cls_with_options.remote(
                    world_rank=self.num_workers_per_host * bundle_index + i,
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
                    set(gpu_ids)) == self.num_workers_per_host, gpu_ids
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

        # In some setups (i.e., Peloton), ray nodes may not have
        # unique host names.
        host_worker_map = {}
        hostnames = ray.get([w.hostname.remote() for w in node_workers])
        for hostname, worker in zip(hostnames, node_workers):
            host_worker_map[hostname] = worker

        return self.workers, list(host_worker_map.values())

    @property
    def num_workers(self):
        return self.num_hosts * self.num_workers_per_host


class PackStrategy(StrategyInterface):
    def __init__(self, *, settings, num_workers, use_gpu, cpus_per_worker,
                 gpus_per_worker):
        self.settings = settings
        self._num_workers = num_workers
        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker or 1
        self.use_gpu = use_gpu

    def resources_per_worker(self):
        num_cpus = self.cpus_per_worker
        num_gpus = self.gpus_per_worker * int(self.use_gpu)
        return dict(CPU=num_cpus, GPU=num_gpus)

    @property
    def num_workers(self):
        return self._num_workers

    def create_workers(self):
        bundles = [
            self.resources_per_worker() for _ in range(self.num_workers)
        ]
        pg = ray.util.placement_group(bundles, strategy="PACK")
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
        # node_workers = []
        for bundle_index in range(len(bundles)):
            remote_cls = ray.remote(BaseHorovodWorker)
            remote_cls_with_options = remote_cls.options(
                num_cpus=self.cpus_per_worker,
                num_gpus=self.gpus_per_worker * int(self.use_gpu),
                placement_group=pg,
                placement_group_bundle_index=bundle_index)
            worker = remote_cls_with_options.remote(
                world_rank=bundle_index, world_size=self.num_workers)
            # if self.use_gpu:
            #     gpu_id_futures.append(worker.get_gpu_ids.remote())
            self.workers.append(worker)

        hostnames = map_blocking(lambda w: w.hostname.remote(), self.workers)
        host_worker_map = {}
        for hostname, worker in zip(hostnames, self.workers):
            host_worker_map[hostname] = worker

        return self.workers, list(host_worker_map.values())
