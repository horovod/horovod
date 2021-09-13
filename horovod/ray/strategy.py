from collections import defaultdict
import logging
from typing import Dict

import ray
from ray.util.placement_group import get_current_placement_group
from horovod.ray.utils import map_blocking
from horovod.ray.worker import BaseHorovodWorker

logger = logging.getLogger(__name__)


def create_placement_group(resources_per_bundle: Dict[str, int],
                           num_bundles: int, pg_timeout: int,
                           pg_strategy: str):
    bundles = [resources_per_bundle.copy() for _ in range(num_bundles)]
    pg = ray.util.placement_group(bundles, strategy=pg_strategy)
    logger.debug("Waiting for placement group to start.")
    ready, _ = ray.wait([pg.ready()], timeout=pg_timeout)
    if ready:
        logger.debug("Placement group has started.")
    else:
        raise TimeoutError("Placement group creation timed out. Make sure "
                           "your cluster either has enough resources or use "
                           "an autoscaling cluster. Current resources "
                           "available: {}, resources requested by the "
                           "placement group: {}".format(
                               ray.available_resources(), pg.bundle_specs))

    return pg, bundles


class BaseStrategy:
    """Base class for implementing different placement strategies."""
    placement_group = None
    workers = None

    def create_workers(self):
        raise NotImplementedError

    @property
    def num_workers(self):
        raise NotImplementedError

    @classmethod
    def get_node_workers(cls, workers):
        """Returns list of one worker per node to use for NIC detection."""

        # In some setups (i.e., Peloton), ray nodes may not have
        # unique host names.
        hostnames = map_blocking(lambda w: w.hostname.remote(), workers)
        host_worker_map = {}
        for hostname, worker in zip(hostnames, workers):
            host_worker_map[hostname] = worker

        return list(host_worker_map.values())

    def shutdown(self):
        if self.placement_group:
            ray.util.remove_placement_group(self.placement_group)

        self.workers = []
        self.placement_group = None


class ColocatedStrategy(BaseStrategy):
    """Ensures that the workers are balanced across all hosts."""

    def __init__(self, *, settings, num_hosts: int, num_workers_per_host: int,
                 use_gpu: bool, cpus_per_worker: int, gpus_per_worker: int):
        self.settings = settings
        self.num_hosts = num_hosts
        self.num_workers_per_host = num_workers_per_host
        self.use_gpu = use_gpu
        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker or 1

    @property
    def num_workers(self) -> int:
        return int(self.num_hosts * self.num_workers_per_host)

    def _resources_per_host(self):
        num_cpus = self.cpus_per_worker * self.num_workers_per_host
        num_gpus = self.gpus_per_worker * self.num_workers_per_host * int(
            self.use_gpu)
        return dict(CPU=num_cpus, GPU=num_gpus)

    def create_workers(self):
        self.placement_group, bundles = create_placement_group(
            resources_per_bundle=self._resources_per_host(),
            num_bundles=self.num_hosts,
            pg_timeout=self.settings.placement_group_timeout_s,
            pg_strategy="STRICT_SPREAD")

        # Placement group has started. Now create the workers.
        self.workers = []

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
                    placement_group_capture_child_tasks=False,
                    placement_group=self.placement_group,
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

        return self.workers, self.get_node_workers(self.workers)


class PGStrategy(BaseStrategy):
    """Packs workers together but does not guarantee balanced hosts.

    Will use the current placement group if one is available.
    """

    def __init__(self, *, settings, num_workers, use_gpu, cpus_per_worker,
                 gpus_per_worker, placement_group=None, force_create_placement_group=False):
        self.settings = settings
        self._num_workers = num_workers
        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker or 1
        self.use_gpu = use_gpu
        if force_create_placement_group:
            self.placement_group = None
        else:
            self.placement_group = placement_group or get_current_placement_group()
        self._placement_group_bundles = self.placement_group.bundle_specs if self.placement_group else None
        self._created_placement_group = False

    @property
    def num_workers(self) -> int:
        return int(self._num_workers)

    def resources_per_worker(self):
        num_cpus = self.cpus_per_worker
        num_gpus = self.gpus_per_worker * int(self.use_gpu)
        return dict(CPU=num_cpus, GPU=num_gpus)

    def create_workers(self):
        if not self.placement_group:
            self.placement_group, _ = create_placement_group(
                resources_per_bundle=self.resources_per_worker(),
                num_bundles=self.num_workers,
                pg_strategy="PACK",
                pg_timeout=self.settings.placement_group_timeout_s)
            self._created_placement_group = True

        # Placement group has started. Now create the workers.
        self.workers = []
        remote_cls = ray.remote(BaseHorovodWorker)

        for worker_index in range(self.num_workers):
            remote_cls_with_options = remote_cls.options(
                num_cpus=self.cpus_per_worker,
                num_gpus=self.gpus_per_worker * int(self.use_gpu),
                placement_group_capture_child_tasks=False,
                placement_group=self.placement_group,
                placement_group_bundle_index=worker_index if self._created_placement_group else -1)
            worker = remote_cls_with_options.remote(
                world_rank=worker_index, world_size=self.num_workers)

            self.workers.append(worker)

        if self.use_gpu:
            node_ids = ray.get(
                [worker.node_id.remote() for worker in self.workers])
            gpus = ray.get(
                [worker.get_gpu_ids.remote() for worker in self.workers])
            node_workers = defaultdict(list)
            node_id_to_gpus = defaultdict(list)
            for worker, node_id, worker_gpu_ids in zip(self.workers, node_ids,
                                                       gpus):
                node_workers[node_id].append(worker)
                node_id_to_gpus[node_id].extend(worker_gpu_ids)

            futures = []
            for node_id, gpu_ids in node_id_to_gpus.items():
                all_ids = ",".join([str(gpu_id) for gpu_id in gpu_ids])

                for worker in node_workers[node_id]:
                    futures.append(
                        worker.update_env_vars.remote({
                            "CUDA_VISIBLE_DEVICES":
                            all_ids
                        }))
            ray.get(futures)
        return self.workers, self.get_node_workers(self.workers)

    def shutdown(self):
        if self._created_placement_group and self.placement_group:
            ray.util.remove_placement_group(self.placement_group)
            self.placement_group = None

        self.workers = []
