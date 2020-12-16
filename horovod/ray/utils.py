import logging
from contextlib import contextmanager
import ray

from horovod.runner.driver import driver_service
from horovod.ray.driver_service import _driver_fn
from horovod.runner.util import network


class _DiscoveryActor:
    def execute(self, func):
        """Executes an arbitrary function on self."""
        return func(self)


@contextmanager
def _maybe_create_workers(all_host_names, existing_workers=None):
    should_create = not bool(existing_workers)
    if should_create:
        existing_workers = []
        RemoteDiscoveryActor = ray.remote(_DiscoveryActor)
        for hostname in all_host_names:
            node_resource = {f"node:{hostname}": 0.001}
            temporary_worker = RemoteDiscoveryActor.options(
                resources=node_resource, num_cpus=0).remote()
            existing_workers.append(temporary_worker)

    yield existing_workers

    if should_create:
        for worker in existing_workers:
            del worker


def detect_nics(settings, all_host_names, node_workers=None):
    """Decomposed version of driver_service.get_common_interfaces()."""
    nics = None
    remote_host_names = network.filter_local_addresses(all_host_names)
    if len(remote_host_names) > 0:
        nics = settings.nics
        if not nics:
            with _maybe_create_workers(
                    all_host_names, existing_workers=node_workers) as workers:
                if settings.verbose >= 2:
                    print('Testing interfaces on all hosts.')

                local_host_names = set(all_host_names) - set(remote_host_names)
                nics = _driver_fn(workers, all_host_names,
                                  local_host_names, settings)

                if settings.verbose >= 2:
                    print('Interfaces on all hosts were successfully checked.')
                    print('Common interface found: ' + ' '.join(nics))
    else:
        nics = driver_service.get_local_interfaces(settings)
    return nics


def nics_to_env_var(nics):
    return {
        "HOROVOD_GLOO_IFACE": list(nics)[0],
        "NCCL_SOCKET_IFNAME": ",".join(nics),  # TODO
    }