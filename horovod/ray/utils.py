from typing import List, Optional, Callable, Any, Dict
from contextlib import contextmanager
import ray

from horovod.runner.driver import driver_service
from horovod.ray.driver_service import _driver_fn
from horovod.runner.util import network


class _DiscoveryActor:
    def execute(self, func: Callable) -> Any:
        """Executes an arbitrary function on self."""
        return func(self)


@contextmanager
def _maybe_create_workers(all_host_names: List[str],
                          existing_workers: Optional[List] = None):
    should_create = not bool(existing_workers)
    if should_create:
        existing_workers = []
        RemoteDiscoveryActor = ray.remote(_DiscoveryActor)
        for hostname in all_host_names:
            node_resource = {f"node:{hostname}": 0.001}
            temporary_worker = RemoteDiscoveryActor.options(
                resources=node_resource, num_cpus=0).remote()
            existing_workers.append(temporary_worker)
    try:
        yield existing_workers
    finally:
        if should_create:
            for worker in existing_workers:
                del worker


def detect_nics(settings,
                all_host_names: List[str],
                node_workers: Optional[List] = None) -> List[str]:
    """Returns available nics on all given nodes.

    Use `nics_to_env_var` to generate the appropriate environent variables
    to be used in starting Horovod.

    This is a decomposed version of driver_service.get_common_interfaces().

    If 'all_host_names' includes a remote hostname, Horovod will run a nic
    detection scheme that pings each adjacent host to find the right nic.

    Args:
        settings: Horovod Settings object.
        all_host_names (list): List of all host names, including localhost.
        node_workers (list): Optional list of Ray Actors. This list is used
            to conduct the detection scheme. If no list is provided,
            Ray will start some lightweight actors on each node and stop
            them after the nics are found.

    Returns:
        List of nics (str).
    """
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
                nics = _driver_fn(workers, all_host_names, local_host_names,
                                  settings)

                if settings.verbose >= 2:
                    print('Interfaces on all hosts were successfully checked.')
                    print('Common interface found: ' + ' '.join(nics))
    else:
        nics = driver_service.get_local_interfaces(settings)
    return nics


def nics_to_env_var(nics: List[str]) -> Dict[str, str]:
    """Converts a given list of available NICs to environment variables."""
    return {
        "HOROVOD_GLOO_IFACE": list(nics)[0],
        "NCCL_SOCKET_IFNAME": ",".join(nics),  # TODO
    }


def map_blocking(fn, collection):
    return ray.get([fn(w) for w in collection])
