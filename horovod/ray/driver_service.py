from horovod.runner.driver.driver_service import (_run_probe,
                                                  HorovodRunDriverService)


def _actor_launch_task_servers(*, node_actors, num_hosts, driver_addresses,
                               settings):
    for index, w in enumerate(node_actors):

        def execute_task_fn(_):
            from horovod.runner.task_fn import _task_fn
            _task_fn(index, num_hosts, driver_addresses, settings)

        w.execute.remote(execute_task_fn)


def _driver_fn(node_actors, all_host_names, local_host_names, settings):
    """Probes routable nics across all hostnames.

    Assumes the task service on each worker has already started.
    Have them register with the driver service.

    Launches the driver service. Each worker probes all the
    interfaces of the worker index + 1 (in a ring manner) and
    only keeps the routed interfaces.

    Returns the intersection of the set of all the routed interfaces
    on all the workers.

    :param all_host_names: list of addresses. for example,
        ['worker-0','worker-1']
        ['10.11.11.11', '10.11.11.12']
    :type all_host_names: list(string)
    :param local_host_names: host names that resolve into a local addresses.
    :type local_host_names: set
    :param settings: the object that contains the setting for running horovod
    :type settings: horovod.runner.common.util.settings.Settings
    :return: example: ['eth0', 'eth1']
    :rtype: list[string]
    """
    # Launch a TCP server called driver service on the host running horovod
    num_hosts = len(all_host_names)
    driver = HorovodRunDriverService(num_hosts, settings.key, settings.nics)
    if settings.verbose >= 2:
        print('Launched horovod server.')
    # Have all the workers register themselves with the service service.
    if len(node_actors) != num_hosts:
        raise ValueError(f"Number of node actors ({len(node_actors)}) "
                         f"must match num_hosts ({num_hosts}).")

    _actor_launch_task_servers(
        node_actors=node_actors,
        num_hosts=len(all_host_names),
        driver_addresses=driver.addresses(),
        settings=settings)

    if settings.verbose >= 2:
        print('Attempted to launch horovod task servers.')
    try:
        return _run_probe(driver, settings, num_hosts)
    finally:
        driver.shutdown()
