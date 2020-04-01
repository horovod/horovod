# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import platform
import threading

import pyspark
from six.moves import queue

from horovod.spark.task import task_service
from horovod.spark.gloo_run import gloo_run
from horovod.spark.mpi_run import mpi_run
from horovod.run.runner import run_controller
from horovod.run.common.util import timeout, host_hash, secret
from horovod.run.common.util import settings as hvd_settings
from horovod.spark.driver import driver_service, job_id


# Spark will fail to initialize correctly locally on Mac OS without this
if platform.system() == 'Darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'


def _task_fn(index, driver_addresses, settings, use_gloo):
    task = task_service.SparkTaskService(index, settings.key, settings.nics)
    try:
        driver_client = driver_service.SparkDriverClient(driver_addresses, settings.key, settings.verbose)
        driver_client.register_task(index, task.addresses(), host_hash.host_hash())
        task.wait_for_initial_registration(settings.timeout)
        # Tasks ping each other in a circular fashion to determine interfaces reachable within
        # the cluster.
        next_task_index = (index + 1) % settings.num_proc
        next_task_addresses = driver_client.all_task_addresses(next_task_index)
        # We request interface matching to weed out all the NAT'ed interfaces.
        next_task_client = \
            task_service.SparkTaskClient(next_task_index, next_task_addresses,
                                         settings.key, settings.verbose,
                                         match_intf=True)
        driver_client.register_task_to_task_addresses(next_task_index, next_task_client.addresses())
        task_indices_on_this_host = driver_client.task_host_hash_indices(
            host_hash.host_hash())
        # With Gloo all tasks wait for the command
        # With MPI task with first index executes orted which will run mpirun_exec_fn for all tasks.
        if use_gloo or task_indices_on_this_host[0] == index:
            task.wait_for_command_start(settings.timeout)
            task.wait_for_command_termination()
        else:
            # The rest of tasks need to wait for the first task to finish.
            first_task_addresses = driver_client.all_task_addresses(task_indices_on_this_host[0])
            first_task_client = \
                task_service.SparkTaskClient(task_indices_on_this_host[0],
                                             first_task_addresses, settings.key,
                                             settings.verbose)
            first_task_client.wait_for_command_termination()
        return task.fn_result()
    finally:
        task.shutdown()


def _make_mapper(driver_addresses, settings, use_gloo):
    def _mapper(index, _):
        yield _task_fn(index, driver_addresses, settings, use_gloo)
    return _mapper


def _make_spark_thread(spark_context, spark_job_group, driver, result_queue,
                       settings, use_gloo):
    """Creates `settings.num_proc` Spark tasks in a parallel thread."""
    def run_spark():
        """Creates `settings.num_proc` Spark tasks, each executing `_task_fn` and waits for them to terminate."""
        try:
            spark_context.setJobGroup(spark_job_group,
                                      "Horovod Spark Run",
                                      interruptOnCancel=True)
            procs = spark_context.range(0, numSlices=settings.num_proc)
            # We assume that folks caring about security will enable Spark RPC
            # encryption, thus ensuring that key that is passed here remains
            # secret.
            result = procs.mapPartitionsWithIndex(_make_mapper(driver.addresses(), settings, use_gloo)).collect()
            result_queue.put(result)
        except:
            driver.notify_spark_job_failed()
            raise

    spark_thread = threading.Thread(target=run_spark)
    spark_thread.start()
    return spark_thread


def _launch_job(use_mpi, use_gloo, settings, driver, env, stdout=None, stderr=None):
    # Determine a set of common interfaces for task-to-task communication.
    nics = set(driver.task_addresses_for_tasks(0).keys())
    for index in range(1, settings.num_proc):
        nics.intersection_update(driver.task_addresses_for_tasks(index).keys())
    if not nics:
        raise Exception('Unable to find a set of common task-to-task communication interfaces: %s'
                        % [(index, driver.task_addresses_for_tasks(index)) for index in range(settings.num_proc)])

    if env is None:
        env = os.environ.copy()

    run_controller(use_gloo, lambda: gloo_run(settings, nics, driver, env),
                   use_mpi, lambda: mpi_run(settings, nics, driver, env, stdout, stderr),
                   False, lambda: None,
                   settings.verbose)


def run(fn, args=(), kwargs={}, num_proc=None, start_timeout=None,
        use_mpi=None, use_gloo=None, extra_mpi_args=None,
        env=None, stdout=None, stderr=None, verbose=1, nics=None):
    """
    Runs Horovod in Spark.  Runs `num_proc` processes executing `fn` using the same amount of Spark tasks.

    Args:
        fn: Function to run.
        args: Arguments to pass to `fn`.
        kwargs: Keyword arguments to pass to `fn`.
        num_proc: Number of Horovod processes.  Defaults to `spark.default.parallelism`.
        start_timeout: Timeout for Spark tasks to spawn, register and start running the code, in seconds.
                       If not set, falls back to `HOROVOD_SPARK_START_TIMEOUT` environment variable value.
                       If it is not set as well, defaults to 600 seconds.
        extra_mpi_args: Extra arguments for mpi_run. Defaults to no extra args.
        env: Environment dictionary to use in Horovod run.  Defaults to `os.environ`.
        stdout: Horovod stdout is redirected to this stream. Defaults to sys.stdout.
        stderr: Horovod stderr is redirected to this stream. Defaults to sys.stderr.
        verbose: Debug output verbosity (0-2). Defaults to 1.
        nics: List of NICs for tcp network communication.

    Returns:
        List of results returned by running `fn` on each rank.
    """

    if start_timeout is None:
        # Lookup default timeout from the environment variable.
        start_timeout = int(os.getenv('HOROVOD_SPARK_START_TIMEOUT', '600'))

    # nics needs to be a set
    if nics and not isinstance(nics, set):
        nics = set(nics)

    tmout = timeout.Timeout(start_timeout,
                            message='Timed out waiting for {activity}. Please check that you have '
                                    'enough resources to run all Horovod processes. Each Horovod '
                                    'process runs in a Spark task. You may need to increase the '
                                    'start_timeout parameter to a larger value if your Spark resources '
                                    'are allocated on-demand.')
    settings = hvd_settings.Settings(verbose=verbose,
                                     extra_mpi_args=extra_mpi_args,
                                     key=secret.make_secret_key(),
                                     timeout=tmout,
                                     nics=nics,
                                     run_func_mode=True)

    spark_context = pyspark.SparkContext._active_spark_context
    if spark_context is None:
        raise Exception('Could not find an active SparkContext, are you '
                        'running in a PySpark session?')

    if num_proc is None:
        num_proc = spark_context.defaultParallelism
        if settings.verbose >= 1:
            print('Running %d processes (inferred from spark.default.parallelism)...' % num_proc)
    else:
        if settings.verbose >= 1:
            print('Running %d processes...' % num_proc)
    settings.num_proc = num_proc

    result_queue = queue.Queue(1)

    # start Spark driver service and launch settings.num_proc Spark tasks
    spark_job_group = 'horovod.spark.run.%d' % job_id.next_job_id()
    driver = driver_service.SparkDriverService(settings.num_proc, fn, args, kwargs,
                                               settings.key, settings.nics)
    spark_thread = _make_spark_thread(spark_context, spark_job_group, driver,
                                      result_queue, settings, use_gloo)
    try:
        # wait for all tasks to register and notify them
        driver.wait_for_initial_registration(settings.timeout)
        if settings.verbose >= 2:
            print('Initial Spark task registration is complete.')
        task_clients = [
            task_service.SparkTaskClient(index,
                                         driver.task_addresses_for_driver(index),
                                         settings.key, settings.verbose)
            for index in range(settings.num_proc)]
        for task_client in task_clients:
            task_client.notify_initial_registration_complete()
        driver.wait_for_task_to_task_address_updates(settings.timeout)
        if settings.verbose >= 2:
            print('Spark task-to-task address registration is complete.')

        # Determine the index grouping based on host hashes.
        # Barrel shift until index 0 is in the first host.
        host_hashes = list(driver.task_host_hash_indices().keys())
        host_hashes.sort()
        while 0 not in driver.task_host_hash_indices()[host_hashes[0]]:
            host_hashes = host_hashes[1:] + host_hashes[:1]

        settings.hosts = ','.join('%s:%d' % (host_hash, len(driver.task_host_hash_indices()[host_hash]))
                                  for host_hash in host_hashes)

        # Determine the ranks to indicies
        ranks_to_indices = []
        for host_hash in host_hashes:
            ranks_to_indices += driver.task_host_hash_indices()[host_hash]
        driver.set_ranks_to_indices(ranks_to_indices)

        # Run the job
        _launch_job(use_mpi, use_gloo, settings, driver, env, stdout, stderr)
    except:
        # Terminate Spark job.
        spark_context.cancelJobGroup(spark_job_group)

        # Re-raise exception.
        raise
    finally:
        spark_thread.join()
        driver.shutdown()

    # Make sure Spark Job did not fail.
    driver.check_for_spark_job_failure()

    # If there's no exception, execution results are in this queue.
    results = result_queue.get_nowait()
    return [results[index] for index in ranks_to_indices]
