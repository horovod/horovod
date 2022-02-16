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

import logging
import os
import platform
import queue
import sys
import time

import pyspark

from horovod.common.util import gloo_built
from horovod.runner.util.threads import in_thread
from horovod.spark.common.util import host_hash
from horovod.spark.task import task_service
from horovod.spark.gloo_run import gloo_run, gloo_run_elastic
from horovod.spark.mpi_run import mpi_run
from horovod.runner.launch import is_gloo_used, run_controller
from horovod.runner.common.util import timeout, secret
from horovod.runner.common.util import settings as hvd_settings
from horovod.runner.elastic import settings as hvd_elastic_settings
from horovod.spark.driver import driver_service, host_discovery, job_id


MINIMUM_COMMAND_LIFETIME_S = 3
WAIT_FOR_COMMAND_START_DELAY_SECONDS = 0.1
WAIT_FOR_SHUTDOWN_DELAY_SECONDS = 0.1


# Spark will fail to initialize correctly locally on Mac OS without this
if platform.system() == 'Darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'


def _task_fn(index, driver_addresses, key, settings, use_gloo, is_elastic):
    # deserialized on Spark workers, settings do not contain the key, so it is given here explicitly
    # Spark RPC communicates the key and supports encryption
    # for convenience, we put it back into settings
    settings.key = key

    # to simplify things, each task is an individual host in Elastic Horovod on Spark
    # further, each attempt (instance) of a task is an individual host in Elastic Horovod on Spark
    # hides availability of shared memory among executors on the same Spark node
    hosthash = host_hash(salt='{}-{}'.format(index, time.time()) if is_elastic else None)

    # provide host hash to mpirun_exec_fn.py via task service
    # gloo_exec_fn.py will get this env var set in request env as well
    os.environ['HOROVOD_HOSTNAME'] = hosthash

    task = task_service.SparkTaskService(index, settings.key, settings.nics,
                                         MINIMUM_COMMAND_LIFETIME_S if is_elastic or use_gloo else None,
                                         settings.verbose)
    try:
        driver_client = driver_service.SparkDriverClient(driver_addresses, settings.key, settings.verbose)
        driver_client.register_task(index, task.addresses(), hosthash)

        if not is_elastic:
            task.wait_for_initial_registration(settings.start_timeout)
            task_indices_on_this_host = driver_client.task_host_hash_indices(hosthash)
            local_rank_zero_index = task_indices_on_this_host[0]
        else:
            local_rank_zero_index = None

        # In elastic all tasks wait for task shutdown signal from driver.
        # With Gloo all tasks wait for the command to start and terminate.
        # With MPI task with first index executes orted which will run mpirun_exec_fn for all tasks.
        if is_elastic:
            # either terminate on task shutdown or command termination
            shutdown_thread = in_thread(driver_client.wait_for_task_shutdown)

            while shutdown_thread.is_alive():
                # Once the command started we wait for its termination
                if task.check_for_command_start(WAIT_FOR_COMMAND_START_DELAY_SECONDS):
                    task.wait_for_command_termination()
                    if task.command_exit_code() != 0:
                        raise Exception('Command failed, making Spark task fail to restart the task')
                    break

                # While no command started, we can shutdown any time
                shutdown_thread.join(WAIT_FOR_SHUTDOWN_DELAY_SECONDS)
        elif use_gloo or index == local_rank_zero_index:
            # Either Gloo or first task with MPI.
            task.wait_for_command_start(settings.start_timeout)
            task.wait_for_command_termination()
        else:
            # The other tasks with MPI need to wait for the first task to finish.
            first_task_addresses = driver_client.all_task_addresses(local_rank_zero_index)
            first_task_client = \
                task_service.SparkTaskClient(local_rank_zero_index,
                                             first_task_addresses, settings.key,
                                             settings.verbose)
            first_task_client.wait_for_command_termination()

        return task.fn_result()
    finally:
        # we must not call into shutdown too quickly, task clients run a command
        # and want to wait on the result, we have told task service not to return
        # from wait_for_command_termination too quickly, so we are safe here to shutdown
        # clients have had enough time to connect to the service already
        #
        # the shutdown has to block on running requests (wait_for_command_exit_code)
        # so they can finish serving the exit code
        # shutdown does block with network.BasicService._server._block_on_close = True
        task.shutdown()


def _make_mapper(driver_addresses, settings, use_gloo, is_elastic):
    # serialised settings do not have a key so we have to copy it and provide it explicitly here
    key = settings.key

    def _mapper(index, _):
        yield _task_fn(index, driver_addresses, key, settings, use_gloo, is_elastic)

    return _mapper


def _make_spark_thread(spark_context, spark_job_group, driver, result_queue,
                       settings, use_gloo, is_elastic):
    """Creates `settings.num_proc` Spark tasks in a parallel thread."""
    def run_spark():
        """Creates `settings.num_proc` Spark tasks, each executing `_task_fn` and waits for them to terminate."""
        try:
            spark_context.setJobGroup(spark_job_group, "Horovod Spark Run", interruptOnCancel=True)
            procs = spark_context.range(0, numSlices=settings.max_np if settings.elastic else settings.num_proc)
            # We assume that folks caring about security will enable Spark RPC encryption,
            # thus ensuring that key that is passed here remains secret.
            mapper = _make_mapper(driver.addresses(), settings, use_gloo, is_elastic)
            result = procs.mapPartitionsWithIndex(mapper).collect()
            result_queue.put(result)
        except:
            driver.notify_spark_job_failed()
            raise

    spark_thread = in_thread(target=run_spark, daemon=False)
    return spark_thread


def _launch_job(use_mpi, use_gloo, settings, driver, env, stdout=None, stderr=None, executable=None):
    nics = driver.get_common_interfaces()
    executable = executable or sys.executable
    run_controller(use_gloo, lambda: gloo_run(executable, settings, nics, driver, env, stdout, stderr),
                   use_mpi, lambda: mpi_run(executable, settings, nics, driver, env, stdout, stderr),
                   False, lambda: None,
                   settings.verbose)


def _register_task_addresses(driver, settings):
    _notify_and_register_task_addresses(driver, settings, notify=False)


def _notify_and_register_task_addresses(driver, settings, notify=True):
    # wait for num_proc tasks to register
    driver.wait_for_initial_registration(settings.start_timeout)
    if settings.verbose >= 2:
        logging.info('Initial Spark task registration is complete.')

    task_indices = driver.task_indices()
    task_pairs = zip(task_indices, task_indices[1:] + task_indices[0:1])

    def notify_and_register(task_index, next_task_index):
        task_client = task_service.SparkTaskClient(task_index,
                                                   driver.task_addresses_for_driver(task_index),
                                                   settings.key, settings.verbose)

        if notify:
            task_client.notify_initial_registration_complete()

        next_task_addresses = driver.all_task_addresses(next_task_index)
        task_to_task_addresses = task_client.get_task_addresses_for_task(next_task_index, next_task_addresses)
        driver.register_task_to_task_addresses(next_task_index, task_to_task_addresses)

    for task_index, next_task_index in task_pairs:
        in_thread(notify_and_register, (task_index, next_task_index))

    driver.wait_for_task_to_task_address_updates(settings.start_timeout)

    if settings.verbose >= 2:
        logging.info('Spark task-to-task address registration is complete.')


def _get_indices_in_rank_order(driver):
    ranks_to_indices = driver.get_ranks_to_indices()
    return [index for _, index in sorted(ranks_to_indices.items(), key=lambda item: item[0])]


def run(fn, args=(), kwargs={}, num_proc=None, start_timeout=None,
        use_mpi=None, use_gloo=None, extra_mpi_args=None,
        env=None, stdout=None, stderr=None, verbose=1, nics=None,
        prefix_output_with_timestamp=False, executable=None):
    """
    Runs Horovod on Spark.  Runs `num_proc` processes executing `fn` using the same amount of Spark tasks.

    Args:
        fn: Function to run.
        args: Arguments to pass to `fn`.
        kwargs: Keyword arguments to pass to `fn`.
        num_proc: Number of Horovod processes.  Defaults to `spark.default.parallelism`.
        start_timeout: Timeout for Spark tasks to spawn, register and start running the code, in seconds.
                       If not set, falls back to `HOROVOD_SPARK_START_TIMEOUT` environment variable value.
                       If it is not set as well, defaults to 600 seconds.
        extra_mpi_args: Extra arguments for mpi_run. Defaults to no extra args.
        env: Environment dictionary to use in Horovod run.
        stdout: Horovod stdout is redirected to this stream. Defaults to sys.stdout when used with MPI.
        stderr: Horovod stderr is redirected to this stream. Defaults to sys.stderr when used with MPI.
        verbose: Debug output verbosity (0-2). Defaults to 1.
        nics: List of NICs for tcp network communication.
        prefix_output_with_timestamp: shows timestamp in stdout/stderr forwarding on the driver
        executable: Optional executable to run when launching the workers. Defaults to `sys.executable`.

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
                                     start_timeout=tmout,
                                     nics=nics,
                                     run_func_mode=True,
                                     prefix_output_with_timestamp=prefix_output_with_timestamp)

    spark_context = pyspark.SparkContext._active_spark_context
    if spark_context is None:
        raise Exception('Could not find an active SparkContext, are you '
                        'running in a PySpark session?')

    if num_proc is None:
        num_proc = spark_context.defaultParallelism
        if settings.verbose >= 1:
            logging.info('Running %d processes (inferred from spark.default.parallelism)...', num_proc)
    else:
        if settings.verbose >= 1:
            logging.info('Running %d processes...', num_proc)
    settings.num_proc = num_proc

    result_queue = queue.Queue(1)

    # start Spark driver service and launch settings.num_proc Spark tasks
    spark_job_group = 'horovod.spark.run.%d' % job_id.next_job_id()
    driver = driver_service.SparkDriverService(settings.num_proc, settings.num_proc,
                                               fn, args, kwargs,
                                               settings.key, settings.nics)
    gloo_is_used = is_gloo_used(use_gloo=use_gloo, use_mpi=use_mpi, use_jsrun=False)
    spark_thread = _make_spark_thread(spark_context, spark_job_group, driver,
                                      result_queue, settings,
                                      use_gloo=gloo_is_used, is_elastic=False)
    try:
        # wait for all tasks to register, notify them and initiate task-to-task address registration
        _notify_and_register_task_addresses(driver, settings)

        # Determine the index grouping based on host hashes.
        # Barrel shift until index 0 is in the first host.
        host_hashes = list(driver.task_host_hash_indices().keys())
        host_hashes.sort()
        while 0 not in driver.task_host_hash_indices()[host_hashes[0]]:
            host_hashes = host_hashes[1:] + host_hashes[:1]

        settings.hosts = ','.join('%s:%d' % (host_hash, len(driver.task_host_hash_indices()[host_hash]))
                                  for host_hash in host_hashes)

        # Run the job
        _launch_job(use_mpi, use_gloo, settings, driver, env, stdout, stderr, executable)
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

    # get ranks from driver
    indices_in_rank_order = _get_indices_in_rank_order(driver)

    # If there's no exception, execution results are in this queue.
    results = result_queue.get_nowait()
    return [results[index] for index in indices_in_rank_order]


def run_elastic(fn, args=(), kwargs={}, num_proc=None, min_np=None, max_np=None,
                start_timeout=None, elastic_timeout=None, reset_limit=None, env=None,
                stdout=None, stderr=None, verbose=1, nics=None,
                prefix_output_with_timestamp=False):
    """
    Runs Elastic Horovod on Spark.  Runs `num_proc` processes executing `fn` using the same amount of Spark tasks.

    Args:
        fn: Function to run.
        args: Arguments to pass to `fn`.
        kwargs: Keyword arguments to pass to `fn`.
        num_proc: Number of Horovod processes.  Defaults to `spark.default.parallelism`.
        start_timeout: Timeout for Spark tasks to spawn, register and start running the code, in seconds.
                       If not set, falls back to `HOROVOD_SPARK_START_TIMEOUT` environment variable value.
                       If it is not set as well, defaults to 600 seconds.
        elastic_timeout: Timeout for elastic initialisation after re-scaling the cluster.
                       If not set, falls back to `HOROVOD_ELASTIC_TIMEOUT` environment variable value.
                       If it is not set as well, defaults to 600 seconds.
        reset_limit: Maximum number of resets after which the job is terminated.
        env: Environment dictionary to use in Horovod run.  Defaults to `os.environ`.
        stdout: Horovod stdout is redirected to this stream.
        stderr: Horovod stderr is redirected to this stream.
        verbose: Debug output verbosity (0-2). Defaults to 1.
        nics: List of NICs for tcp network communication.
        prefix_output_with_timestamp: shows timestamp in stdout/stderr forwarding on the driver

    Returns:
        List of results returned by running `fn` on each rank.
    """

    if not gloo_built(verbose=(verbose >= 2)):
        raise ValueError('Gloo support is required to use elastic training, but has not been built.  Ensure CMake is '
                         'installed and reinstall Horovod with HOROVOD_WITH_GLOO=1 to debug the build error.')

    spark_context = pyspark.SparkContext._active_spark_context
    if spark_context is None:
        raise Exception('Could not find an active SparkContext, are you '
                        'running in a PySpark session?')

    if start_timeout is None:
        # Lookup default timeout from the environment variable.
        start_timeout = int(os.getenv('HOROVOD_SPARK_START_TIMEOUT', '600'))

    # nics needs to be a set
    if nics and not isinstance(nics, set):
        nics = set(nics)

    if num_proc is None:
        # TODO: #2023 try spark.dynamicAllocation.initialExecutors
        num_proc = spark_context.defaultParallelism
        if verbose >= 1:
            logging.info('Running %d processes (inferred from spark.default.parallelism)...', num_proc)
    else:
        if verbose >= 1:
            logging.info('Running %d processes...', num_proc)

    if min_np is None:
        # TODO: #2023 try spark.dynamicAllocation.minExecutors
        min_np = num_proc
    if max_np is None:
        # TODO: #2023 try spark.dynamicAllocation.maxExecutors
        max_np = num_proc

    # start Spark driver service and launch settings.num_proc Spark tasks
    key = secret.make_secret_key()
    spark_job_group = 'horovod.spark.run.%d' % job_id.next_job_id()
    driver = driver_service.SparkDriverService(num_proc, max_np,
                                               fn, args, kwargs,
                                               key, nics)

    discovery = host_discovery.SparkDriverHostDiscovery(driver)

    tmout = timeout.Timeout(start_timeout,
                            message='Timed out waiting for {activity}. Please check that you have '
                                    'enough resources to run all Horovod processes. Each Horovod '
                                    'process runs in a Spark task. You may need to increase the '
                                    'start_timeout parameter to a larger value if your Spark resources '
                                    'are allocated on-demand.')
    settings = hvd_elastic_settings.ElasticSettings(discovery=discovery,
                                                    min_np=min_np,
                                                    max_np=max_np,
                                                    elastic_timeout=elastic_timeout,
                                                    reset_limit=reset_limit,
                                                    num_proc=num_proc,
                                                    verbose=verbose,
                                                    key=key,
                                                    start_timeout=tmout,
                                                    nics=nics,
                                                    run_func_mode=True,
                                                    prefix_output_with_timestamp=prefix_output_with_timestamp)

    result_queue = queue.Queue(1)

    # launch settings.num_proc / settings.max_np Spark tasks
    spark_thread = _make_spark_thread(spark_context, spark_job_group, driver,
                                      result_queue, settings, use_gloo=True, is_elastic=True)
    try:
        # Register task addresses of initial num_proc tasks
        _register_task_addresses(driver, settings)

        # Run the job
        gloo_run_elastic(settings, driver, env, stdout, stderr)
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

    # get ranks from driver
    indices_in_rank_order = _get_indices_in_rank_order(driver)

    # If there's no exception, execution results are in this queue.
    results = result_queue.get_nowait()
    return [results[index] for index in indices_in_rank_order]
