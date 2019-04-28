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
import pyspark
from six.moves import queue
import sys
import threading

from horovod.spark.task import task_service
from horovod.run.common.util import codec, env as env_util, safe_shell_exec, \
    timeout, host_hash, secret
from horovod.run.common.util import settings as hvd_settings
from horovod.spark.driver import driver_service, job_id


def _task_fn(index, driver_addresses, settings):
    task = task_service.SparkTaskService(index, settings.key)
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
        if task_indices_on_this_host[0] == index:
            # Task with first index will execute orted that will run mpirun_exec_fn for all tasks.
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


def _make_mapper(driver_addresses, settings):
    def _mapper(index, _):
        yield _task_fn(index, driver_addresses, settings)
    return _mapper


def _make_spark_thread(spark_context, spark_job_group, driver, result_queue,
                       settings):
    def run_spark():
        try:
            spark_context.setJobGroup(spark_job_group,
                                      "Horovod Spark Run",
                                      interruptOnCancel=True)
            procs = spark_context.range(0, numSlices=settings.num_proc)
            # We assume that folks caring about security will enable Spark RPC
            # encryption, thus ensuring that key that is passed here remains
            # secret.
            result = procs.mapPartitionsWithIndex(_make_mapper(driver.addresses(), settings)).collect()
            result_queue.put(result)
        except:
            driver.notify_spark_job_failed()
            raise

    spark_thread = threading.Thread(target=run_spark)
    spark_thread.start()
    return spark_thread


def run(fn, args=(), kwargs={}, num_proc=None, start_timeout=None, env=None,
        stdout=None, stderr=None, verbose=1):
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
        env: Environment dictionary to use in Horovod run.  Defaults to `os.environ`.
        stdout: Horovod stdout is redirected to this stream. Defaults to sys.stdout.
        stderr: Horovod stderr is redirected to this stream. Defaults to sys.stderr.
        verbose: Debug output verbosity (0-2). Defaults to 1.

    Returns:
        List of results returned by running `fn` on each rank.
    """

    if start_timeout is None:
        # Lookup default timeout from the environment variable.
        start_timeout = int(os.getenv('HOROVOD_SPARK_START_TIMEOUT', '600'))

    tmout = timeout.Timeout(start_timeout,
                            message='Timed out waiting for {activity}. Please check that you have '
                                    'enough resources to run all Horovod processes. Each Horovod '
                                    'process runs in a Spark task. You may need to increase the '
                                    'start_timeout parameter to a larger value if your Spark resources '
                                    'are allocated on-demand.')
    settings = hvd_settings.Settings(verbose=verbose,
                                     key=secret.make_secret_key(),
                                     timeout=tmout)

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

    spark_job_group = 'horovod.spark.run.%d' % job_id.next_job_id()
    driver = driver_service.SparkDriverService(settings.num_proc, fn, args, kwargs,
                                               settings.key)
    spark_thread = _make_spark_thread(spark_context, spark_job_group, driver,
                                      result_queue, settings)
    try:
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

        # Determine a set of common interfaces for task-to-task communication.
        common_intfs = set(driver.task_addresses_for_tasks(0).keys())
        for index in range(1, settings.num_proc):
            common_intfs.intersection_update(driver.task_addresses_for_tasks(index).keys())
        if not common_intfs:
            raise Exception('Unable to find a set of common task-to-task communication interfaces: %s'
                            % [(index, driver.task_addresses_for_tasks(index)) for index in range(settings.num_proc)])

        # Determine the index grouping based on host hashes.
        # Barrel shift until index 0 is in the first host.
        host_hashes = list(driver.task_host_hash_indices().keys())
        host_hashes.sort()
        while 0 not in driver.task_host_hash_indices()[host_hashes[0]]:
            host_hashes = host_hashes[1:] + host_hashes[:1]

        ranks_to_indices = []
        for host_hash in host_hashes:
            ranks_to_indices += driver.task_host_hash_indices()[host_hash]
        driver.set_ranks_to_indices(ranks_to_indices)

        if env is None:
            env = os.environ.copy()

        # Pass secret key through the environment variables.
        env[secret.HOROVOD_SECRET_KEY] = codec.dumps_base64(settings.key)

        mpirun_command = (
            'mpirun --allow-run-as-root --tag-output '
            '-np {num_proc} -H {hosts} '
            '-bind-to none -map-by slot '
            '-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include {common_intfs} '
            '-x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME={common_intfs} '
            '{env} '  # expect a lot of environment variables
            '-mca plm_rsh_agent "{python} -m horovod.spark.driver.mpirun_rsh {encoded_driver_addresses} {settings}" '
            '{python} -m horovod.spark.task.mpirun_exec_fn {encoded_driver_addresses} {settings}'
                .format(num_proc=settings.num_proc,
                        hosts=','.join('%s:%d' % (host_hash, len(driver.task_host_hash_indices()[host_hash]))
                                       for host_hash in host_hashes),
                        common_intfs=','.join(common_intfs),
                        env=' '.join('-x %s' % key for key in env.keys() if env_util.is_exportable(key)),
                        python=sys.executable,
                        encoded_driver_addresses=codec.dumps_base64(driver.addresses()),
                        settings=codec.dumps_base64(settings)))
        if settings.verbose >= 2:
            print('+ %s' % mpirun_command)
        exit_code = safe_shell_exec.execute(mpirun_command, env, stdout, stderr)
        if exit_code != 0:
            raise Exception('mpirun exited with code %d, see the error above.' % exit_code)
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
