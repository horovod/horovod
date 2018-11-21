# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

from horovod.spark import codec, host_hash, task_service, driver_service, network, timeout, safe_shell_exec, secret


def _task_fn(index, driver_addresses, num_proc, start_timeout_at, key):
    tmout = timeout.Timeout(start_timeout_at)
    task = task_service.TaskService(index, key)
    try:
        driver_client = driver_service.DriverClient(driver_addresses, key)
        driver_client.register_task(index, task.addresses(), host_hash.host_hash())
        task.wait_for_initial_registration(tmout)
        # Tasks ping each other in a circular fashion to determine interfaces reachable within
        # the cluster.
        next_task_index = (index + 1) % num_proc
        next_task_addresses = driver_client.all_task_addresses(next_task_index)
        next_task_client = task_service.TaskClient(next_task_index, next_task_addresses, key)
        driver_client.register_task_to_task_addresses(next_task_index, next_task_client.addresses())
        task_indices_on_this_host = driver_client.task_host_hash_indices(host_hash.host_hash())
        if task_indices_on_this_host[0] == index:
            # Task with first index will execute orted that will run mpirun_exec_fn for all tasks.
            task.wait_for_command_start(tmout)
            task.wait_for_command_termination()
        else:
            # The rest of tasks need to wait for the first task to finish.
            first_task_addresses = driver_client.all_task_addresses(task_indices_on_this_host[0])
            first_task_client = task_service.TaskClient(task_indices_on_this_host[0], first_task_addresses, key)
            first_task_client.wait_for_command_termination()
        return task.fn_result()
    except network.DrainError as e:
        # Drop traceback for Python 3 since it's not informative in this case.
        exc = Exception('Terminating due to an earlier error: %s' % str(e))
        exc.__cause__ = None
        raise exc
    finally:
        task.shutdown()


def _make_mapper(driver_addresses, num_proc, start_timeout_at, key):
    def _mapper(index, _):
        yield _task_fn(index, driver_addresses, num_proc, start_timeout_at, key)
    return _mapper


def _make_barrier_mapper(driver_addresses, num_proc, start_timeout_at, key):
    def _mapper(_):
        ctx = pyspark.BarrierTaskContext.get()
        ctx.barrier()
        index = ctx.partitionId()
        yield _task_fn(index, driver_addresses, num_proc, start_timeout_at, key)
    return _mapper


def _make_spark_thread(spark_context, num_proc, driver, start_timeout_at, key, result_queue):
    def run_spark():
        try:
            # We assume that folks caring about security will enable Spark RPC encryption,
            # thus ensuring that key that is passed here remains secret.
            procs = spark_context.range(0, numSlices=num_proc)
            if hasattr(procs, 'barrier'):
                # Use .barrier() functionality if it's available.
                procs = procs.barrier()
                result = procs.mapPartitions(
                    _make_barrier_mapper(driver.addresses(), num_proc, start_timeout_at, key)).collect()
            else:
                result = procs.mapPartitionsWithIndex(
                    _make_mapper(driver.addresses(), num_proc, start_timeout_at, key)).collect()
            result_queue.put(result)
        except:
            driver.notify_spark_job_failed()
            raise

    spark_thread = threading.Thread(target=run_spark)
    spark_thread.start()
    return spark_thread


def run(fn, args=(), kwargs={}, num_proc=None, start_timeout=None, env=None, verbose=1):
    """
    Runs Horovod in Spark.  Runs `num_proc` processes executing `fn` using the same amount of Spark tasks.

    Args:
        fn: Function to run.
        args: Arguments to pass to `fn`.
        kwargs: Keyword arguments to pass to `fn`.
        num_proc: Number of Horovod processes.  Defaults to `spark.default.parallelism`.
        start_timeout: Timeout for Spark tasks to spawn, register and start running the code, in seconds.
                       If not set, falls back to `HOROVOD_SPARK_START_TIMEOUT` environment variable value.
                       If it is not set as well, defaults to 300 seconds.
        env: Environment dictionary to use in training.  Defaults to `os.environ`.
        verbose: Output verbosity (0-2). Defaults to 1.

    Returns:
        List of results returned by running `fn` on each rank.
    """
    spark_context = pyspark.SparkContext._active_spark_context
    if spark_context is None:
        raise Exception('Could not find an active SparkContext, are you running in a PySpark session?')

    if num_proc is None:
        num_proc = spark_context.defaultParallelism
        if verbose >= 1:
            print('Running %d processes (inferred from spark.default.parallelism)...' % num_proc)
    else:
        if verbose >= 1:
            print('Running %d processes...' % num_proc)

    if start_timeout is None:
        # Lookup default timeout from the environment variable.
        start_timeout = int(os.getenv('HOROVOD_SPARK_START_TIMEOUT', '300'))

    result_queue = queue.Queue(1)
    start_timeout_at = timeout.timeout_at(start_timeout)
    tmout = timeout.Timeout(start_timeout_at)
    key = secret.make_secret_key()
    driver = driver_service.DriverService(num_proc, fn, args, kwargs, key)
    spark_thread = _make_spark_thread(spark_context, num_proc, driver, start_timeout_at, key, result_queue)
    try:
        driver.wait_for_initial_registration(tmout)
        if verbose >= 2:
            print('Initial Spark task registration is complete.')
        task_clients = [task_service.TaskClient(index, driver.task_addresses_for_driver(index), key)
                        for index in range(num_proc)]
        for task_client in task_clients:
            task_client.notify_initial_registration_complete()
        driver.wait_for_task_to_task_address_updates(tmout)
        if verbose >= 2:
            print('Spark task-to-task address registration is complete.')

        # Determine a set of common interfaces for task-to-task communication.
        common_intfs = set(driver.task_addresses_for_tasks(0).keys())
        for index in range(1, num_proc):
            common_intfs.intersection_update(driver.task_addresses_for_tasks(index).keys())
        if not common_intfs:
            raise Exception('Unable to find a set of common task-to-task communication interfaces: %s'
                            % [(index, driver.task_addresses_for_tasks(index)) for index in range(num_proc)])

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

        mpirun_command = (
            'mpirun --allow-run-as-root --tag-output '
            '-np {num_proc} -H {hosts} '
            '-bind-to none -map-by slot '
            '-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include {common_intfs} '
            '-x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME={common_intfs} '
            '{env} '  # expect a lot of environment variables
            '-mca plm_rsh_agent "{python} -m horovod.spark.mpirun_rsh {encoded_driver_addresses}" '
            '{python} -m horovod.spark.mpirun_exec_fn {encoded_driver_addresses} '
            .format(num_proc=num_proc,
                    hosts=','.join('%s:%d' % (host_hash, len(driver.task_host_hash_indices()[host_hash]))
                                   for host_hash in host_hashes),
                    common_intfs=','.join(common_intfs),
                    env=' '.join('-x %s' % key for key in os.environ.keys()),
                    python=sys.executable,
                    encoded_driver_addresses=codec.dumps_base64(driver.addresses())))
        if verbose >= 2:
            print('+ %s' % mpirun_command)
        exit_code = safe_shell_exec.execute(
            mpirun_command,
            env=os.environ if env is None else env)
        if exit_code != 0:
            raise Exception('mpirun exited with code %d, see the error above.' % exit_code)
    except Exception as e:
        try:
            # Naked raise re-raises last exception.  Since notifications below use exception-swallowing which could
            # corrupt the last exception, we immediately re-raise here to ensure the correct exception is raised
            # and use finally to execute the notification code before it happens.
            raise
        finally:
            # Schedule driver for shutdown, so tasks trying to connect due to Spark retries will fail fast.
            driver.drain(str(e))

            # Interrupt waiting tasks.  This is useful if the main flow quickly terminated, e.g. due to mpirun error,
            # and tasks are still waiting for a command to be executed on them.  This request is best-effort and is
            # not required for the proper shutdown, it just speeds it up and provides clear error message.
            for index in driver.registered_task_indices():
                # We only need to do this housekeeping while Spark Job is in progress.  If Spark job has finished,
                # it means that all the tasks are already terminated.
                if spark_thread.is_alive():
                    try:
                        task_client = task_service.TaskClient(index, driver.task_addresses_for_driver(index), key)
                        task_client.interrupt_waits(str(e))
                    except:
                        pass
    finally:
        spark_thread.join()
        driver.shutdown()

    # Make sure Spark Job did not fail.
    driver.check_for_spark_job_failure()

    # If there's no exception, execution results are in this queue.
    results = result_queue.get_nowait()
    return [results[index] for index in ranks_to_indices]
