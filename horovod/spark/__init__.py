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

import pyspark
import os
import sys
import threading

from horovod.spark import codec, host_hash, task_service, driver_service, timeout, safe_shell_exec


def _make_mapper(driver_addresses, num_proc, start_timeout):
    def _mapper(index, _):
        tmout = timeout.Timeout(start_timeout)
        task = task_service.TaskService(index)
        try:
            driver_client = driver_service.DriverClient(driver_addresses)
            driver_client.register(index, task.addresses(), host_hash.host_hash())
            task.wait_for_initial_registration(tmout)
            # Tasks ping each other in a circular fashion to determine interfaces reachable within
            # the cluster.
            next_task_index = (index + 1) % num_proc
            next_task_addresses = driver_client.all_task_addresses(next_task_index)
            next_task_client = task_service.TaskClient(next_task_index, next_task_addresses)
            driver_client.register_task_to_task_addresses(next_task_index, next_task_client.addresses())
            driver_client.wait_for_termination()
            yield
        finally:
            task.shutdown()
    return _mapper


def _make_spark_thread(spark_context, num_proc, driver, start_timeout):
    def run_spark():
        # TODO: what is a future way to run GPU workers?
        spark_context \
            .range(0, numSlices=num_proc) \
            .mapPartitionsWithIndex(_make_mapper(driver.addresses(), num_proc, start_timeout)) \
            .collect()

    spark_thread = threading.Thread(target=run_spark)
    spark_thread.start()
    return spark_thread


def _autodetect_spark_context():
    if ('__main__' in sys.modules and hasattr(sys.modules['__main__'], 'spark')
            and isinstance(sys.modules['__main__'].spark, pyspark.sql.session.SparkSession)):
        return sys.modules['__main__'].spark.sparkContext

    if ('pyspark.shell' in sys.modules and hasattr(sys.modules['pyspark.shell'], 'spark')
            and isinstance(sys.modules['pyspark.shell'].spark, pyspark.sql.session.SparkSession)):
        return sys.modules['pyspark.shell'].spark.sparkContext

    raise Exception('Could not find SparkContext, please set it via spark_context= parameter.')


def train(fn, num_proc=None, spark_context=None, start_timeout=180):
    if spark_context is None:
        spark_context = _autodetect_spark_context()
    if num_proc is None:
        num_proc = spark_context.defaultParallelism

    tmout = timeout.Timeout(start_timeout)
    driver = driver_service.DriverService(num_proc, fn)
    spark_thread = _make_spark_thread(spark_context, num_proc, driver, start_timeout)
    try:
        driver.wait_for_initial_registration(tmout)
        task_clients = [task_service.TaskClient(index, driver.task_addresses_for_driver(index))
                        for index in range(num_proc)]
        for task_client in task_clients:
            task_client.notify_initial_registration_complete()
        driver.wait_for_task_to_task_address_updates(tmout)

        # Determine a set of common interfaces for task-to-task communication.
        common_intfs = set(driver.task_addresses_for_tasks(0).keys())
        for index in range(1, num_proc):
            common_intfs.intersection_update(driver.task_addresses_for_tasks(index).keys())
        if not common_intfs:
            raise Exception('Unable to find a set of common task-to-task communication interfaces: %s'
                            % [(index, driver.task_addresses_for_tasks(index)) for index in range(num_proc)])

        # Determine the index grouping based on host hashes.
        # Barrel shift until index 0 is in the first host.
        host_hashes = driver.task_host_hash_indices().keys()
        host_hashes.sort()
        while 0 not in driver.task_host_hash_indices()[host_hashes[0]]:
            host_hashes = host_hashes[1:] + host_hashes[:1]

        exit_code = safe_shell_exec.execute(
            'mpirun --allow-run-as-root '
            '-np {num_proc} -H {hosts} '
            '-bind-to none -map-by slot '
            '-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include {common_intfs} '
            '-x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME={common_intfs} '
            '{env} '  # expect a lot of environment variables
            '-mca plm_rsh_agent "python -m horovod.spark.mpirun_rsh {encoded_driver_addresses}" '
            'python -m horovod.spark.mpirun_exec_fn {encoded_driver_addresses} '
            .format(num_proc=num_proc,
                    hosts=','.join('%s:%d' % (host_hash, len(driver.task_host_hash_indices()[host_hash]))
                                   for host_hash in host_hashes),
                    common_intfs=','.join(common_intfs),
                    env=' '.join('-x %s' % key for key in os.environ.keys()),
                    encoded_driver_addresses=codec.dumps_base64(driver.addresses())))
        if exit_code != 0:
            raise Exception('mpirun exited with code %d, see the error above.' % exit_code)
    finally:
        driver.shutdown()
        spark_thread.join()
