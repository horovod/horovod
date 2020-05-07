# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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

SPARK_CONF_MAX_INT = '2147483647'
SPARK_CONF_MAX_INT_MINUS_ONE = '2147483646'

# required for elastic fault-tolerance and auto-scale
# Horovod has retry counters and limits, no need to limit Spark's retries
SPARK_CONF_ALWAYS_RESTART_FAILED_TASK = ('spark.task.maxFailures', SPARK_CONF_MAX_INT)

# executor and node retry counters (Spark blacklist)
# see https://spark.apache.org/docs/latest/configuration.html#scheduling
SPARK_CONF_BLACKLIST_DISABLED = ('spark.blacklist.enabled', 'false')
SPARK_CONF_BLACKLIST_ENABLED = ('spark.blacklist.enabled', 'true')

# executors where any task fails due to exceptions can potentially be reused for any other task,
# including the task itself, unless SPARK_CONF_DONT_REUSE_EXECUTOR_FOR_SAME_TASK is set
# executors lost e.g. due to node failure can't be reused in any way
# requires SPARK_CONF_BLACKLIST_ENABLED
SPARK_CONF_REUSE_FAILED_EXECUTOR = ('spark.blacklist.stage.maxFailedTasksPerExecutor', SPARK_CONF_MAX_INT)
SPARK_CONF_DONT_REUSE_FAILED_EXECUTOR = ('spark.blacklist.stage.maxFailedTasksPerExecutor', '1')

# nodes with failed executors may have other executors that can still be used
# requires SPARK_CONF_BLACKLIST_ENABLED
# SPARK_CONF_REUSE_FAILING_NODE requires SPARK_CONF_REUSE_FAILED_EXECUTOR
SPARK_CONF_REUSE_FAILING_NODE = ('spark.blacklist.stage.maxFailedExecutorsPerNode', SPARK_CONF_MAX_INT_MINUS_ONE)
SPARK_CONF_DONT_REUSE_FAILING_NODE = ('spark.blacklist.stage.maxFailedExecutorsPerNode', '1')

# executors where tasks fail due to exceptions can potentially be reused for the same task
# executors lost e.g. due to node failure can't be reused in any way
# requires SPARK_CONF_BLACKLIST_ENABLED
SPARK_CONF_REUSE_EXECUTOR_ALWAYS_FOR_SAME_TASK = ('spark.blacklist.task.maxTaskAttemptsPerExecutor', SPARK_CONF_MAX_INT)
SPARK_CONF_REUSE_EXECUTOR_ONCE_FOR_SAME_TASK = ('spark.blacklist.task.maxTaskAttemptsPerExecutor', '2')
SPARK_CONF_DONT_REUSE_EXECUTOR_FOR_SAME_TASK = ('spark.blacklist.task.maxTaskAttemptsPerExecutor', '1')

# nodes (with multiple executors) where tasks fail due to exceptions can potentially be reused for the same task
# nodes lost e.g. due to node failure can't be reused in any way
# requires SPARK_CONF_BLACKLIST_ENABLED
# ? SPARK_CONF_REUSE_NODE_ALWAYS_FOR_SAME_TASK requires SPARK_CONF_REUSE_EXECUTOR_ALWAYS_FOR_SAME_TASK
SPARK_CONF_REUSE_NODE_ALWAYS_FOR_SAME_TASK = ('spark.blacklist.task.maxTaskAttemptsPerNode', SPARK_CONF_MAX_INT_MINUS_ONE)
SPARK_CONF_REUSE_NODE_ONCE_FOR_SAME_TASK = ('spark.blacklist.task.maxTaskAttemptsPerNode', '2')
SPARK_CONF_DONT_REUSE_NODE_FOR_SAME_TASK = ('spark.blacklist.task.maxTaskAttemptsPerNode', '1')

# executors where any task fails due to exceptions can potentially be reused for any other task,
# including the task itself, unless SPARK_CONF_DONT_REUSE_EXECUTOR_FOR_SAME_TASK is set
# executors lost e.g. due to node failure can't be reused in any way
# NOTE: in dynamic allocation, only executors blacklisted for the entire app
#       can get reclaimed by the cluster manager
# requires SPARK_CONF_BLACKLIST_ENABLED
SPARK_CONF_REUSE_FAILED_EXECUTOR_IN_APP = ('spark.blacklist.application.maxFailedTasksPerExecutor', SPARK_CONF_MAX_INT)
SPARK_CONF_DONT_REUSE_FAILED_EXECUTOR_IN_APP = ('spark.blacklist.application.maxFailedTasksPerExecutor', '1')

# nodes with failed executors may have other executors that can still be used across the app
# NOTE: in dynamic allocation, only executors blacklisted for the entire app
#       can get reclaimed by the cluster manager
# requires SPARK_CONF_BLACKLIST_ENABLED
SPARK_CONF_REUSE_FAILING_NODE_IN_APP = ('spark.blacklist.application.maxFailedExecutorsPerNode', SPARK_CONF_MAX_INT)
SPARK_CONF_DONT_REUSE_FAILING_NODE_IN_APP = ('spark.blacklist.application.maxFailedExecutorsPerNode', '1')

# default values: https://spark.apache.org/docs/latest/configuration.html
SPARK_CONF_DEFAULT_VALUES = {
                                'spark.task.maxFailures': '4',
                                'spark.blacklist.enabled': 'false',
                                'spark.blacklist.stage.maxFailedTasksPerExecutor': '2',
                                'spark.blacklist.stage.maxFailedExecutorsPerNode': '2',
                                'spark.blacklist.task.maxTaskAttemptsPerExecutor': '1',
                                'spark.blacklist.task.maxTaskAttemptsPerNode': '2',
                                'spark.blacklist.application.maxFailedTasksPerExecutor': '2',
                                'spark.blacklist.application.maxFailedExecutorsPerNode': '2'
}
