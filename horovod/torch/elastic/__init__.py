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

from horovod.common.elastic import run_fn
from horovod.torch.mpi_ops import init, shutdown

from horovod.torch.elastic.sampler import ElasticSampler
from horovod.torch.elastic.state import TorchState


def run(func):
    """Decorator used to run the elastic training process.

    The purpose of this decorator is to allow for uninterrupted execution of the wrapped function
    across multiple workers in parallel, as workers come and go from the system. When a new worker is added,
    its state needs to be brought to the same point as the other workers, which is done by synchronizing
    the state object before executing `func`.

    When a worker is added or removed, other workers will raise an exception to bring them back to such a sync
    point before executing `func` again. This ensures that workers do not diverge when such reset events occur.

    It's important to note that collective operations (e.g., broadcast, allreduce) cannot be the call to
    the wrapped function. Otherwise, new workers could execute these operations during their initialization
    while other workers are attempting to sync state, resulting in deadlock.

    Args:
        func: a wrapped function taking any number of args or kwargs. The first argument
              must be a `horovod.common.elastic.State` object used to synchronize state across
              workers.
    """
    return run_fn(func, _reset)


def _reset():
    shutdown()
    init()


__all__ = [
    'TorchState',
    'ElasticSampler',
    'run',
]
