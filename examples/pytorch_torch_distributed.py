# Copyright 2026 The Horovod Authors. All Rights Reserved.
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
# =============================================================================
"""
Launch PyTorch DDP training with `horovodrun`; communicate via `torch.distributed`.

This build of Horovod is a *launcher-only* package: `horovodrun` spawns one process
per slot and sets HOROVOD_* environment variables, but all distributed communication
is done by PyTorch's `torch.distributed`. The TensorFlow/PyTorch/MXNet Horovod
integrations (and the C++ allreduce core) have been removed.

Single node, one process per GPU (Gloo launcher):
    horovodrun -np 4 --gloo python examples/pytorch_torch_distributed.py

Single node via external mpirun:
    horovodrun -np 4 --mpi python examples/pytorch_torch_distributed.py

The mapping from the HOROVOD_* variables set by the launcher to the
RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT variables expected by
`torch.distributed.init_process_group(init_method="env://")` is done in
`init_torch_distributed()` below.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def _read_env(*names):
    """Return the first environment variable in `names` that is set, as an int."""
    for name in names:
        if name in os.environ:
            return int(os.environ[name])
    raise RuntimeError('None of {} found in the environment; are you launching with '
                       'horovodrun?'.format(list(names)))


def init_torch_distributed():
    """Map the env vars set by the horovodrun launcher to the env vars
    `torch.distributed` expects, then initialize the process group.

    The Gloo launcher sets HOROVOD_*; `--mpi` (external mpirun) sets
    OMPI_COMM_WORLD_*. Both are handled here."""
    rank = _read_env('HOROVOD_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK')
    world_size = _read_env('HOROVOD_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE')
    local_rank = _read_env('HOROVOD_LOCAL_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK')

    # torch.distributed env:// init needs MASTER_ADDR / MASTER_PORT.
    #  - Gloo launcher: reuse the rendezvous address (the driver IP, reachable by
    #    every rank). Pick a port DISTINCT from HOROVOD_GLOO_RENDEZVOUS_PORT, which
    #    the launcher already binds.
    #  - mpirun: there is no rendezvous var, so default to localhost for single
    #    node. For multi-node MPI, set MASTER_ADDR / MASTER_PORT yourself.
    # Override MASTER_PORT in the environment if the default is not free.
    master_addr = (os.environ.get('MASTER_ADDR')
                   or os.environ.get('HOROVOD_GLOO_RENDEZVOUS_ADDR')
                   or '127.0.0.1')
    os.environ['MASTER_ADDR'] = master_addr
    os.environ.setdefault('MASTER_PORT', '29500')

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, init_method='env://',
                            rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def main():
    rank, world_size, local_rank = init_torch_distributed()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(local_rank) if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.set_device(local_rank)

    # A tiny model + synthetic dataset so the example runs without downloading data.
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
    model = DDP(model, device_ids=[local_rank] if use_cuda else None)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for step in range(10):
        data = torch.randn(64, 32, device=device)
        target = torch.randn(64, 10, device=device)
        optimizer.zero_grad()
        loss = loss_fn(model(data), target)
        loss.backward()          # DDP allreduces gradients across ranks here
        optimizer.step()
        if rank == 0:
            print('step {} loss {:.4f}'.format(step, loss.item()), flush=True)

    dist.barrier()
    if rank == 0:
        print('finished training on {} ranks'.format(world_size), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
