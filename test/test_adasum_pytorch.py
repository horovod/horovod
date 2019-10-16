# Copyright 2019 Microsoft. All Rights Reserved.
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

import torch
import horovod.torch as hvd
import numpy as np
import time
from horovod.torch.mpi_ops import synchronize
import os

device = None
size = 0
local_size = 0
rank = 0

data_type = None

def initialize(dtype=np.float32):
  hvd.init()
  global device
  global size
  global local_size
  global rank
  global data_type
  device = torch.device('cuda:{}'.format(hvd.local_rank()))
  torch.cuda.set_device(device)
  np.random.seed(2)
  torch.manual_seed(2)
  size = hvd.size()
  local_size = hvd.local_size()
  rank = hvd.rank()

  data_type = dtype
  
def test_orthogonal(denominator):
  all_Ns = [size*20 - 17, size*2+1, size+2]
  tensors = []
  all_qs = []
  for N in all_Ns:
    a = np.random.normal(0, 1, (N,size)).astype(np.float64)
    q, r = np.linalg.qr(a)
    q = q.astype(data_type)
    all_qs.append(q)
    tensors.append(q[:,hvd.rank()])

  tensors = list(map(lambda x: torch.from_numpy(x).to(device), tensors))

  handles = [
    hvd.allreduce_async(tensor, op=hvd.Adasum)
    for tensor in tensors
  ]

  reduced_tensors = [synchronize(h) for h in handles]

  expected = [np.sum(q,axis=1) / denominator for q in all_qs]
  all_comp = [np.alltrue(np.isclose(e, rt.cpu().numpy())) for e,rt in zip(expected,reduced_tensors)]
  if np.alltrue(all_comp):
    print('Orthogonal test passed')
  else:
    for c,e,rt in zip(all_comp, expected, reduced_tensors):
      if c == False:
        print('computed: ', rt)
        print('expected: ', e)
        print(np.isclose(e, rt.cpu().numpy()))
  
def test_parallel():
  all_Ns = [size*20 - 13, size*2+1, size+2]
  tensors = []
  all_qs = []
  for N in all_Ns:
    a = np.random.normal(0, 1, (N, 1)).astype(np.float64)
    r = np.random.normal(0, 1, (size, 1)).astype(np.float64)
    q = np.dot(a,r.T)
    q = q.astype(data_type)
    all_qs.append(q)
    tensors.append(q[:,hvd.rank()])

  tensors = list(map(lambda x: torch.from_numpy(x).to(device), tensors))

  handles = [
    hvd.allreduce_async(tensor, op=hvd.Adasum)
    for tensor in tensors
  ]

  reduced_tensors = [synchronize(h) for h in handles]

  expected = [np.sum(q,axis=1) / size for q in all_qs]
  all_comp = [np.alltrue(np.isclose(e, rt.cpu().numpy())) for e,rt in zip(expected,reduced_tensors)]
  if np.alltrue(all_comp):
    print('Parallel test passed')
  else:
    for c,e,rt in zip(all_comp, expected, reduced_tensors):
      if c == False:
        print('computed: ', rt)
        print('expected: ', e)
        print(np.isclose(e, rt.cpu().numpy()))
  
if __name__ == '__main__':
  initialize()
  test_orthogonal(local_size)
  test_parallel()
