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
import math

device = None
size = 0
local_size = 0
rank = 0

data_type = None

def initialize(dtype=np.float16):
  hvd.init()
  global device
  global size
  global local_size
  global rank
  global data_type
  device = torch.device('cuda:{}'.format(hvd.local_rank()))
  np.random.seed(2)
  torch.manual_seed(2)
  size = hvd.size()
  local_size = hvd.local_size()
  rank = hvd.rank()

  data_type = dtype

def diff_ratio(true_vec, comp_vec):
  norm_diff = np.linalg.norm(true_vec-comp_vec)
  norm_true = np.linalg.norm(true_vec)
  return norm_diff/norm_true/100.

def are_close(true_vec, comp_vec):
  return diff_ratio(true_vec, comp_vec) < np.finfo(data_type).eps
  
def test_orthogonal(denominator):
  all_Ns = [size*20 - 17, size*2+1, size+2, 2**19]
  tensors = []
  all_qs = []
  for N in all_Ns:
    a = np.random.normal(0, 1, (N,size)).astype(np.float64)
    q, r = np.linalg.qr(a)
    q = q.astype(data_type)
    all_qs.append(q.astype(np.float64))
    tensors.append(q[:,hvd.rank()])

  tensors = list(map(lambda x: torch.from_numpy(x).to(device), tensors))

  handles = [
    hvd.allreduce_async(tensor, op=hvd.Adasum)
    for tensor in tensors
  ]

  reduced_tensors = [synchronize(h) for h in handles]

  expected = [np.sum(q,axis=1) / denominator for q in all_qs]
  all_comp = [are_close(e, rt.cpu().numpy()) for e,rt in zip(expected,reduced_tensors)]
  if np.alltrue(all_comp):
    print('Orthogonal test passed')
  else:
    for c,e,rt in zip(all_comp, expected, reduced_tensors):
      if c == False:
        print('computed: ', rt)
        print('expected: ', e)
        print('off by: ', diff_ratio(e,rt.cpu().numpy()))
  
  
def test_parallel():
  all_Ns = [size*20 - 13, size*2+1, size+2, 2**19]
  tensors = []
  all_qs = []
  for N in all_Ns:
    a = np.random.normal(0, 1, (N, 1)).astype(np.float64)
    r = np.random.normal(0, 1, (size, 1)).astype(np.float64)
    q = np.dot(a,r.T)
    q = q.astype(data_type)
    all_qs.append(q.astype(np.float64))
    tensors.append(q[:,hvd.rank()])

  tensors = list(map(lambda x: torch.from_numpy(x).to(device), tensors))

  handles = [
    hvd.allreduce_async(tensor, op=hvd.Adasum)
    for tensor in tensors
  ]

  reduced_tensors = [synchronize(h) for h in handles]

  expected = [np.sum(q,axis=1) / size for q in all_qs]
  all_comp = [are_close(e, rt.cpu().numpy()) for e,rt in zip(expected,reduced_tensors)]
  if np.alltrue(all_comp):
    print('Parallel test passed')
  else:
    for c,e,rt in zip(all_comp, expected, reduced_tensors):
      if c == False:
        print('computed: ', rt)
        print('expected: ', e)
        print('off by: ', diff_ratio(e,rt.cpu().numpy()))
 
def test_stability():
  N = 1024
  a = np.random.normal(0, np.finfo(data_type).tiny, (N, 1)).astype(np.float64)
  r = np.random.normal(0, 1, (size, 1)).astype(np.float64)
  q = np.dot(a,r.T).astype(data_type).astype(np.float64)
  tensor = np.zeros(N,dtype=data_type)
  tensor[:] = q[:,hvd.rank()]

  tensor = torch.from_numpy(tensor).to(device)

  hvd.allreduce_(tensor, op=hvd.Adasum)

  expected = np.sum(q,axis=1) / size
  comp = are_close(expected, tensor.cpu().numpy()) 
  if comp:
    print('Stability test passed')
  else:
    print('computed: ', tensor)
    print('expected: ', expected)
    print('off by: ', diff_ratio(expected,tensor.cpu().numpy()))

def test_stability_2():
  N = 1024
  dt_min = np.finfo(data_type).tiny.astype(np.float64)
  dt_max = math.sqrt(np.finfo(data_type).max.astype(np.float64))
  a = np.random.normal(0, 1, (N, 1)).astype(np.float64)
  r = np.array([dt_max**(float(i+1)/float(size))*dt_min**(float(size-i-1)/float(size)) for i in range(size)]).reshape(size,1).astype(np.float64)
  np.random.shuffle(r)
  q = np.dot(a,r.T).astype(data_type).astype(np.float64)
  tensor = np.zeros(N,dtype=data_type)
  tensor[:] = q[:,hvd.rank()]

  tensor = torch.from_numpy(tensor).to(device)

  hvd.allreduce_(tensor, op=hvd.Adasum)

  expected = np.sum(q,axis=1) / size
  comp = are_close(expected, tensor.cpu().numpy()) 
  if comp:
    print('Stability 2 test passed')
  else:
    print('computed: ', tensor)
    print('expected: ', expected)
    print('off by: ', diff_ratio(expected,tensor.cpu().numpy()))

 
if __name__ == '__main__':
  initialize()
  test_orthogonal(local_size)
  test_parallel()
  test_stability()
  test_stability_2()
