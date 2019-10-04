import torch
import horovod.torch as hvd
import numpy as np
import time
from horovod.torch.mpi_ops import synchronize
from horovod.torch.mpi_ops import AllreduceType
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
  device = torch.device('cuda', hvd.local_rank())
  np.random.seed(2)
  torch.manual_seed(2)
  size = hvd.size()
  local_size = hvd.local_size()
  rank = hvd.rank()

  data_type = dtype
  
def test_orthogonal(do_average, denominator):
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
    hvd.allreduce_async(tensor, average=do_average, allreduce_type=AllreduceType.Adasum)
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
  
def test_parallel(do_average):
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
    hvd.allreduce_async(tensor, average=do_average, allreduce_type=AllreduceType.Adasum)
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
  if 'HOROVOD_ADASUM' not in os.environ:
    raise ValueError('Please set HOROVOD_ADASUM env variable.')
  initialize()
  do_average = False
  denominator = 1
  if os.environ['HOROVOD_ADASUM'] == 'GPU_NCCL_LOCAL_AVG':
    do_average = True
    denominator = local_size
  test_orthogonal(do_average, denominator)
  test_parallel(do_average)
