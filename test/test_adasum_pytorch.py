import torch
import horovod.torch as hvd
import numpy as np
import time
from horovod.torch.mpi_ops import synchronize

hvd.init()

device = torch.device('cuda', hvd.local_rank())

np.random.seed(2)
torch.manual_seed(0)
size = hvd.size()
rank = hvd.rank()
num_tensors_per_rank = 3
all_Ns = [size*20 - 17, size*2+1, size+2]
tensors = []
all_qs = []
for N in all_Ns:
  a = np.random.normal(0, 1, (N,size)).astype(np.float32)
  q, r = np.linalg.qr(a)
  all_qs.append(q)
  tensors.append(q[:,hvd.rank()])

tensors = list(map(lambda x: torch.from_numpy(x).to(device), tensors))

handles = [
  hvd.allreduce_async(tensor, average=False)
  for tensor in tensors
]

reduced_tensors = [synchronize(h) for h in handles]

expected = [np.sum(q,axis=1)/4 for q in all_qs]
if rank >= 0:
  print(reduced_tensors[1])
  print(expected[1])
  print([(np.isclose(e, rt.cpu().numpy())) for e,rt in zip(expected,reduced_tensors)])
print([np.alltrue(np.isclose(e, rt.cpu().numpy())) for e,rt in zip(expected,reduced_tensors)])
