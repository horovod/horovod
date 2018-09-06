import  horovod.torch as hvd
import torch
hvd.init()
rank = hvd.rank()
size = hvd.size()
tensor=torch.FloatTensor([4,5,6])
if rank==0:
    tensor = torch.FloatTensor([1,2,3])
else:
    root_tensor = torch.FloatTensor([4,5,6])

allreduce_tensor = hvd.allreduce(tensor,average=True, name="allreduceTenseor")
print(allreduce_tensor)

