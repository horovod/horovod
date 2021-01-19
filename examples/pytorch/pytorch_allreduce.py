import torch
import horovod.torch as hvd

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name = name)
    return avg_tensor.item()

if __name__ == "__main__":
    hvd.init()
    tensor = torch.randn(1)
    print("process is {0} and tensor is {1}".format(hvd.rank(), tensor))



