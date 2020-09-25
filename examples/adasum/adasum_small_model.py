import torch
import random
import horovod.torch as hvd
import numpy as np
import argparse
import os

def sq(x):
    m2 = 1.
    m1 = -20.
    m0 = 50.
    return m2*x*x + m1*x + m0

def qu(x):
    m3 = 10.
    m2 = 5.
    m1 = -20.
    m0 = -5.
    return m3*x*x*x + m2*x*x + m1*x + m0

class Net(torch.nn.Module):    
    def __init__(self, mode = "sq"):
        super(Net, self).__init__()
        
        if mode == "square":
            self.mode = 0
            self.param = torch.nn.Parameter(torch.FloatTensor([1., -1.]))
        else:
            self.mode = 1
            self.param = torch.nn.Parameter(torch.FloatTensor([1., -1., 1.]))

    def forward(self, x):
        if ~self.mode:
            return x*x + self.param[0]*x + self.param[1]
        else:
            return 10*x*x*x + self.param[0]*x*x + self.param[1]*x + self.param[2]

def train(args):
    hvd.init()

    net = Net(args.mode)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
    )

    num_steps = 50

    np.random.seed(1 + hvd.rank())
    torch.manual_seed(1234)

    prev_zero = False

    for step in range(1, num_steps + 1):
        features = torch.Tensor(np.random.rand(1) * 2 * args.x_max - args.x_max)
        if args.mode == "square": 
            labels = sq(features)  
        else: 
            labels = qu(features)
        optimizer.zero_grad()
        outputs = net(features)
        loss = torch.nn.MSELoss()(outputs, labels)
        loss.backward()

        if args.op == "Average":
            net.param.grad.data = hvd.allreduce(net.param.grad.data, op=hvd.Average)
        elif args.op == "Adasum":
            net.param.grad.data = hvd.allreduce(net.param.grad.data, op=hvd.Adasum)

        optimizer.step()

        #Uncomment below lines to see how loss and gradients change with Adasum
        #if hvd.rank() == 0:
        #    print(step, loss.item(), net.param.grad.data[0].item(), net.param.grad.data[1].item())

        if net.param.grad.data[0].item() == 0 and net.param.grad.data[1].item() == 0:
            if prev_zero:
                break
            else:
                prev_zero = True
        else:
            prev_zero = False


    if step == num_steps:
        step = 100

    if hvd.rank() == 0:
            print(args.x_max, args.op, args.learning_rate, hvd.size(), step)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="square", choices=["square", "cubic"])
    parser.add_argument('--op', type=str, default="Average", choices=["Average", "Adasum"], dest='op')
    parser.add_argument('--learning_rate', type=float, default=0.1, dest='learning_rate')
    parser.add_argument('--x_max', type=float, default=1., dest='x_max')
    args = parser.parse_args()

    train(args)