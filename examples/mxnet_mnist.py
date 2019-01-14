# Step 1: import required packages
import os
import argparse
import logging
import mxnet as mx
import horovod.mxnet as hvd
from mxnet.test_utils import get_mnist_ubyte

# Step 1: Initialize horovod
hvd.init()

# Training settings
parser = argparse.ArgumentParser(description='MXNet MNIST Example')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='training data type (default: float32)')
parser.add_argument('--gpus', type=str, default='0',
                    help='number of gpus to use (default: 0)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of training epochs (default: 10)')
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate (default: 0.05)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(args)

# Horovod: pin GPU to local rank.
context = mx.cpu() if args.gpus is None or args.gpus == '0' \
                   else mx.gpu(hvd.local_rank())

# Step 2: data loading
# download mnist data set
get_mnist_ubyte()
mx.random.seed(42)
batch_size = args.batch_size
input_shape = (1, 28, 28)

train_iter = mx.io.MNISTIter(
    image="data/train-images-idx3-ubyte",
    label="data/train-labels-idx1-ubyte",
    input_shape=input_shape,
    batch_size=batch_size,
    shuffle=True,
    flat=False,
    num_parts=hvd.size(),
    part_index=hvd.rank()
)

val_iter = mx.io.MNISTIter(
    image="data/t10k-images-idx3-ubyte",
    label="data/t10k-labels-idx1-ubyte",
    input_shape=input_shape,
    batch_size=batch_size,
    flat=False,
    num_parts=hvd.size(),
    part_index=hvd.rank()
)


# Step 3: define network
def mlp():
    # placeholder for data
    data = mx.sym.var('data')
    # Flatten the data from 4-D shape into 2-D 
    # (batch_size, num_channel*width*height)
    data = mx.sym.flatten(data=data)

    # The first fully-connected layer and the corresponding activation function
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")

    # The second fully-connected layer and the corresponding activation
    # function
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, act_type="relu")

    # MNIST has 10 classes
    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)

    return fc3


# Step 4: fit the model
net = mlp()
# Softmax with cross entropy loss
loss = mx.sym.SoftmaxOutput(data=net, name='softmax')
mlp_model = mx.mod.Module(symbol=loss, context=context)
optimizer_params = {'learning_rate': args.lr * hvd.size(),
                    'rescale_grad': 1.0 / batch_size}
opt = mx.optimizer.create('sgd', sym=net, **optimizer_params)

# Horovod: Wrap optimizer with DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                             magnitude=2)
mlp_model.bind(data_shapes=train_iter.provide_data,
               label_shapes=train_iter.provide_label)
mlp_model.init_params(initializer)

# Horovod: fetch and broadcast parameters
(arg_params, aux_params) = mlp_model.get_params()
if arg_params is not None:
    hvd.broadcast_parameters(arg_params, root_rank=0)
if aux_params is not None:
    hvd.broadcast_parameters(aux_params, root_rank=0)
mlp_model.set_params(arg_params=arg_params, aux_params=aux_params)

mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer=opt,  # use SGD to train
              eval_metric='acc',  # report accuracy during training
              batch_end_callback=mx.callback.Speedometer(batch_size),
              num_epoch=args.epochs)  # train for at most 10 dataset passes

# Step 5: Evaluate model accuracy
acc = mx.metric.Accuracy()
mlp_model.score(val_iter, acc)

if hvd.rank() == 0:
    print(acc)
    assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than \
                                expected (0.96)" % acc.get()[1]
