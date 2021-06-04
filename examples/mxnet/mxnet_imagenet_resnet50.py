# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import logging
import math
import os
import time

from gluoncv.model_zoo import get_model
import horovod.mxnet as hvd
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, lr_scheduler
from mxnet.io import DataBatch, DataIter


# Training settings
parser = argparse.ArgumentParser(description='MXNet ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--use-rec', action='store_true', default=False,
                    help='use image record iter for data input (default: False)')
parser.add_argument('--data-nthreads', type=int, default=2,
                    help='number of threads for data decoding (default: 2)')
parser.add_argument('--rec-train', type=str, default='',
                    help='the training data')
parser.add_argument('--rec-train-idx', type=str, default='',
                    help='the index of training data')
parser.add_argument('--rec-val', type=str, default='',
                    help='the validation data')
parser.add_argument('--rec-val-idx', type=str, default='',
                    help='the index of validation data')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (default: 128)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training (default: float32)')
parser.add_argument('--num-epochs', type=int, default=90,
                    help='number of training epochs (default: 90)')
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate for a single GPU (default: 0.05)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer (default: 0.9)')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate (default: 0.0001)')
parser.add_argument('--lr-mode', type=str, default='poly',
                    help='learning rate scheduler mode. Options are step, \
                    poly and cosine (default: poly)')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate (default: 0.1)')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays (default: 40,60)')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate (default: 0.0)')
parser.add_argument('--warmup-epochs', type=int, default=10,
                    help='number of warmup epochs (default: 10)')
parser.add_argument('--last-gamma', action='store_true', default=False,
                    help='whether to init gamma of the last BN layer in \
                    each bottleneck to 0 (default: False)')
parser.add_argument('--model', type=str, default='resnet50_v1',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--mode', type=str, default='module',
                    help='mode in which to train the model. options are \
                    module, gluon (default: module)')
parser.add_argument('--use-pretrained', action='store_true', default=False,
                    help='load pretrained model weights (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--eval-epoch', action='store_true', default=False,
                    help='evaluate validation accuracy after each epoch \
                    when training in module mode (default: False)')
parser.add_argument('--eval-frequency', type=int, default=0,
                    help='frequency of evaluating validation accuracy \
                    when training with gluon mode (default: 0)')
parser.add_argument('--log-interval', type=int, default=0,
                    help='number of batches to wait before logging (default: 0)')
parser.add_argument('--save-frequency', type=int, default=0,
                    help='frequency of model saving (default: 0)')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')


args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(args)

# Horovod: initialize Horovod
hvd.init()
num_workers = hvd.size()
rank = hvd.rank()
local_rank = hvd.local_rank()

num_classes = 1000
num_training_samples = 1281167
batch_size = args.batch_size
epoch_size = \
    int(math.ceil(int(num_training_samples // num_workers) / batch_size))

if args.lr_mode == 'step':
    lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
    steps = [epoch_size * x for x in lr_decay_epoch]
    lr_sched = lr_scheduler.MultiFactorScheduler(
        step=steps,
        factor=args.lr_decay,
        base_lr=(args.lr * num_workers),
        warmup_steps=(args.warmup_epochs * epoch_size),
        warmup_begin_lr=args.warmup_lr
    )
elif args.lr_mode == 'poly':
    lr_sched = lr_scheduler.PolyScheduler(
        args.num_epochs * epoch_size,
        base_lr=(args.lr * num_workers),
        pwr=2,
        warmup_steps=(args.warmup_epochs * epoch_size),
        warmup_begin_lr=args.warmup_lr
    )
elif args.lr_mode == 'cosine':
    lr_sched = lr_scheduler.CosineScheduler(
        args.num_epochs * epoch_size,
        base_lr=(args.lr * num_workers),
        warmup_steps=(args.warmup_epochs * epoch_size),
        warmup_begin_lr=args.warmup_lr
    )
else:
    raise ValueError('Invalid lr mode')


# Function for reading data from record file
# For more details about data loading in MXNet, please refer to
# https://mxnet.incubator.apache.org/tutorials/basic/data.html?highlight=imagerecorditer
def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size,
                 data_nthreads):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]

    train_iter = mx.io.ImageRecordIter(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        preprocess_threads=data_nthreads,
        shuffle=True,
        batch_size=batch_size,
        label_width=1,
        data_shape=(3, 224, 224),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        rand_mirror=True,
        rand_crop=False,
        random_resized_crop=True,
        max_aspect_ratio=4. / 3.,
        min_aspect_ratio=3. / 4.,
        max_random_area=1,
        min_random_area=0.08,
        verbose=False,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
        num_parts=num_workers,
        part_index=rank,
        device_id=local_rank
    )
    # Kept each node to use full val data to make it easy to monitor results
    val_iter = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        preprocess_threads=data_nthreads,
        shuffle=False,
        batch_size=batch_size,
        resize=256,
        label_width=1,
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3, 224, 224),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        device_id=local_rank
    )

    return train_iter, val_iter


# Return data and label from batch data
def get_data_label(batch, ctx):
    data = batch.data[0].as_in_context(ctx)
    label = batch.label[0].as_in_context(ctx)
    return data, label


# Create data iterator for synthetic data
class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype, ctx):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size, ])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype,
                                ctx=ctx)
        self.label = mx.nd.array(label, dtype=self.dtype,
                                 ctx=ctx)

    def __iter__(self):
        return self

    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label',
                               (self.batch_size,), self.dtype)]

    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0


# Horovod: pin GPU to local rank
context = mx.cpu(local_rank) if args.no_cuda else mx.gpu(local_rank)

if args.use_rec:
    # Fetch training and validation data if present
    train_data, val_data = get_data_rec(args.rec_train,
                                        args.rec_train_idx,
                                        args.rec_val,
                                        args.rec_val_idx,
                                        batch_size,
                                        args.data_nthreads)
else:
    # Otherwise use synthetic data
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    train_data = SyntheticDataIter(num_classes, data_shape, epoch_size,
                                   np.float32, context)
    val_data = None


# Get model from GluonCV model zoo
# https://gluon-cv.mxnet.io/model_zoo/index.html
kwargs = {'ctx': context,
          'pretrained': args.use_pretrained,
          'classes': num_classes}
if args.last_gamma:
    kwargs['last_gamma'] = True
net = get_model(args.model, **kwargs)
net.cast(args.dtype)

# Create initializer
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                             magnitude=2)


def train_gluon():
    def evaluate(epoch):
        if not args.use_rec:
            return

        val_data.reset()
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(5)
        for _, batch in enumerate(val_data):
            data, label = get_data_label(batch, context)
            output = net(data.astype(args.dtype, copy=False))
            acc_top1.update([label], [output])
            acc_top5.update([label], [output])

        top1_name, top1_acc = acc_top1.get()
        top5_name, top5_acc = acc_top5.get()
        logging.info('Epoch[%d] Rank[%d]\tValidation-%s=%f\tValidation-%s=%f',
                     epoch, rank, top1_name, top1_acc, top5_name, top5_acc)

    # Hybridize and initialize model
    net.hybridize()
    net.initialize(initializer, ctx=context)

    # Horovod: fetch and broadcast parameters
    params = net.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    # Create optimizer
    optimizer_params = {'wd': args.wd,
                        'momentum': args.momentum,
                        'lr_scheduler': lr_sched}
    if args.dtype == 'float16':
        optimizer_params['multi_precision'] = True
    opt = mx.optimizer.create('sgd', **optimizer_params)

    # Horovod: create DistributedTrainer, a subclass of gluon.Trainer
    trainer = hvd.DistributedTrainer(params, opt,
                                     gradient_predivide_factor=args.gradient_predivide_factor)

    # Create loss function and train metric
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()

    # Train model
    for epoch in range(args.num_epochs):
        tic = time.time()
        if args.use_rec:
            train_data.reset()
        metric.reset()

        btic = time.time()
        for nbatch, batch in enumerate(train_data, start=1):
            data, label = get_data_label(batch, context)
            with autograd.record():
                output = net(data.astype(args.dtype, copy=False))
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size)

            metric.update([label], [output])
            if args.log_interval and nbatch % args.log_interval == 0:
                name, acc = metric.get()
                logging.info('Epoch[%d] Rank[%d] Batch[%d]\t%s=%f\tlr=%f',
                             epoch, rank, nbatch, name, acc, trainer.learning_rate)
                if rank == 0:
                    batch_speed = num_workers * batch_size * args.log_interval / (time.time() - btic)
                    logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec',
                                 epoch, nbatch, batch_speed)
                btic = time.time()

        # Report metrics
        elapsed = time.time() - tic
        _, acc = metric.get()
        logging.info('Epoch[%d] Rank[%d] Batch[%d]\tTime cost=%.2f\tTrain-accuracy=%f',
                     epoch, rank, nbatch, elapsed, acc)
        if rank == 0:
            epoch_speed = num_workers * batch_size * nbatch / elapsed
            logging.info('Epoch[%d]\tSpeed: %.2f samples/sec', epoch, epoch_speed)

        # Evaluate performance
        if args.eval_frequency and (epoch + 1) % args.eval_frequency == 0:
            evaluate(epoch)

        # Save model
        if args.save_frequency and (epoch + 1) % args.save_frequency == 0:
            net.export('%s-%d' % (args.model, rank), epoch=epoch)

    # Evaluate performance at the end of training
    evaluate(epoch)


def train_module():
    # Create input symbol
    data = mx.sym.var('data')
    if args.dtype == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)
        net.cast(np.float16)

    # Create output symbol
    out = net(data)
    if args.dtype == 'float16':
        out = mx.sym.Cast(data=out, dtype=np.float32)
    softmax = mx.sym.SoftmaxOutput(out, name='softmax')

    # Create model
    mod = mx.mod.Module(softmax, context=context)

    # Initialize parameters
    if args.use_pretrained:
        arg_params = {}
        for x in net.collect_params().values():
            x.reset_ctx(mx.cpu())
            arg_params[x.name] = x.data()
    else:
        arg_params = None
    aux_params = None
    mod.bind(data_shapes=train_data.provide_data,
             label_shapes=train_data.provide_label)
    mod.init_params(initializer, arg_params=arg_params, aux_params=aux_params)

    # Horovod: fetch and broadcast parameters
    (arg_params, aux_params) = mod.get_params()
    if arg_params is not None:
        hvd.broadcast_parameters(arg_params, root_rank=0)
    if aux_params is not None:
        hvd.broadcast_parameters(aux_params, root_rank=0)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)

    # Create optimizer
    # Note that when using Module API, we need to specify rescale_grad since
    # we create optimizer first and wrap it with DistributedOptimizer. For
    # Gluon API, it is handled in Trainer.step() function so there is no need
    # to specify rescale_grad (see above train_gluon() function). 
    optimizer_params = {'wd': args.wd,
                        'momentum': args.momentum,
                        'rescale_grad': 1.0 / batch_size,
                        'lr_scheduler': lr_sched}
    if args.dtype == 'float16':
        optimizer_params['multi_precision'] = True
    opt = mx.optimizer.create('sgd', **optimizer_params)

    # Horovod: wrap optimizer with DistributedOptimizer
    dist_opt = hvd.DistributedOptimizer(opt,
                                        gradient_predivide_factor=args.gradient_predivide_factor)

    # Setup validation data and callback during training
    eval_data = None
    if args.eval_epoch:
        eval_data = val_data
    batch_callback = None
    if args.log_interval > 0 and rank == 0:
        batch_callback = mx.callback.Speedometer(batch_size * num_workers,
                                                 args.log_interval)

    epoch_callback = None
    if args.save_frequency > 0:
        epoch_callback = mx.callback.do_checkpoint(
            '%s-%d' % (args.model, rank),
            period=args.save_frequency)

    # Train model
    mod.fit(train_data,
            eval_data=eval_data,
            num_epoch=args.num_epochs,
            kvstore=None,
            batch_end_callback=batch_callback,
            epoch_end_callback=epoch_callback,
            optimizer=dist_opt)

    # Evaluate performance if not using synthetic data
    if args.use_rec:
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(5)
        res = mod.score(val_data, [acc_top1, acc_top5])
        for name, val in res:
            logging.info('Epoch[%d] Rank[%d] Validation-%s=%f',
                         args.num_epochs - 1, rank, name, val)


if __name__ == '__main__':
    if args.mode == 'module':
        train_module()
    elif args.mode == 'gluon':
        train_gluon()
    else:
        raise ValueError('Invalid training mode.')
