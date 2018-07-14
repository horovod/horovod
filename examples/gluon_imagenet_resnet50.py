import argparse, time, logging, os

import numpy as np

import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

# pip install gluoncv for model definitions

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs
import re, math
from lr_scheduler import MultiFactorScheduler, PolyScheduler, WarmupScheduler

#import hvd.gluon as hvd

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                    help='training and validation pictures to use.')
parser.add_argument('--rec-train', type=str, default='/media/ramdisk/pass-through/train-passthrough.rec',
                    help='the training data')
parser.add_argument('--rec-train-idx', type=str, default='/media/ramdisk/pass-through/train-passthrough.idx',
                    help='the index of training data')
parser.add_argument('--rec-val', type=str, default='/media/ramdisk/pass-through/val-passthrough.rec',
                    help='the validation data')
parser.add_argument('--rec-val-idx', type=str, default='/media/ramdisk/pass-through/val-passthrough.idx',
                    help='the index of validation data')
parser.add_argument('--use-rec', action='store_true', default='true',
                    help='use image record iter for data input. default is true.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--gpus', type=str, default='0',
                    help='number of gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=32, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-mode', type=str, default='step',
                    help='learning rate scheduler mode. options are step, poly.')
parser.add_argument('--lr-poly-power', type=int, default=2,
                    help='if learning rate scheduler mode is poly, then power is used')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--last-gamma', action='store_true',
                    help='whether to initialize the gamma of the last BN layer in each bottleneck to zero')
parser.add_argument('--mode', type=str, default='hybrid',
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--save-frequency', type=int, default=0,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--logging-dir', type=str, default='logs',
                    help='directory of training logs')
parser.add_argument('--kvstore', type=str, default='device')
parser.add_argument('--top-k', type=int, default=0, help='give 5 for top5 accuracy, if 0 only prints top1 accuracy')
opt = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(opt)

batch_size = opt.batch_size
classes = 1000
num_training_samples = 1281167

num_gpus = len(opt.gpus.split(','))
batch_size *= max(1, num_gpus)
context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

kv = mx.kv.create(opt.kvstore)

num_batches = int(math.ceil(int(num_training_samples // kv.num_workers)/batch_size))
epoch_size = num_batches

if opt.lr_mode == 'poly':
    lr_scheduler = PolyScheduler(opt.num_epochs * epoch_size, base_lr=opt.lr, pwr=opt.lr_poly_power, warmup_steps=(opt.warmup_epochs * epoch_size))
elif opt.lr_mode == 'step':
    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    steps = [epoch_size * x for x in lr_decay_epoch]
    lr_scheduler = MultiFactorScheduler(step=steps, factor=lr_decay, base_lr=opt.lr)
else:
    raise ValueError

if opt.warmup_epochs > 0:
    lr_scheduler = WarmupScheduler(base_lr=opt.warmup_lr, warmup_steps=(opt.warmup_epochs * epoch_size), scheduler=lr_scheduler)

model_name = opt.model

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
if model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm
elif model_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

if opt.last_gamma and model_name == 'resnet50_v1b':
    kwargs['last_gamma'] = True

optimizer = 'sgd'
optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True

net = get_model(model_name, **kwargs)
net.cast(opt.dtype)

# Two functions for reading data from record file or raw images
def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,
        label_width         = 1,
        data_shape          = (3, 224, 224),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        rand_mirror         = True,
        rand_crop           = False,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        verbose             = False,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
        num_parts           = kv.num_workers,
        part_index          = kv.rank
    )
    # kept each node to use full val data to make it easy to monitor results
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,
        resize              = 256,
        label_width         = 1,
        rand_crop           = False,
        rand_mirror         = False,
        data_shape          = (3, 224, 224),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2]
    )

    if 'dist' in opt.kvstore and not 'async' in opt.kvstore:
        train_data = mx.io.ResizeIter(train_data, epoch_size)

    return train_data, val_data, batch_fn

def get_data_loader(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256, keep_ratio=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if 'sync' in opt.kvstore:
        raise ValueError("Need to resize iterator for distributed training to not hang at the end")
    
    return train_data, val_data, batch_fn

if opt.use_rec:
    train_data, val_data, batch_fn = get_data_rec(opt.rec_train, opt.rec_train_idx,
                                                  opt.rec_val, opt.rec_val_idx,
                                                  batch_size, num_workers)
else:
    train_data, val_data, batch_fn = get_data_loader(opt.data_dir, batch_size, num_workers)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)

save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0

def test(ctx, val_data):
    if opt.use_rec:
        val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

def train(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(initializer, ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, kvstore=kv)

    L = gluon.loss.SoftmaxCrossEntropyLoss()

    best_val_score = 1

    for epoch in range(opt.num_epochs):
        tic = time.time()
        if opt.use_rec:
            train_data.reset()
        acc_top1.reset()
        btic = time.time()

        for i, batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)
            with ag.record():
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)

            acc_top1.update(label, outputs)
            if opt.log_interval and not (i+1)%opt.log_interval:
                _, top1 = acc_top1.get()
                err_top1 = 1-top1
                logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\tlr=%f\taccuracy=%f'%(
                             epoch, i, batch_size*opt.log_interval/(time.time()-btic), trainer.learning_rate, top1))
                btic = time.time()

        _, top1 = acc_top1.get()
        err_top1 = 1-top1
        throughput = int(batch_size * i /(time.time() - tic))

        err_top1_val, err_top5_val = test(ctx, val_data)

        logging.info('[Epoch %d] Train-accuracy=%f'%(epoch, top1))
        logging.info('[Epoch %d] Speed: %d samples/sec\tTime cost=%f'%(epoch, throughput, time.time()-tic))
        logging.info('[Epoch %d] Validation-accuracy=%f'%(epoch, 1 - err_top1_val))
        logging.info('[Epoch %d] Validation-top_k_accuracy_5=%f'%(epoch, 1 - err_top5_val))

        if save_frequency and err_top1_val < best_val_score and epoch > 50:
            best_val_score = err_top1_val
            net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, opt.num_epochs-1))

def main():
    if opt.mode == 'symbolic':
        data = mx.sym.var('data')
        if opt.dtype == 'float16':
            data = mx.sym.Cast(data=data, dtype=np.float16)
            net.cast(np.float16)
        out = net(data)
        if opt.dtype == 'float16':
            out = mx.sym.Cast(data=out, dtype=np.float32)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=context)
        if opt.use_pretrained:
            arg_params = {} 
            for x in net.collect_params().values():
                x.reset_ctx(mx.cpu())
                arg_params[x.name] = x.data()
        else:
            arg_params = None
        mod.fit(train_data,
                arg_params=arg_params,
                eval_data = val_data,
                num_epoch=opt.num_epochs,
                kvstore=kv,
                batch_end_callback = mx.callback.Speedometer(batch_size, max(1, opt.log_interval)),
                epoch_end_callback = mx.callback.do_checkpoint('imagenet-%s'% opt.model, period=save_frequency),
                optimizer = optimizer,
                optimizer_params=optimizer_params,
                initializer=initializer)
        if opt.save_frequency:
            mod.save_parameters('imagenet-%s-%d-final.params'%(opt.model, opt.epochs))
    else:
        if opt.mode == 'hybrid':
            net.hybridize(static_alloc=True, static_shape=True)
        train(context)

if __name__ == '__main__':
    main()
