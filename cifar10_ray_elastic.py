import argparse
import os
import time
import random
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level="DEBUG")# , filename='example.log')
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import horovod.torch as hvd
from horovod.torch.elastic.sampler import ElasticSampler
from horovod.ray.elastic import RayHostDiscovery

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--data-dir', default='./new_data',
                    help='cifar10 dataset directory')

parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate for a single GPU')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--change-frequency-s', type=int, default=60,
                    help='random seed')

# Elastic Horovod settings
parser.add_argument('--batches-per-commit', type=int, default=50,
                    help='number of batches processed before calling `state.commit()`; '
                         'commits prevent losing progress if an error occurs, but slow '
                         'down training.')
parser.add_argument('--batches-per-host-check', type=int, default=10,
                    help='number of batches processed before calling `state.check_host_updates()`; '
                         'this check is very fast compared to state.commit() (which calls this '
                         'as part of the commit process), but because still incurs some cost due '
                         'to broadcast, so we may not want to perform it every batch.')

args = parser.parse_args()


class TestDiscovery(RayHostDiscovery):
    def __init__(self, min_hosts, max_hosts, change_frequency_s, use_gpu=False, cpus_per_slot=1, gpus_per_slot=1):
        super().__init__(use_gpu=use_gpu, cpus_per_slot=cpus_per_slot, gpus_per_slot=gpus_per_slot)
        self._min_hosts = min_hosts
        self._max_hosts = max_hosts
        self._change_frequency_s = change_frequency_s
        self._last_reset_t = None
        self._removed_hosts = set()

    def add_host(self, hosts):
        available_hosts = self._removed_hosts & hosts.keys()
        if available_hosts:
            host = random.choice(list(available_hosts))
            print('ADD HOST', host, hosts, self._removed_hosts)
            self._removed_hosts.remove(host)

        else:
            print("No hosts to add.")

    def remove_host(self, hosts):
        from ray.autoscaler._private.commands import kill_node
        # host = kill_node(os.path.expanduser("~/ray_bootstrap_config.yaml"), True, False, None)
        good_hosts = [k for k in hosts if k not in self._removed_hosts]
        if good_hosts:
            host = random.choice(good_hosts)
            print('REMOVE HOST', host, hosts, self._removed_hosts)
            self._removed_hosts.add(host)

    def change_hosts(self, hosts):
        for host in self._removed_hosts:
            if host not in hosts:
                self._removed_hosts.remove(host)
        current_hosts = len(hosts) - len(self._removed_hosts)
        if current_hosts <= self._min_hosts:
            self.add_host(hosts)
        elif current_hosts >= self._max_hosts:
            self.remove_host(hosts)
        else:
            if random.random() < 0.5:
                self.add_host(hosts)
            else:
                self.remove_host(hosts)

    def find_available_hosts_and_slots(self):
        t = time.time()
        if self._last_reset_t is None:
            self._last_reset_t = t
        hosts = super().find_available_hosts_and_slots()
        if t - self._last_reset_t >= self._change_frequency_s:
            self.change_hosts(hosts)
            self._last_reset_t = t
        print(f"Total hosts: {len(hosts)}")
        remaining = {k: v for k, v in hosts.items() if k not in self._removed_hosts}
        print(f"Remaining hosts: {len(remaining)} -- {remaining}")
        return remaining


def load_data():
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    # kwargs['multiprocessing_context'] = 'spawn'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    from filelock import FileLock
    with FileLock(os.path.expanduser("~/.datalock")):
        train_dataset = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform_train)
    train_sampler = ElasticSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4,
        sampler=train_sampler, **kwargs)

    val_dataset = datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test)
    val_sampler = ElasticSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                             sampler=val_sampler, **kwargs)

    return train_loader, val_loader, train_sampler, val_sampler


def train(state, train_loader, log_writer, verbose):
    epoch = state.epoch
    batch_offset = state.batch

    state.model.train()
    state.train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:

        for batch_idx, (data, target) in enumerate(train_loader):
            # Elastic Horovod: update the current batch index this epoch
            # and commit / check for host updates. Do not check hosts when
            # we commit as it would be redundant.
            state.batch = batch_offset + batch_idx
            if args.batches_per_commit > 0 and \
                    state.batch % args.batches_per_commit == 0:
                state.commit()
            elif args.batches_per_host_check > 0 and \
                    state.batch % args.batches_per_host_check == 0:
                state.check_host_updates()

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            state.optimizer.zero_grad()

            output = state.model(data)
            train_accuracy.update(accuracy(output, target))

            loss = F.cross_entropy(output, target)
            train_loss.update(loss)
            loss.backward()
            state.optimizer.step()

            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(state, val_loader, log_writer, verbose):
    state.model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(state.epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = state.model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, state.epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, state.epoch)


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(state):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=state.epoch + 1)
        state = {
            'model': state.model.state_dict(),
            'optimizer': state.optimizer.state_dict(),
            'scheduler': state.scheduler.state_dict(),
        }
        torch.save(state, filepath)


def end_epoch(state):
    state.epoch += 1
    state.batch = 0
    state.train_sampler.set_epoch(state.epoch)
    state.commit()


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def run():
    import logging
    # logging.basicConfig(level="DEBUG")
    hvd.init()

    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Load cifar10 dataset
    train_loader, val_loader, train_sampler, val_sampler = load_data()

    # Set up standard ResNet-50 model.
    model = ResNet50()
    if args.cuda:
        model.cuda()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr * np.sqrt(hvd.size()),
                          momentum=0.9, weight_decay=5e-4)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    def on_state_reset():
        # Horovod: scale the learning rate as controlled by the LR schedule
        scheduler.base_lrs = [args.lr * hvd.size() for _ in scheduler.base_lrs]

    state = hvd.elastic.TorchState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        epoch=resume_from_epoch,
        batch=0)
    state.register_reset_callbacks([on_state_reset])

    @hvd.elastic.run
    def full_train(state, train_loader, val_loader):
        # Horovod: print logs on the first worker.
        verbose = 1 if hvd.rank() == 0 else 0

        # Horovod: write TensorBoard logs on first worker.
        # log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
        log_writer = None
        while state.epoch < args.epochs:
            train(state, train_loader, log_writer, verbose)
            validate(state, val_loader, log_writer, verbose)
            state.scheduler.step()
            save_checkpoint(state)
            end_epoch(state)

    full_train(state, train_loader, val_loader)


if __name__ == '__main__':
    import logging
    from horovod.ray import ElasticRayExecutor
    import ray
    ray.init(address="auto")
    settings = ElasticRayExecutor.create_settings(verbose=2)
    # settings.discovery = TestDiscovery(
    #     min_hosts=2,
    #     max_hosts=5,
    #     change_frequency_s=args.change_frequency_s,
    #     use_gpu=True,
    #     cpus_per_slot=1
    # )
    executor = ElasticRayExecutor(settings, use_gpu=True, cpus_per_slot=1)#, override_discovery=False)
    executor.start()
    executor.run(run)
