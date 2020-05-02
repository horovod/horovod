Horovod with PyTorch
====================
To use Horovod with PyTorch, make the following modifications to your training script:

1. Run ``hvd.init()``.

.. raw:: html

    <p/>

2. Pin each GPU to a single process.

   With the typical setup of one GPU per process, set this to *local rank*. The first process on
   the server will be allocated the first GPU, the second process will be allocated the second GPU, and so forth.

   .. code-block:: python

       if torch.cuda.is_available():
           torch.cuda.set_device(hvd.local_rank())

.. raw:: html

    <p/>

3. Scale the learning rate by the number of workers.

   Effective batch size in synchronous distributed training is scaled by the number of workers.
   An increase in learning rate compensates for the increased batch size.

.. raw:: html

    <p/>

4. Wrap the optimizer in ``hvd.DistributedOptimizer``.

   The distributed optimizer delegates gradient computation to the original optimizer, averages gradients using *allreduce* or *allgather*, and then applies those averaged gradients.

.. raw:: html

    <p/>

5. Broadcast the initial variable states from rank 0 to all other processes:

   .. code-block:: python

       hvd.broadcast_parameters(model.state_dict(), root_rank=0)
       hvd.broadcast_optimizer_state(optimizer, root_rank=0)

   This is necessary to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.

.. raw:: html

    <p/>

6. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them.

   Accomplish this by guarding model checkpointing code with ``hvd.rank() != 0``.

.. raw:: html

    <p/>

Example (also see a full training `example <https://github.com/horovod/horovod/blob/master/examples/pytorch_mnist.py>`__):

.. code-block:: python

    import torch
    import horovod.torch as hvd

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    # Define dataset...
    train_dataset = ...

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

    # Build model...
    model = ...
    model.cuda()

    optimizer = optim.SGD(model.parameters())

    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    for epoch in range(100):
       for batch_idx, (data, target) in enumerate(train_loader):
           optimizer.zero_grad()
           output = model(data)
           loss = F.nll_loss(output, target)
           loss.backward()
           optimizer.step()
           if batch_idx % args.log_interval == 0:
               print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(
                   epoch, batch_idx * len(data), len(train_sampler), loss.item()))


.. NOTE:: PyTorch GPU support requires NCCL 2.2 or later. It also works with NCCL 2.1.15 if you are not using RoCE or InfiniBand.


PyTorch Lightning
-----------------

Horovod is supported as a distributed backend in `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ from v0.7.4 and above.

With PyTorch Lightning, distributed training using Horovod requires only a single line code change to your existing training script:

.. code-block:: python

    # train Horovod on GPU (number of GPUs / machines provided on command-line)
    trainer = pl.Trainer(distributed_backend='horovod', gpus=1)

    # train Horovod on CPU (number of processes / machines provided on command-line)
    trainer = pl.Trainer(distributed_backend='horovod')

Start the training job and specify the number of workers on the command line as you normally would when using Horovod:

.. code-block:: bash

    # run training with 4 GPUs on a single machine
    $ horovodrun -np 4 python train.py

    # run training with 8 GPUs on two machines (4 GPUs each)
    $ horovodrun -np 8 -H hostname1:4,hostname2:4 python train.py

See the PyTorch Lightning `docs <https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html#horovod>`_ for more details.
