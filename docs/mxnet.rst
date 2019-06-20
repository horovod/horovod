Horovod with MXNet
==================
Horovod supports Apache MXNet and regular TensorFlow in similar ways.

See full training `MNIST <https://github.com/horovod/horovod/blob/master/examples/mxnet_mnist.py>`__ and `ImageNet <https://github.com/horovod/horovod/blob/master/examples/mxnet_imagenet_resnet50.py>`__ examples.
The script below provides a simple skeleton of code block based on the Apache MXNet Gluon API.

.. code-block:: python

    import mxnet as mx
    import horovod.mxnet as hvd
    from mxnet import autograd

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank
    context = mx.gpu(hvd.local_rank())
    num_workers = hvd.size()

    # Build model
    model = ...
    model.hybridize()

    # Create optimizer
    optimizer_params = ...
    opt = mx.optimizer.create('sgd', **optimizer_params)

    # Initialize parameters
    model.initialize(initializer, ctx=context)

    # Fetch and broadcast parameters
    params = model.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    # Create DistributedTrainer, a subclass of gluon.Trainer
    trainer = hvd.DistributedTrainer(params, opt)

    # Create loss function
    loss_fn = ...

    # Train model
    for epoch in range(num_epoch):
        train_data.reset()
        for nbatch, batch in enumerate(train_data, start=1):
            data = batch.data[0].as_in_context(context)
            label = batch.label[0].as_in_context(context)
            with autograd.record():
                output = model(data.astype(dtype, copy=False))
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size)



.. NOTE:: The `known issue <https://github.com/horovod/horovod/issues/884>`__ when running Horovod with MXNet on a Linux system with GCC version 5.X and above has been resolved. Please use MXNet 1.4.1 or later releases with Horovod 0.16.2 or later releases to avoid the GCC incompatibility issue. MXNet 1.4.0 release works with Horovod 0.16.0 and 0.16.1 releases with the GCC incompatibility issue unsolved.
