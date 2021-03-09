Horovod with MXNet
==================
Horovod supports Apache MXNet and regular TensorFlow in similar ways.

See full training `MNIST <https://github.com/horovod/horovod/blob/master/examples/mxnet/mxnet_mnist.py>`__ and `ImageNet <https://github.com/horovod/horovod/blob/master/examples/mxnet/mxnet_imagenet_resnet50.py>`__ examples.
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



.. NOTE:: Some MXNet versions do not work with Horovod:

    - MXNet 1.4.0 and earlier have `GCC incompatibility issues <https://github.com/horovod/horovod/issues/884>`__. Use MXNet 1.4.1 or later with Horovod 0.16.2 or later to avoid these incompatibilities.
    - MXNet 1.5.1, 1.6.0, 1.7.0, and 1.7.0.post1 are missing MKLDNN headers, so they do not work with Horovod. Use 1.5.1.post0, 1.6.0.post0, and 1.7.0.post0, respectively.
    - MXNet 1.6.0.post0 and 1.7.0.post0 are only available as mxnet-cu101 and mxnet-cu102.
