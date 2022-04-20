Horovod with TensorFlow Data Service
====================================
To run a TensorFlow Data Service via Horovod, run the follow Python command:

   .. code-block:: bash

       python -m horovod.tensorflow.data.compute_worker \
           --compute-service-config-file /tmp/compute.json \
           --dispatchers 2 \
           --workers-per-dispatcher 4 \
           --dispatcher-side compute

This starts a TensorFlow Data Service with ``2`` dispatchers and ``4`` workers per dispatcher. All dispatchers
and workers are started and registered to the service. You training script can then move CPU intensive dataset operations
to this data service by calling ``.send_to_data_service(…)`` on the TensorFlow dataset:

   .. code-block:: bash

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    dataset = dataset.repeat() \
        .shuffle(10000) \
        .batch(128) \
        .send_to_data_service(compute_config, rank, size, reuse_dataset=reuse_dataset, round_robin=round_robin) \
        .prefetch(tf.data.experimental.AUTOTUNE)

All transformations before calling ``send_to_data_service`` will be executed by the data service,
while all transformations after it are executed locally by the training script.

You can find an example in the `examples <https://github.com/horovod/horovod/blob/master/examples/tensorflow2>`_ directory.

First start the data service as shown above. While the data service is running, start the example training script:

   .. code-block:: bash

       python tensorflow2_mnist_data_service.py --compute-service-config-file /tmp/compute.json
