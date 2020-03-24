
.. inclusion-marker-start-do-not-remove


Benchmarks
==========


.. image:: https://user-images.githubusercontent.com/16640218/38965607-bf5c46ca-4332-11e8-895a-b9c137e86013.png
   :alt: 512-GPU Benchmark


The above benchmark was done on 128 servers with 4 Pascal GPUs each connected by a RoCE-capable 25 Gbit/s network. Horovod
achieves 90% scaling efficiency for both Inception V3 and ResNet-101, and 68% scaling efficiency for VGG-16.

To reproduce the benchmarks:

1. Install Horovod using the instructions provided on the `Horovod on GPU <https://github.com/horovod/horovod/blob/master/docs/gpus.rst>`__ page.

2. Clone `https://github.com/tensorflow/benchmarks <https://github.com/tensorflow/benchmarks>`__

.. code-block:: bash

    $ git clone https://github.com/tensorflow/benchmarks
    $ cd benchmarks


3. Run the benchmark. Examples below are for Open MPI.

.. code-block:: bash

    $ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 \
        python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
            --model resnet101 \
            --batch_size 64 \
            --variable_update horovod


4. At the end of the run, you will see the number of images processed per second:

.. code-block:: bash

    total images/sec: 1656.82


Real data benchmarks
~~~~~~~~~~~~~~~~~~~~
The benchmark instructions above are for the synthetic data benchmark.

To run the benchmark on a real data, you need to download the `ImageNet dataset <http://image-net.org/download-images>`__
and convert it using the TFRecord `preprocessing script <https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh>`__.

Now, simply add ``--data_dir /path/to/imagenet/tfrecords --data_name imagenet --num_batches=2000`` to your training command:

.. code-block:: bash

    $ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 \
        python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
            --model resnet101 \
            --batch_size 64 \
            --variable_update horovod \
            --data_dir /path/to/imagenet/tfrecords \
            --data_name imagenet \
            --num_batches=2000


Horovod synthetic benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Horovod also comes with out-of-the-box benchmarking support for
`TensorFlow v1 <https://github.com/horovod/horovod/tree/master/examples/tensorflow_synthetic_benchmark.py>`__,
`TensorFlow v2 <https://github.com/horovod/horovod/tree/master/examples/tensorflow2_synthetic_benchmark.py>`__, and
`PyTorch <https://github.com/horovod/horovod/tree/master/examples/pytorch_synthetic_benchmark.py>`__.

These benchmarks allow you to measure Horovod's performance and scalability in your environment, as well as try advanced
Horovod features like gradient compression:

.. code-block:: bash

    $ horovodrun -np 4 -H server1:2,server2:2 \
        python --fp16-allreduce tensorflow2_synthetic_benchmark.py

When diagnosing performance issues, we recommend running these synthetic benchmarks first to ensure that the issues are
not originating from the training script itself.

.. inclusion-marker-end-do-not-remove
