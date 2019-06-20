Horovod with Keras
==================
Horovod supports Keras and regular TensorFlow in similar ways.

See full training `simple <https://github.com/horovod/horovod/blob/master/examples/keras_mnist.py>`_ and `advanced <https://github.com/horovod/horovod/blob/master/examples/keras_mnist_advanced.py>`_ examples.

.. NOTE:: Keras 2.0.9 has a `known issue <https://github.com/fchollet/keras/issues/8353>`_ that makes each worker allocate all GPUs on the server, instead of the GPU assigned by the *local rank*. If you have multiple GPUs per server, upgrade to Keras 2.1.2 or downgrade to Keras 2.0.8.