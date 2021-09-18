Horovod with XLA in Tensorflow
===============================

Basic usage
-----------

XLA Horovod ops can be enabled by setting ``HOROVOD_ENABLE_XLA_OPS = 1`` by controlling the registration of the ops to Tensorflow/XLA.

There are two main ways to enable XLA and they could work with Horovod in different ways:

For **Explicit compilation with tf.function(jit_compile=True)**:

.. code-block:: python

    os.environ["HOROVOD_ENABLE_XLA_OPS"] = "1"

     @tf.function(jit_compile=True)
     def compiled_hvd_allreduce(self, dtype, dim):
         tensor = self.random_uniform(
             [17] * dim, -100, 100, dtype=dtype)
         summed = hvd.allreduce(tensor, average=False)
         return summed

In this way, all the ops in the ``compiled_hvd_allreduce`` function are lowered into XLA per the compilation requirement. If the XLA Horovod ops are not enabled, XLA will report compilation errors.


For **Auto-clustering**:

Auto-clustering is a convenient way to use XLA by simply setting ``TF_XLA_FLAGS=--tf_xla_auto_jit=2`` and the XLA JIT automatically selects ops in the Tensorflow graph to be lowered into XLA. In this mode, enabling XLA Horovod ops is optional, because the auto-clustering can work even if the Horovod ops are left to be run by Tensorflow (devices) while only parts of the graphs are lowered onto XLA (devices).

List of supported XLA Horovod ops
---------------------------------

The supported op list is:

``HorovodAllreduce``

