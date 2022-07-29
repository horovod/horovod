"""Tests for horovod.tensorflow.keras in TensorFlow 2 that add/remove process sets after initialization.

With TensorFlow 2.9 and MPI the option HOROVOD_DYNAMIC_PROCESS_SETS has been observed to cause significant
slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we separate
out tests that depend on that setting to this script.
"""

import tensorflow as tf
import os
import warnings

from distutils.version import LooseVersion

import pytest

from tensorflow import keras

import horovod.tensorflow.keras as hvd


_PRE_TF_2_2_0 = LooseVersion(tf.__version__) < LooseVersion("2.2.0")

# Set environment variable to enable adding/removing process sets after
# initializing Horovod.
os.environ["HOROVOD_DYNAMIC_PROCESS_SETS"] = "1"


@pytest.mark.skipif(LooseVersion(tf.__version__) <
                    LooseVersion('2.0.0'), reason='TensorFlow v2 tests')
class Tf2KerasProcessSetsTests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.keras that add/remove process sets after initialization.
    """

    def __init__(self, *args, **kwargs):
        super(Tf2KerasProcessSetsTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        hvd.init()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()], 'GPU')

    def test_process_set_optimizer(self):
        """ Note that this test makes the most sense when running with > 2 processes. """
        size = hvd.size()

        if size == 1:
            self.skipTest("Only one worker available")

        subset = hvd.add_process_set(range(0, size, 2))

        class TestOptimizer(keras.optimizers.Optimizer):
            def __init__(self, name, **kwargs):
                super(TestOptimizer, self).__init__(name, **kwargs)

            def get_gradients(self, loss, params):
                assert len(params) == 1
                return [tf.constant([float(hvd.rank())])]

            def _create_slots(self, var_list):
                pass

            def _resource_apply_dense(self, grad, var, apply_state):
                return var.assign_add(grad)

            def get_config(self):
                config = super(TestOptimizer, self).get_config()
                return config

        opt = TestOptimizer(name="TestOpti")
        opt = hvd.DistributedOptimizer(opt, process_set=subset)

        variable = tf.Variable([0.0])
        gradient, = opt.get_gradients(None, [variable])
        opt.apply_gradients([(gradient, variable)])
        computed_value = variable.numpy()

        if subset.included():
            self.assertAlmostEqual(
                computed_value, sum(
                    range(
                        0, size, 2)) / subset.size())
        else:
            self.assertAlmostEqual(computed_value, float(hvd.rank()))

        hvd.remove_process_set(subset)
