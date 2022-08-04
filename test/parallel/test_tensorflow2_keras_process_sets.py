"""Tests for horovod.tensorflow.keras in TensorFlow 2 using multiple process sets.

With TensorFlow 2.9 and MPI the option HOROVOD_DYNAMIC_PROCESS_SETS has been observed to cause significant
slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we collect
tests for multiple process sets in this script that initializes Horovod with static process sets.
"""

import tensorflow as tf
import os
import sys
import warnings
from distutils.version import LooseVersion
import pytest
from tensorflow import keras

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

import horovod.tensorflow.keras as hvd

from common import mpi_env_rank_and_size


_PRE_TF_2_2_0 = LooseVersion(tf.__version__) < LooseVersion("2.2.0")

# Set environment variable to enable adding/removing process sets after
# initializing Horovod.
os.environ["HOROVOD_DYNAMIC_PROCESS_SETS"] = "1"


@pytest.mark.skipif(LooseVersion(tf.__version__) <
                    LooseVersion('2.0.0'), reason='TensorFlow v2 tests')
class Tf2KerasProcessSetsTests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.keras using multiple process sets.
    """

    def __init__(self, *args, **kwargs):
        super(Tf2KerasProcessSetsTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    @classmethod
    def setUpClass(cls):
        """Initializes Horovod with two process sets"""
        _, mpi_size = mpi_env_rank_and_size()
        gloo_size = int(os.getenv('HOROVOD_SIZE', -1))
        size = max(mpi_size, gloo_size)

        cls.even_ranks = [rk for rk in range(0, size) if rk % 2 == 0]
        cls.odd_ranks = [rk for rk in range(0, size) if rk % 2 == 1]
        cls.even_set = hvd.ProcessSet(cls.even_ranks)
        cls.odd_set = hvd.ProcessSet(cls.odd_ranks)

        hvd.init(process_sets=[cls.even_set, cls.odd_set])

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
        opt = hvd.DistributedOptimizer(opt, process_set=self.even_set)

        variable = tf.Variable([0.0])
        gradient, = opt.get_gradients(None, [variable])
        opt.apply_gradients([(gradient, variable)])
        computed_value = variable.numpy()

        if self.even_set.included():
            self.assertAlmostEqual(computed_value,
                                   sum(range(0, size, 2)) / self.even_set.size())
        else:
            self.assertAlmostEqual(computed_value, float(hvd.rank()))
