"""Tests for horovod.tensorflow.keras in TensorFlow 2 using multiple process sets.

With TensorFlow 2.9 and MPI the option HOROVOD_DYNAMIC_PROCESS_SETS has been observed to cause significant
slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we collect
tests for multiple process sets in this script that initializes Horovod with static process sets.
"""

import tensorflow as tf
import warnings
from packaging import version
import pytest
from tensorflow import keras

import horovod.tensorflow.keras as hvd
from horovod.runner.common.util.env import get_env_rank_and_size


_PRE_TF_2_2_0 = version.parse(tf.__version__) < version.parse("2.2.0")

@pytest.mark.skipif(version.parse(tf.__version__) <
                    version.parse('2.0.0'), reason='TensorFlow v2 tests')
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
        _, size = get_env_rank_and_size()

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

    def tearDown(self):
        """Prevent that one process shuts down Horovod too early"""
        with tf.device("/cpu:0"):
            b = hvd.allreduce(tf.constant([0.]), name="global_barrier_after_test")
            _ = self.evaluate(b)

    def test_process_set_optimizer(self):
        """ Note that this test makes the most sense when running with > 2 processes. """
        size = hvd.size()

        if size == 1:
            self.skipTest("Only one worker available")

        optimizer_class = keras.optimizers.Optimizer

        class TestOptimizer(optimizer_class):
            def __init__(self, name, **kwargs):
                super().__init__(name=name, **kwargs)
                if hasattr(self, '_build_learning_rate'):
                    self._learning_rate = self._build_learning_rate(0.1)

            def get_gradients(self, loss, params):
                assert len(params) == 1
                return [tf.constant([float(hvd.rank())])]

            def _create_slots(self, var_list):
                pass

            def _resource_apply_dense(self, grad, var, apply_state):
                return var.assign_add(grad)

            def get_config(self):
                config = super().get_config()
                return config

            def update_step(self, gradient, variable):
                variable.assign_add(gradient)

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
