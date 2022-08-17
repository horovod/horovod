"""Tests for horovod.tensorflow.mpi_ops that add/remove process sets after initialization.

With TensorFlow 2.9 and MPI the option HOROVOD_DYNAMIC_PROCESS_SETS has been observed to cause significant
slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we collect
tests that depend on that setting in this script.
"""

import tensorflow as tf

import horovod.tensorflow as hvd

from base_test_tensorflow import *

# Set environment variable to enable adding/removing process sets after initializing Horovod.
os.environ["HOROVOD_DYNAMIC_PROCESS_SETS"] = "1"

class TensorFlowProcessSetsDynamicTests(BaseTensorFlowTests):
    """
    Tests for ops in horovod.tensorflow that add/remove process sets after initialization.
    """
    def __init__(self, *args, **kwargs):
        super(TensorFlowProcessSetsDynamicTests, self).__init__(*args, **kwargs)

    def tearDown(self):
        """Prevent that one process shuts down Horovod too early"""
        with tf.device("/cpu:0"):
            b = hvd.allreduce(tf.constant([0.]), name="global_barrier_after_test")
            _ = self.evaluate(b)

    def test_horovod_add_get_remove_process_set(self):
        hvd.init()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Here we test some implementation details (numeric process set id values) using an internal function. We only
        # test the concrete value 0 because IDs will be reassigned between eager and graph-mode test runs and may
        # change.
        ps = hvd.mpi_ops._basics._get_process_set_ids_and_ranks()
        self.assertDictEqual(ps, {0: list(range(size))})

        set1 = hvd.add_process_set([0])
        set2 = hvd.add_process_set(range(1, size))

        ps = hvd.mpi_ops._basics._get_process_set_ids_and_ranks()
        self.assertDictEqual(ps, {0: list(range(size)),
                                  set1.process_set_id: [0],
                                  set2.process_set_id: list(range(1, size))})

        # Ensure process set ids are equal across processes.
        with tf.device("/cpu:0"):
            for a_set in [set1, set2]:
                ids_on_ranks = list(self.evaluate(hvd.allgather(tf.convert_to_tensor([a_set.process_set_id]))))
                self.assertTrue(all(an_id == a_set.process_set_id for an_id in ids_on_ranks))

        # Test stringification
        self.assertListEqual([str(p) for p in [hvd.global_process_set, set1, set2]],
                             [f"ProcessSet(process_set_id=0, ranks={list(range(size))}, mpi_comm=None)",
                              f"ProcessSet(process_set_id={set1.process_set_id}, ranks=[0], mpi_comm=None)",
                              f"ProcessSet(process_set_id={set2.process_set_id}, ranks={list(range(1, size))}, mpi_comm=None)",
                              ])

        old_id_of_set1 = set1.process_set_id
        hvd.remove_process_set(set1)
        self.assertIsNone(set1.process_set_id)  # invalidated

        ps = hvd.mpi_ops._basics._get_process_set_ids_and_ranks()
        self.assertDictEqual(ps, {0: list(range(size)),
                                  set2.process_set_id: list(range(1, size))})

        # test re-adding set1
        hvd.add_process_set(set1)
        ps = hvd.mpi_ops._basics._get_process_set_ids_and_ranks()
        self.assertDictEqual(ps, {0: list(range(size)),
                                  set1.process_set_id: [0],
                                  set2.process_set_id: list(range(1, size))})
        hvd.remove_process_set(set1)


        if size > 2:
            set3 = hvd.add_process_set([0, size - 1])
            self.assertEqual(old_id_of_set1, set3.process_set_id) # id reuse
        else:
            with self.assertRaises(ValueError):  # duplicate of the global process set
                set3 = hvd.add_process_set([0, size - 1])
            set3 = hvd.global_process_set

        with self.assertRaises(ValueError):  # duplicate of set2
            set4 = hvd.add_process_set(range(size - 1, 0, -1))

        with self.assertRaises(ValueError):  # duplicate of the global process set
            set5 = hvd.add_process_set(range(0, size))

        ps = hvd.mpi_ops._basics._get_process_set_ids_and_ranks()
        if size > 2:
            self.assertDictEqual(ps, {0: list(range(size)),
                                      set2.process_set_id: list(range(1, size)),
                                      set3.process_set_id: [0, size-1]})
        else:
            self.assertDictEqual(ps, {0: list(range(size)),
                                      set2.process_set_id: list(range(1, size))})
        hvd.remove_process_set(set2)
        hvd.remove_process_set(set3)

        ps = hvd.mpi_ops._basics._get_process_set_ids_and_ranks()
        self.assertDictEqual(ps, {0: list(range(size))})

        self.assertFalse(hvd.remove_process_set(hvd.global_process_set),
                         "Removing the global process set should be impossible.")
