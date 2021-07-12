import os
import sys
import time
import unittest

import horovod.tensorflow as hvd

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import mpi_env_rank_and_size

class ProcessSetsStaticTests(unittest.TestCase):
    """ Since this test case initializes Horovod and shuts it down, it must be run in a separate process. """
    def test_static(self):
        mpi_rank, mpi_size = mpi_env_rank_and_size()
        gloo_rank = int(os.getenv('HOROVOD_RANK', -1))
        gloo_size = int(os.getenv('HOROVOD_SIZE', -1))
        is_mpi = gloo_rank == -1

        rank = max(mpi_rank, gloo_rank)
        size = max(mpi_size, gloo_size)

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        if is_mpi:
            try:
                import mpi4py
                mpi4py.rc.initialize = False
            except ImportError:
                pass

        if rank == 0:
            my_process_sets = [hvd.ProcessSet([0]),
                               hvd.ProcessSet(range(1, size)),
                               hvd.ProcessSet(range(size - 1, -1, -1)),  # duplicate
                               hvd.ProcessSet([0])  # duplicate
                               ]
        else:
            my_process_sets = [hvd.ProcessSet([0]),
                               hvd.ProcessSet(reversed(range(1, size))),  # permuting a process set does not matter
                               hvd.ProcessSet(range(size - 1, -1, -1)),  # duplicate
                               hvd.ProcessSet([0])  # duplicate
                               ]
        with self.assertRaises(ValueError):
            hvd.init(process_sets=my_process_sets)

        if rank == 0:
            my_process_sets = [hvd.ProcessSet([0]),
                               hvd.ProcessSet(range(1, size)),
                               ]
        else:
            my_process_sets = [hvd.ProcessSet([0]),
                               hvd.ProcessSet(reversed(range(1, size))),  # permuting a process set does not matter
                               ]
        hvd.init(process_sets=my_process_sets)

        self.assertEqual(hvd.global_process_set.process_set_id, 0)
        self.assertListEqual(hvd.global_process_set.ranks, list(range(size)))

        # Here we test some implementation details (numeric process set id values) using an internal function.
        ps = hvd.mpi_ops._basics._get_process_set_ids_and_ranks()
        self.assertDictEqual(ps, {0: list(range(size)),
                                  1: [0],
                                  2: list(range(1, size))})

        # If another process initiates shutdown while this process is still processing _get_process_set_ids_and_ranks(),
        # a race condition may be triggered. Avoid with a barrier.
        try:
            if is_mpi:
                # barrier before shutdown
                from mpi4py import MPI
                MPI.COMM_WORLD.barrier()
            else:
                time.sleep(0.1)
        except ImportError:
            time.sleep(0.1)

        hvd.shutdown()
