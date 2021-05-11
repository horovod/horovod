import os
import sys
import unittest

import horovod.tensorflow as hvd

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import mpi_env_rank_and_size

class ProcessSetsStaticTests(unittest.TestCase):
    """ Since this test case initializes Horovod and shuts it down, it must be run in a separate process. """
    def test_static(self):
        _, mpi_size = mpi_env_rank_and_size()
        gloo_size = int(os.getenv('HOROVOD_SIZE', -1))
        if gloo_size != -1:
            self.skipTest("Multiple process sets currently do not support Gloo controller.")

        size = max(mpi_size, gloo_size)

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        hvd.init(process_sets=[[0], list(range(1, size))])
        ps = hvd.get_process_sets()
        self.assertDictEqual(ps, {0: list(range(size)),
                                  1: [0],
                                  2: list(range(1, size))})

        # barrier before shutdown
        import mpi4py
        mpi4py.rc.initialize = False
        from mpi4py import MPI
        MPI.COMM_WORLD.barrier()

        hvd.shutdown()
