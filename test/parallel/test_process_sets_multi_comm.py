import os
import sys
import unittest

import horovod.tensorflow as hvd

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

class ProcessSetsMultiCommTests(unittest.TestCase):
    """ Since this test case initializes Horovod and shuts it down, it must be run in a separate process. """

    def test_multi_comm(self):
        gloo_size = int(os.getenv('HOROVOD_SIZE', -1))
        if gloo_size != -1:
            self.skipTest("This test is specific to MPI and does not apply with Gloo controller.")

        try:
            from mpi4py import MPI
        except ImportError:
            self.skipTest("This test requires mpi4py")

        # This will be our baseline world communicator
        comm = MPI.COMM_WORLD
        # Split COMM_WORLD into subcommunicators
        subcomm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.rank % 2,
                                       key=MPI.COMM_WORLD.rank)

        # And here is our array of communicators
        comms = [comm, subcomm]

        hvd.init(comm=comms)
        size = hvd.size()

        ps = hvd.get_process_sets()
        self.assertDictEqual(ps, {0: list(range(size)),
                                  1: list(range(0, size, 2)),
                                  2: list(range(1, size, 2))})

        global_id = hvd.comm_process_set(comm)
        self.assertEqual(global_id, 0)

        split_id = hvd.comm_process_set(subcomm)
        if hvd.rank() % 2 == 0:
            self.assertEqual(split_id, 1)
        else:
            self.assertEqual(split_id, 2)

        MPI.COMM_WORLD.barrier()

        hvd.shutdown()
