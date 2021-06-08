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
            self.skipTest("This test requires mpi4py.")

        # This will be our baseline world communicator
        comm = MPI.COMM_WORLD
        # Split COMM_WORLD into subcommunicators
        subcomm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.rank % 2,
                                       key=MPI.COMM_WORLD.rank)
        comm_clone = comm.Dup()
        subcomm_clone = subcomm.Dup()

        # No distinct process sets will be built from the clones.
        hvd.init(comm=comm, process_sets=[hvd.ProcessSet(subcomm),
                                          hvd.ProcessSet(range(0, comm.size, 2)), # another clone
                                          hvd.ProcessSet(comm_clone),
                                          hvd.ProcessSet(subcomm_clone),
                                          hvd.ProcessSet([0]),
                                          ])
        size = hvd.size()
        if size < 2:
            self.skipTest("This test requires multiple workers.")

        ps = hvd.get_process_set_ids_and_ranks()
        if size > 2:
            self.assertDictEqual(ps, {0: list(range(size)),
                                      1: list(range(0, size, 2)),
                                      2: list(range(1, size, 2)),
                                      3: [0],
                                      })
        else:
            self.assertDictEqual(ps, {0: list(range(size)),
                                      1: list(range(0, size, 2)),
                                      2: list(range(1, size, 2)),
                                      })


        global_id = hvd.comm_process_set_id(comm)
        self.assertEqual(global_id, 0)

        split_id = hvd.comm_process_set_id(subcomm)
        split_dup_id = hvd.comm_process_set_id(subcomm_clone)
        if hvd.rank() % 2 == 0:
            self.assertEqual(split_id, 1)
            self.assertEqual(split_dup_id, 1)
        else:
            self.assertEqual(split_id, 2)
            self.assertEqual(split_dup_id, 2)

        global_dup_id = hvd.comm_process_set_id(comm_clone)
        self.assertEqual(global_dup_id, 0)

        MPI.COMM_WORLD.barrier()

        hvd.shutdown()
