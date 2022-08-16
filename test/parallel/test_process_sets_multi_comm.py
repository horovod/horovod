import os
import unittest

import horovod.tensorflow as hvd

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

        size = comm.size
        if size < 2:
            self.skipTest("This test requires multiple workers.")

        # Split COMM_WORLD into subcommunicators
        subcomm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.rank % 2,
                                       key=MPI.COMM_WORLD.rank)
        comm_clone = comm.Dup()
        subcomm_clone = subcomm.Dup()
        subcomm_effective_clone = hvd.ProcessSet(range(0, comm.size, 2))  # identified as a clone on even ranks

        # 3+ duplicates
        my_process_sets = [hvd.ProcessSet(subcomm),
                           hvd.ProcessSet(comm_clone),
                           hvd.ProcessSet(subcomm_clone),
                           subcomm_effective_clone,
                           hvd.ProcessSet([0]),
                           ]
        with self.assertRaises(ValueError):
            hvd.init(comm=comm, process_sets=my_process_sets)

        ## Internally Horovod has been initialized successfully, but we need to call hvd.init() with a valid list of
        ## process sets to proceed.

        # 2+ duplicates
        my_process_sets = [hvd.ProcessSet(subcomm),
                           hvd.ProcessSet(comm_clone),
                           subcomm_effective_clone,
                           hvd.ProcessSet([0]),
                           ]
        with self.assertRaises(ValueError):
            hvd.init(comm=comm, process_sets=my_process_sets)

        # 1+ duplicates
        my_process_sets = [hvd.ProcessSet(subcomm),
                           hvd.ProcessSet(comm_clone),
                           hvd.ProcessSet([0]),
                           ]
        with self.assertRaises(ValueError):
            hvd.init(comm=comm, process_sets=my_process_sets)

        # 1+ duplicates
        my_process_sets = [hvd.ProcessSet(subcomm),
                           subcomm_effective_clone,
                           hvd.ProcessSet([0]),
                           ]
        if hvd.size() == 2 or hvd.rank() % 2 == 0:
            with self.assertRaises(ValueError):
                hvd.init(comm=comm, process_sets=my_process_sets)
        else:
            hvd.init(comm=comm, process_sets=my_process_sets)

        # no duplicates
        if size > 2:
            my_process_sets = [hvd.ProcessSet(subcomm),
                               hvd.ProcessSet([0]),
                               ]
            hvd.init(comm=comm, process_sets=my_process_sets)
        else:
            my_process_sets = [hvd.ProcessSet(subcomm), ]
            hvd.init(comm=comm, process_sets=my_process_sets)


        self.assertEqual(hvd.global_process_set.process_set_id, 0)
        self.assertListEqual(hvd.global_process_set.ranks, list(range(size)))
        self.assertEqual(hvd.global_process_set.mpi_comm, comm)

        # Here we test some implementation details (numeric process set id values) using an internal function.
        ps = hvd.mpi_ops._basics._get_process_set_ids_and_ranks()
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

        if hvd.rank() % 2 == 0:
            self.assertEqual(my_process_sets[0].process_set_id, 1)
        else:
            self.assertEqual(my_process_sets[0].process_set_id, 2)

        # If another process initiates shutdown while this process is still processing _get_process_set_ids_and_ranks(),
        # a race condition may be triggered. Avoid with a barrier.
        MPI.COMM_WORLD.barrier()

        hvd.shutdown()
