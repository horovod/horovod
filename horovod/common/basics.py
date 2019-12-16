# Copyright (C) 2019 Uber Technologies, Inc.
# Modifications copyright Microsoft
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import atexit
import ctypes

from horovod.common import util as util


class HorovodBasics(object):
    """Wrapper class for the basic Horovod API."""

    def __init__(self, pkg_path, *args):
        full_path = util.get_extension_full_path(pkg_path, *args)
        self.MPI_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

        self.Average = self.MPI_LIB_CTYPES.horovod_reduce_op_average()
        self.Sum = self.MPI_LIB_CTYPES.horovod_reduce_op_sum()
        self.Adasum = self.MPI_LIB_CTYPES.horovod_reduce_op_adasum()

    def init(self, comm=None):
        """A function that initializes Horovod.

        Args:
          comm: List specifying ranks for the communicator, relative to the MPI_COMM_WORLD
            communicator OR the MPI communicator to use. Given communicator will be duplicated.
            If None, Horovod will use MPI_COMM_WORLD Communicator.
        """
        if comm is None:
            comm = []

        atexit.register(self.shutdown)

        if not isinstance(comm, list):
            mpi_built = self.MPI_LIB_CTYPES.horovod_mpi_built()
            if not bool(mpi_built):
                raise ValueError(
                    "Horovod has not been built with MPI support. Ensure MPI is installed and "
                    "reinstall Horovod with HOROVOD_WITH_MPI=1 to debug the build error.")

            from mpi4py import MPI
            if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
                MPI_Comm = ctypes.c_int
            else:
                MPI_Comm = ctypes.c_void_p
                self.MPI_LIB_CTYPES.horovod_init_comm.argtypes = [MPI_Comm]

            comm_obj = MPI_Comm.from_address(MPI._addressof(comm))
            self.MPI_LIB_CTYPES.horovod_init_comm(comm_obj)
        else:
            comm_size = len(comm)
            self.MPI_LIB_CTYPES.horovod_init(
                (ctypes.c_int * comm_size)(*comm), ctypes.c_int(comm_size))

    def shutdown(self):
        """A function that shuts Horovod down."""
        self.MPI_LIB_CTYPES.horovod_shutdown()

    def size(self):
        """A function that returns the number of Horovod processes.

        Returns:
          An integer scalar containing the number of Horovod processes.
        """
        size = self.MPI_LIB_CTYPES.horovod_size()
        if size == -1:
            raise ValueError(
                'Horovod has not been initialized; use hvd.init().')
        return size

    def local_size(self):
        """A function that returns the number of Horovod processes within the
        node the current process is running on.

        Returns:
          An integer scalar containing the number of local Horovod processes.
        """
        local_size = self.MPI_LIB_CTYPES.horovod_local_size()
        if local_size == -1:
            raise ValueError(
                'Horovod has not been initialized; use hvd.init().')
        return local_size

    def rank(self):
        """A function that returns the Horovod rank of the calling process.

        Returns:
          An integer scalar with the Horovod rank of the calling process.
        """
        rank = self.MPI_LIB_CTYPES.horovod_rank()
        if rank == -1:
            raise ValueError(
                'Horovod has not been initialized; use hvd.init().')
        return rank

    def local_rank(self):
        """A function that returns the local Horovod rank of the calling process, within the
        node that it is running on. For example, if there are seven processes running
        on a node, their local ranks will be zero through six, inclusive.

        Returns:
          An integer scalar with the local Horovod rank of the calling process.
        """
        local_rank = self.MPI_LIB_CTYPES.horovod_local_rank()
        if local_rank == -1:
            raise ValueError(
                'Horovod has not been initialized; use hvd.init().')
        return local_rank

    def is_homogeneous(self):
        """Returns True if the cluster is homogeneous.

        Returns:
          A boolean value indicating whether every node in the cluster has same number of ranks.
        """
        is_homogeneous = self.MPI_LIB_CTYPES.horovod_is_homogeneous()
        return bool(is_homogeneous)

    def mpi_threads_supported(self):
        """A function that returns a flag indicating whether MPI multi-threading is supported.

        If MPI multi-threading is supported, users may mix and match Horovod usage with other
        MPI libraries, such as `mpi4py`.

        Returns:
          A boolean value indicating whether MPI multi-threading is supported.
        """
        mpi_enabled = self.MPI_LIB_CTYPES.horovod_mpi_enabled()
        if not bool(mpi_enabled):
            raise ValueError(
                'Horovod MPI is not enabled; Please make sure it\'s installed and enabled.')

        mpi_threads_supported = self.MPI_LIB_CTYPES.horovod_mpi_threads_supported()
        if mpi_threads_supported == -1:
            raise ValueError(
                'Horovod has not been initialized; use hvd.init().')
        return bool(mpi_threads_supported)

    def mpi_enabled(self):
        """Returns True if MPI is mode is currently enabled at runtime.

        If MPI is enabled, users can use it for controller or data transfer operations.

        Returns:
          A boolean value indicating whether MPI is enabled.
        """
        mpi_enabled = self.MPI_LIB_CTYPES.horovod_mpi_enabled()
        return bool(mpi_enabled)

    def mpi_built(self):
        """Returns True if Horovod was compiled with MPI support.

        Returns:
          A boolean value indicating whether MPI support was compiled.
        """
        return bool(self.MPI_LIB_CTYPES.horovod_mpi_built())

    def gloo_enabled(self):
        """Returns True if Gloo is mode is currently enabled at runtime.

        If Gloo is enabled, users can use it for controller or data transfer operations.

        Returns:
          A boolean value indicating whether Gloo is enabled.
        """
        gloo_enabled = self.MPI_LIB_CTYPES.horovod_gloo_enabled()
        return bool(gloo_enabled)

    def gloo_built(self):
        """Returns True if Horovod was compiled with Gloo support.

        Returns:
          A boolean value indicating whether Gloo support was compiled.
        """
        return bool(self.MPI_LIB_CTYPES.horovod_gloo_built())

    def nccl_built(self):
        """Returns True if Horovod was compiled with NCCL support.

        Returns:
          A boolean value indicating whether NCCL support was compiled.
        """
        return bool(self.MPI_LIB_CTYPES.horovod_nccl_built())

    def ddl_built(self):
        """Returns True if Horovod was compiled with DDL support.

        Returns:
          A boolean value indicating whether DDL support was compiled.
        """
        return bool(self.MPI_LIB_CTYPES.horovod_ddl_built())

    def ccl_built(self):
        """Returns True if Horovod was compiled with oneCCL support.

        Returns:
          A boolean value indicating whether oneCCL support was compiled.
        """
        return bool(self.MPI_LIB_CTYPES.horovod_ccl_built())
