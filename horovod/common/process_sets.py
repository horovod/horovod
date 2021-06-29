from typing import *
# for type annotations, importing mpi4py has dangerous side effects
try:
    from horovod.common.basics import HorovodBasics
except ImportError:
    class HorovodBasics:
        ...
class MPI:
    class Comm:
        ...

from horovod.common.util import is_iterable


_basics = None  # type: Optional[HorovodBasics]


class ProcessSet:
    """ Representation of a set of Horovod processes that will run collective operations together

    Initialize a ProcessSet with a list of process ranks or an MPI communicator. Then pass this instance to hvd.init()
    or hvd.add_process_set(). If a valid process set has been initialized, process_set_id will be set to a numeric
    value.
    """
    process_set_id = None
    ranks = None
    mpi_comm = None

    def __init__(self, ranks_or_comm: Union[Sequence[int], MPI.Comm]):
        if is_iterable(ranks_or_comm):
            ranks_or_comm = sorted(ranks_or_comm)
            if any(not isinstance(rk, int) for rk in ranks_or_comm):
                raise ValueError(
                    "ProcessSet should be initialized with a list of process ranks or an mpi4py Comm object")
            self.ranks = ranks_or_comm
        else:
            assert _basics is not None, "process_sets._setup() must be called first"
            if not _basics.mpi_built():
                raise ValueError(
                    "Apparently you tried to build a ProcessSet from an MPI communicator, "
                    "but Horovod has not been built with MPI support. Ensure MPI is installed and "
                    "reinstall Horovod with HOROVOD_WITH_MPI=1 to debug the build error.")
            from mpi4py import MPI
            if not isinstance(ranks_or_comm, MPI.Comm):
                raise ValueError(
                    "ProcessSet should be initialized with a list of process ranks or an mpi4py Comm object")
            self.mpi_comm = ranks_or_comm

    def _invalidate(self):
        self.process_set_id = None

    def size(self) -> Optional[int]:
        """ Return size of the process set or None if not initialized. """
        if self.process_set_id is None:
            return None
        return _basics._process_set_size(self.process_set_id)

    def rank(self) -> Optional[int]:
        """ Return rank relative to this process set or None if not initialized.

        This is useful, e.g., to process the result of hvd.allgather().

        Please note that, even with process sets, Horovod operations like hvd.broadcast() are not parameterized by this
        relative rank, but by the global rank as obtained from hvd.rank().
        """
        if self.process_set_id is None:
            return None
        return _basics._process_set_rank(self.process_set_id)


    def included(self) -> Optional[bool]:
        """ Return whether the current process is part of this process set or None if not initialized. """
        if self.ranks is None:
            return None
        return _basics.rank() in self.ranks

    def __str__(self) -> str:
        return f"ProcessSet(process_set_id={self.process_set_id}, ranks={self.ranks}, mpi_comm={self.mpi_comm})"


def _temp_process_set_object(process_set_id: int) -> ProcessSet:
    """ For Horovod-internal usage where we don't have a ProcessSet instance at hand but know a valid process_set_id.
    """
    ps = ProcessSet.__new__(ProcessSet)
    ps.process_set_id = process_set_id
    return ps

global_process_set = ProcessSet([])
global_process_set.process_set_id = 0


def _setup(basics):
    # type: (Optional[HorovodBasics]) -> None
    """" Horovod internal, to be called after the Horovod C++ module has been loaded. """
    global _basics
    _basics = basics


def _init_process_sets(process_set_list: List[ProcessSet]):
    """ Update process_set_id and ranks entries of all passed process set objects and invalidate any clones.

    Horovod internal, to be called from hvd.init(). """
    # Update process set objects in passed list:
    ids_seen_in_process_set_list = {0}  # global_process_set is not in list
    id_to_ranks_dict = _basics._get_process_set_ids_and_ranks()
    ranks_to_id_dict = {tuple(ranks): process_set_id for process_set_id, ranks in id_to_ranks_dict.items()}
    for ps in process_set_list:
        if ps.ranks is not None:
            ps.process_set_id = ranks_to_id_dict[tuple(ps.ranks)]
        elif ps.mpi_comm is not None:
            ps.process_set_id = _basics._comm_process_set_id(ps.mpi_comm)
            ps.ranks = list(id_to_ranks_dict[ps.process_set_id])
        if ps.process_set_id in ids_seen_in_process_set_list:
            ps._invalidate()
        else:
            ids_seen_in_process_set_list.add(ps.process_set_id)

    # Update ranks in global process set object
    if global_process_set.ranks != id_to_ranks_dict[0]:
        global_process_set.ranks = id_to_ranks_dict[0]


def add_process_set(process_set: Union[ProcessSet, Sequence[int]]) -> ProcessSet:
    """ Add a new process_set after Horovod initialization and return it.

    Requires running with HOROVOD_DYNAMIC_PROCESS_SETS=1. No process set containing the same ranks may exist already.
    The returned process set will be fully initialized.
    """
    assert _basics is not None
    if not isinstance(process_set, ProcessSet):
        process_set = ProcessSet(process_set)
    if process_set.ranks is None and process_set.mpi_comm is not None:
        raise NotImplementedError(
            "Dynamically adding process sets defined by an MPI communicator is not implemented. "
            "Please build the process set via a list of ranks.")
    assert process_set.ranks is not None

    process_set_id = _basics._add_process_set_impl(process_set.ranks)
    if process_set_id is None:
        raise ValueError(f"Attempted to add a duplicate process set: {process_set}")
    process_set.process_set_id = process_set_id
    return process_set


def remove_process_set(process_set: ProcessSet) -> bool:
    """ Attempt to remove process set and return whether this attempt is successful.

    Requires running with HOROVOD_DYNAMIC_PROCESS_SETS=1. If removal is successful, we will invalidate the process_set
    object.
    """
    assert _basics is not None
    process_set_id = process_set.process_set_id

    if process_set_id is None:
        # process set has not been initialized
        return False
    if process_set_id == 0:
        # will not remove the global process set
        return False

    process_set._invalidate()
    returned_id = _basics._remove_process_set_impl(process_set_id)
    return returned_id is not None
