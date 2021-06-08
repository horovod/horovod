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
    or hvd.add_process_set().
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
            assert _basics is not None, "process_sets.setup() must be called first"
            if not _basics.mpi_built():
                raise ValueError(
                    "Horovod has not been built with MPI support. Ensure MPI is installed and "
                    "reinstall Horovod with HOROVOD_WITH_MPI=1 to debug the build error.")
            from mpi4py import MPI
            if not isinstance(ranks_or_comm, MPI.Comm):
                raise ValueError(
                    "ProcessSet should be initialized with a list of process ranks or an mpi4py Comm object")
            self.mpi_comm = ranks_or_comm

    def invalidate(self):
        self.process_set_id = None
        self.ranks = None
        self.mpi_comm = None

    def __str__(self) -> str:
        return f"ProcessSet(process_set_id={self.process_set_id}, ranks={self.ranks}, mpi_comm={self.mpi_comm})"


global_process_set = ProcessSet([])
global_process_set.process_set_id = 0

_id_to_process_sets: Dict[int, List[ProcessSet]] = {0: [global_process_set]}


def setup(basics):
    # type: (Optional[HorovodBasics]) -> None
    """" Horovod internal, to be called after the Horovod C++ module has been loaded. """
    global _basics
    _basics = basics


def update_process_sets(process_sets: List[ProcessSet]):
    """ Horovod internal, to be called from hvd.init() """

    # Update process set objects in passed list:
    id_to_ranks_dict = _basics.get_process_set_ids_and_ranks()
    ranks_to_id_dict = {tuple(ranks): process_set_id for process_set_id, ranks in id_to_ranks_dict.items()}
    for ps in process_sets:
        if ps.ranks is not None:
            ps.process_set_id = ranks_to_id_dict[tuple(ps.ranks)]
        elif ps.mpi_comm is not None:
            ps.process_set_id = _basics.comm_process_set_id(ps.mpi_comm)
            ps.ranks = list(id_to_ranks_dict[ps.process_set_id])

    # Update process set storage _id_to_process_sets according to passed list:
    for ps in process_sets:
        list_in_storage = _id_to_process_sets.setdefault(ps.process_set_id, [])
        if ps not in list_in_storage:
            list_in_storage.append(ps)

    # Update process set storage _id_to_process_sets according to remaining entries from id_to_ranks_dict:
    for process_set_id, ranks in id_to_ranks_dict.items():
        if process_set_id not in _id_to_process_sets:
            process_set = ProcessSet(ranks)
            process_set.process_set_id = process_set_id
            _id_to_process_sets[process_set_id] = [process_set]

    # Update ranks in global process set objects
    for global_ps in _id_to_process_sets[0]:
        if global_ps.ranks != id_to_ranks_dict[0]:
            global_ps.ranks = id_to_ranks_dict[0]


def add_process_set(process_set: Union[ProcessSet, Sequence[int]]) -> ProcessSet:
    """ Add the process_set after Horovod initialization.

    Requires running with HOROVOD_DYNAMIC_PROCESS_SETS=1. If process_set is a list of ranks, a new ProcessSet object
    will be initialized. In any case the added ProcessSet instance will be returned.
    """
    assert _basics is not None
    if not isinstance(process_set, ProcessSet):
        process_set = ProcessSet(process_set)
    if process_set.ranks is None and process_set.mpi_comm is not None:
        raise NotImplementedError(
            "Dynamically adding process sets defined by an MPI communicator is not implemented. "
            "Please build the process set via a list of ranks.")
    assert process_set.ranks is not None

    process_set_id = _basics.add_process_set_impl(process_set.ranks)
    process_set.process_set_id = process_set_id
    _id_to_process_sets.setdefault(process_set_id, []).append(process_set)
    return process_set


def remove_process_set(process_set: ProcessSet) -> bool:
    """ Attempt to remove process set and return whether this attempt is successful.

    Requires running with HOROVOD_DYNAMIC_PROCESS_SETS=1. If removal is succesfull, we will invalidate the process_set
    object and all known equivalent ProcessSet objects.
    """
    assert _basics is not None
    assert process_set.process_set_id is not None
    process_set_id = process_set.process_set_id

    if process_set_id == 0:
        # will not remove the global process set
        return False

    if process_set.process_set_id in _id_to_process_sets:
        for some_process_set in _id_to_process_sets.pop(process_set_id):
            some_process_set.invalidate()

    returned_id = _basics.remove_process_set_impl(process_set_id)
    return returned_id is not None


def process_set_by_id(process_set_id: int) -> ProcessSet:
    """ Return a registered process set object with the given id. """
    if not process_set_id in _id_to_process_sets:
        raise ValueError(f"No known process set with id {process_set_id}")
    return _id_to_process_sets[process_set_id][0]


def process_sets() -> List[ProcessSet]:
    """ Return a list containing registered process set objects for each registered id. """
    return [_id_to_process_sets[process_set_id][0]
            for process_set_id in sorted(_id_to_process_sets.keys())]
