"""Reduction operations for combining tensors. Pass one of the following to
DistributedOptimizer to configure how gradients from different processes
are combined.

Average Compute a sum and then divide by hvd.size()
Sum     Compute a sum.
Adasum  Compute a sum when inputs are orthogonal, average when inputs are
        non-orthogonal, and smoothly interpolate in between.
"""

# Please keep these values in sync with ReduceOp in horovod/common/operations.h
Average = 0
Sum = 1
Adasum = 2

def handle_average_backwards_compatibility(op, average):
    if op != None:
        if average != None:
            raise ValueError('The op parameter supersedes average. Please provide only one of them.')
        return op
    elif average != None:
        return Average if average else Sum
    else:
        return Average