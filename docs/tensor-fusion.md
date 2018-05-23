## Tensor Fusion

One of the unique things about Horovod is its ability to interleave communication and computation coupled with the ability
to batch small *allreduce* operations, which results in improved performance. We call this batching feature Tensor Fusion.

Tensor Fusion works by attempting to combine all the tensors that are ready to be reduced at given moment of time into
one reduction operation. The algorithm of Tensor Fusion is as follows:

1. Determine which tensors are ready to be reduced. Select first few tensors that fit in `HOROVOD_FUSION_THRESHOLD` 
 bytes and have the same data type.
2. Allocate fusion buffer of size `HOROVOD_FUSION_THRESHOLD` if it was not allocated before. Default fusion buffer size 
 is 64 MB.
3. Copy data of selected tensors into the fusion buffer.
4. Execute the *allreduce* operation on the fusion buffer.
5. Copy data from the fusion buffer into the output tensors.
6. Repeat until there are no more tensors to reduce in this cycle.

The fusion buffer size can be tweaked using the `HOROVOD_FUSION_THRESHOLD` environment variable:

```bash
$ HOROVOD_FUSION_THRESHOLD=33554432 mpirun -np 4 -x HOROVOD_FUSION_THRESHOLD python train.py
```

Setting the `HOROVOD_FUSION_THRESHOLD` environment variable to zero disables Tensor Fusion:

```bash
$ HOROVOD_FUSION_THRESHOLD=0 mpirun -np 4 -x HOROVOD_FUSION_THRESHOLD python train.py
```

You can tweak time between cycles (defined in milliseconds) using the `HOROVOD_CYCLE_TIME` environment variable:

```bash
$ HOROVOD_CYCLE_TIME=3.5 mpirun -np 4 -x HOROVOD_FUSION_THRESHOLD python train.py
```
