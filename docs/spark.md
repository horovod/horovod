## Horovod in Spark

The `horovod.spark` package provides a convenient wrapper around Open
MPI that makes running Horovod jobs in Spark clusters easy.

In situations where training data originates from Spark this enables
tight model design loop in which data processing, model training and
model evaluation are all done in Spark.

The toy example of running a Horovod job in Spark is provided below:

```
$ pyspark
[PySpark welcome message]

>>> def fn(magic_number):
...   import horovod.torch as hvd
...   hvd.init()
...   print('Hello, rank = %d, local_rank = %d, size = %d, local_size = %d, magic_number = %d' % (hvd.rank(), hvd.local_rank(), hvd.size(), hvd.local_size(), magic_number))
...   return hvd.rank()
...
>>> import horovod.spark
>>> horovod.spark.run(fn, args=(42,))
Running 16 processes...
[Stage 0:>                                                        (0 + 16) / 16]
Hello, rank = 15, local_rank = 3, size = 16, local_size = 4, magic_number = 42
Hello, rank = 13, local_rank = 1, size = 16, local_size = 4, magic_number = 42
Hello, rank = 8, local_rank = 0, size = 16, local_size = 4, magic_number = 42
Hello, rank = 9, local_rank = 1, size = 16, local_size = 4, magic_number = 42
Hello, rank = 10, local_rank = 2, size = 16, local_size = 4, magic_number = 42
Hello, rank = 11, local_rank = 3, size = 16, local_size = 4, magic_number = 42
Hello, rank = 6, local_rank = 2, size = 16, local_size = 4, magic_number = 42
Hello, rank = 4, local_rank = 0, size = 16, local_size = 4, magic_number = 42
Hello, rank = 0, local_rank = 0, size = 16, local_size = 4, magic_number = 42
Hello, rank = 1, local_rank = 1, size = 16, local_size = 4, magic_number = 42
Hello, rank = 2, local_rank = 2, size = 16, local_size = 4, magic_number = 42
Hello, rank = 5, local_rank = 1, size = 16, local_size = 4, magic_number = 42
Hello, rank = 3, local_rank = 3, size = 16, local_size = 4, magic_number = 42
Hello, rank = 12, local_rank = 0, size = 16, local_size = 4, magic_number = 42
Hello, rank = 7, local_rank = 3, size = 16, local_size = 4, magic_number = 42
Hello, rank = 14, local_rank = 2, size = 16, local_size = 4, magic_number = 42
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
>>>
```

(TODO) E2E example with Petastorm

### Security

Horovod in Spark uses Open MPI to run the Horovod jobs in Spark, and so
it's as secure as the Open MPI implementation itself.

Since Open MPI does not use encrypted communication and is capable of
launching new processes, it's recommended to **use network level
security to isolate Horovod jobs from potential attackers**.

### Environment knobs

* `HOROVOD_SPARK_START_TIMEOUT` - sets the default timeout for Spark
tasks to spawn, register and start running the code.  If executors for
Spark tasks are scheduled on-demand and can take long time to start,
it may be useful to increase this timeout on a system level.
