## Horovod in Spark

The `horovod.spark` package provides a convenient wrapper around Open
MPI that makes running Horovod jobs in Spark clusters easy.

In situations where training data originates from Spark, this enables
a tight model design loop in which data processing, model training, and
model evaluation are all done in Spark.

A toy example of running a Horovod job in Spark is provided below:

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

### End-to-end example

[keras_spark_rossmann.py script](../examples/keras_spark_rossmann.py) provides
an example of end-to-end data preparation and training of a model for the
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle
competition.

It is inspired by an article [An Introduction to Deep Learning for Tabular Data](https://www.fast.ai/2018/04/29/categorical-embeddings/)
and leverages the code of the notebook referenced in the article.

The example is split into three parts:
1. The first part performs complicated data preprocessing over an initial set
of CSV files provided by the competition and gathered by the community.
2. The second part defines a Keras model and performs a distributed training
of the model using Horovod in Spark.
3. The third part performs prediction using the best model and creates
a submission file.

To run the example, please install the following dependencies:
* `pyspark`
* `petastorm >= 0.5.1`
* `h5py >= 2.9.0`
* `tensorflow-gpu >= 1.12.0` (or `tensorflow >= 1.12.0`)
* `horovod >= 0.15.3`

Run the example:
```bash
$ wget https://raw.githubusercontent.com/uber/horovod/master/examples/keras_spark_rossmann.py
$ wget http://files.fast.ai/part2/lesson14/rossmann.tgz
$ tar zxvf rossmann.tgz
$ python keras_spark_rossmann.py
```

### Spark cluster setup

As deep learning workloads tend to have very different resource requirements
from typical data processing workloads, there are certain considerations
for DL Spark cluster setup.

#### GPU training

For GPU training, one approach is to set up a separate GPU Spark cluster
and configure each executor with `# of CPU cores` = `# of GPUs`. This can
be accomplished in standalone mode as follows:
```bash
$ echo "export SPARK_WORKER_CORES=<# of GPUs>" >> /path/to/spark/conf/spark-env.sh
$ /path/to/spark/sbin/start-all.sh
```

This approach turns the `spark.task.cpus` setting to control # of GPUs
requested per process (defaults to 1).

The ongoing [SPARK-24615](https://issues.apache.org/jira/browse/SPARK-24615) effort aims to
introduce GPU-aware resource scheduling in future versions of Spark.

#### CPU training

For CPU training, one approach is to specify the `spark.task.cpus` setting
during the training session creation:
```python
conf = SparkConf().setAppName('training') \
    .setMaster('spark://training-cluster:7077') \
    .set('spark.task.cpus', '16')
spark = SparkSession.builder.config(conf=conf).getOrCreate()
```

This approach allows you to reuse the same Spark cluster for data preparation
and training.

### Security

Horovod in Spark uses Open MPI to run the Horovod jobs in Spark, so
it's as secure as the Open MPI implementation itself.

Since Open MPI does not use encrypted communication and is capable of
launching new processes, it's recommended to **use network level
security to isolate Horovod jobs from potential attackers**.

### Environment knobs

* `HOROVOD_SPARK_START_TIMEOUT` - sets the default timeout for Spark
tasks to spawn, register, and start running the code.  If executors for
Spark tasks are scheduled on-demand and can take a long time to start,
it may be useful to increase this timeout on a system level.
