.. inclusion-marker-start-do-not-remove

Horovod on Spark
================

The ``horovod.spark`` package provides a convenient wrapper around Horovod that makes running distributed training
jobs in Spark clusters easy.

In situations where training data originates from Spark, this enables
a tight model design loop in which data processing, model training, and
model evaluation are all done in Spark.

We provide two APIs for running Horovod on Spark: a high level **Estimator** API and a lower level **Run** API. Both
use the same underlying mechanism to launch Horovod on Spark executors, but the Estimator API abstracts the data
processing (from Spark DataFrames to deep learning datasets), model training loop, model checkpointing, metrics
collection, and distributed training.

We recommend using Horovod Spark Estimators if you:

* Are using Keras (``tf.keras`` or ``keras``) or PyTorch for training.
* Want to train directly on a Spark DataFrame from ``pyspark``.
* Are using a standard gradient descent optimization process as your training loop.

If for whatever reason the Estimator API does not meet your needs, the Run API offers more fine-grained control.

Installation
------------

When installing Horovod for usage with Spark, use the extra ``[spark]`` to install all Spark dependencies as well:

.. code-block:: bash

    $ ... pip install horovod[spark]

Note that Horovod Spark Estimators require the following:

*  ``horovod >= 0.19.0``
*  ``pyspark >= 2.3.2``

Not included in the list of dependencies by ``[spark]`` are deep learning frameworks (TensorFlow or PyTorch).
Horovod Spark Estimators additionally require at least one of these combinations:

*  ``tensorflow-gpu >= 1.12.0`` or ``tensorflow >= 1.12.0`` (for ``KerasEstimator``)
*  ``torch >= 1.0.0`` and ``tensorboard >= 1.14.0`` (for ``TorchEstimator``)
*  ``torch >= 1.4.0`` and ``pytorch_lightning >= 1.2.9`` (for ``LightningEstimator``)


Horovod Spark Estimators
~~~~~~~~~~~~~~~~~~~~~~~~
Horovod Spark Estimators allow you to train your deep neural network directly on an existing Spark DataFrame,
leveraging Horovod’s ability to scale across multiple workers, without any specialized code for distributed training:

.. code-block:: python

    from tensorflow import keras
    import tensorflow as tf
    import horovod.spark.keras as hvd

    model = keras.models.Sequential()
        .add(keras.layers.Dense(8, input_dim=2))
        .add(keras.layers.Activation('tanh'))
        .add(keras.layers.Dense(1))
        .add(keras.layers.Activation('sigmoid'))

    # NOTE: unscaled learning rate
    optimizer = keras.optimizers.SGD(lr=0.1)
    loss = 'binary_crossentropy'

    store = HDFSStore('/user/username/experiments')
    keras_estimator = hvd.KerasEstimator(
        num_proc=4,
        store=store,
        model=model,
        optimizer=optimizer,
        loss=loss,
        feature_cols=['features'],
        label_cols=['y'],
        batch_size=32,
        epochs=10)


    keras_model = keras_estimator.fit(train_df) \
        .setOutputCols(['predict'])
    predict_df = keras_model.transform(test_df)

The Estimator hides the complexity of gluing Spark DataFrames to a deep learning training script, reading data into a
format interpretable by the training framework, and distributing the training using Horovod.  The user only needs to
provide a Keras or PyTorch model, and the Estimator will do the work of fitting it to the DataFrame.

After training, the Estimator returns a Transformer representation of the trained model.  The model transformer can
be used like any Spark ML transformer to make predictions on an input DataFrame, writing them as new columns in the
output DataFrame.

Estimators can be used to track experiment history through model checkpointing, hot-start retraining, and metric
logging (for Tensorboard) using the Estimator ``Store`` abstraction.  Stores are used for persisting all training
artifacts including intermediate representations of the training data.  Horovod natively supports stores for HDFS
and local filesystems.

End-to-end example
------------------
`keras_spark_rossmann_estimator.py script <../examples/spark/keras/keras_spark_rossmann_estimator.py>`__ provides
an example of end-to-end data preparation and training of a model for the
`Rossmann Store Sales <https://www.kaggle.com/c/rossmann-store-sales>`__ Kaggle
competition. It is inspired by an article `An Introduction to Deep Learning for Tabular Data <https://www.fast.ai/2018/04/29/categorical-embeddings/>`__
and leverages the code of the notebook referenced in the article. The example is split into three parts:

#. The first part performs complicated data preprocessing over an initial set of CSV files provided by the competition and gathered by the community.
#. The second part defines a Keras model and performs a distributed training of the model using Horovod on Spark.
#. The third part performs prediction using the best model and creates a submission file.

To run the example, be sure to install Horovod with ``[spark]``, then:

.. code-block:: bash

    $ wget https://raw.githubusercontent.com/horovod/horovod/master/examples/spark/keras/keras_spark_rossmann_estimator.py
    $ wget http://files.fast.ai/part2/lesson14/rossmann.tgz
    $ tar zxvf rossmann.tgz
    $ python keras_spark_rossmann_estimator.py

For pytorch, you can check `pytorch_lightning_spark_mnist.py script <../examples/spark/pytorch/pytorch_lightning_spark_mnist.py>`__ for how to use use lightning estimator with horovod backend to train mnist model on spark.

Training on existing Parquet datasets
-------------------------------------

If your data is already in the Parquet format and you wish to train on it with Horovod Spark Estimators, you
can do so without needing to reprocess the data in Spark. Using `Estimator.fit_on_parquet()`, you can train directly
on an existing Parquet dataset:

.. code-block:: python

    store = HDFSStore(train_path='/user/username/training_dataset', val_path='/user/username/val_dataset')
    keras_estimator = hvd.KerasEstimator(
        num_proc=4,
        store=store,
        model=model,
        optimizer=optimizer,
        loss=loss,
        feature_cols=['features'],
        label_cols=['y'],
        batch_size=32,
        epochs=10)

    keras_model = keras_estimator.fit_on_parquet()

The resulting ``keras_model`` can then be used the same way as any Spark Transformer, or you can extract the underlying
Keras model and use it outside of Spark:

.. code-block:: python

    model = keras_model.getModel()
    pred = model.predict([np.ones([1, 2], dtype=np.float32)])

This approach will work on datasets created using ``horovod.spark.common.util.prepare_data``. It will also work with
any Parquet file that contains no Spark user-defined data types (like ``DenseVector`` or ``SparseVector``).  It's
recommended to use ``prepare_data`` to ensure the data is properly prepared for training even if you have an existing
dataset in Parquet format.  Using ``prepare_data`` allows you to properly partition the dataset for the number of
training processes you intend to use, as well as compress large sparse data columns:

.. code-block:: python

    store = HDFSStore(train_path='/user/username/training_dataset', val_path='/user/username/val_dataset')
    with util.prepare_data(num_processes=4,
                           store=store,
                           df=df,
                           feature_columns=['features'],
                           label_columns=['y'],
                           validation=0.1,
                           compress_sparse=True):
        keras_estimator = hvd.KerasEstimator(
            num_proc=4,
            store=store,
            model=model,
            optimizer=optimizer,
            loss=loss,
            feature_cols=['features'],
            label_cols=['y'],
            batch_size=32,
            epochs=10)

        keras_model = keras_estimator.fit_on_parquet()

Once the data has been prepared, you can reuse it in future Spark applications without needing to call
``util.prepare_data`` again.

Horovod Spark Run
~~~~~~~~~~~~~~~~~
You can also use Horovod on Spark to run the same code you would within an ordinary training script using any
framework supported by Horovod.  To do so, simply write your training logic within a function, then use
``horovod.spark.run`` to execute the function in parallel with MPI on top of Spark.

Because Horovod on Spark uses ``cloudpickle`` to send the training function to workers for execution, you can capture
local variables from your training script or notebook within the training function, similar to using a user-defined
function in PySpark.

A toy example of running a Horovod job in Spark is provided below:

.. code-block:: bash

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

A more complete example can be found in `keras_spark_rossmann_run.py <../examples/spark/keras/keras_spark_rossmann_run.py>`__, which
shows how you can use the low level ``horovod.spark.run`` API to train a model end-to-end in the following steps:

.. code-block:: bash

    $ wget https://raw.githubusercontent.com/horovod/horovod/master/examples/spark/keras/keras_spark_rossmann_run.py
    $ wget http://files.fast.ai/part2/lesson14/rossmann.tgz
    $ tar zxvf rossmann.tgz
    $ python keras_spark_rossmann_run.py


Spark cluster setup
~~~~~~~~~~~~~~~~~~~
As deep learning workloads tend to have very different resource requirements
from typical data processing workloads, there are certain considerations
for DL Spark cluster setup.

GPU training
------------
For GPU training, one approach is to set up a separate GPU Spark cluster
and configure each executor with ``# of CPU cores`` = ``# of GPUs``. This can
be accomplished in standalone mode as follows:

.. code-block:: bash

    $ echo "export SPARK_WORKER_CORES=<# of GPUs>" >> /path/to/spark/conf/spark-env.sh
    $ /path/to/spark/sbin/start-all.sh


This approach turns the ``spark.task.cpus`` setting to control # of GPUs
requested per process (defaults to 1).

The ongoing `SPARK-24615 <https://issues.apache.org/jira/browse/SPARK-24615>`__ effort aims to
introduce GPU-aware resource scheduling in future versions of Spark.

CPU training
------------
For CPU training, one approach is to specify the ``spark.task.cpus`` setting
during the training session creation:

.. code-block:: python

    conf = SparkConf().setAppName('training') \
        .setMaster('spark://training-cluster:7077') \
        .set('spark.task.cpus', '16')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()


This approach allows you to reuse the same Spark cluster for data preparation
and training.

Security
--------
Horovod on Spark uses Open MPI to run the Horovod jobs in Spark, so
it's as secure as the Open MPI implementation itself.

Since Open MPI does not use encrypted communication and is capable of
launching new processes, it's recommended to **use network level
security to isolate Horovod jobs from potential attackers**.

Environment knobs
-----------------
* ``HOROVOD_SPARK_START_TIMEOUT`` - sets the default timeout for Spark tasks to spawn, register, and start running the code.  If executors for Spark tasks are scheduled on-demand and can take a long time to start, it may be useful to increase this timeout on a system level.

Horovod on Databricks
------------------------------
To run Horovod in Spark on Databricks, create a Store instance with a DBFS path in one of the following patterns:

* ``/dbfs/...``
* ``dbfs:/...``
* ``file:///dbfs/...``

.. code-block:: python

    store = Store.create(dbfs_path)
    # or explicitly using DBFSLocalStore
    store = DBFSLocalStore(dbfs_path)

The `DBFSLocalStore` uses Databricks File System (DBFS) local file APIs
(`AWS <https://docs.databricks.com/data/databricks-file-system.html#local-file-apis>`__ |
`Azure <https://docs.microsoft.com/en-us/azure/databricks/data/databricks-file-system#--local-file-apis>`__)
as a store of intermediate data and training artifacts.

Databricks pre-configures GPU-aware scheduling on Databricks Runtime 7.0 ML GPU and above. See GPU scheduling instructions
(`AWS <https://docs.databricks.com/clusters/gpu.html#gpu-scheduling-1>`__ |
`Azure <https://docs.microsoft.com/en-us/azure/databricks/clusters/gpu#gpu-scheduling>`__)
for details.

With the Estimator API, horovod will launch ``# of tasks on each worker = # of GPUs on each worker``, and each task will
pin GPU to the assigned GPU from spark.

With the Run API, the function ``get_available_devices()`` from ``horovod.spark.task`` will return a list of assigned GPUs
for the spark task from which ``get_available_devices()`` is called.
See `keras_spark3_rossmann.py <../examples/spark/keras/keras_spark3_rossmann.py>`__ for an example of using
``get_available_devices()`` with the Run API.

.. inclusion-marker-end-do-not-remove
