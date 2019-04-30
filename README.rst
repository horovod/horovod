.. raw:: html

    <img src="docs/images/logo-120.png" width="120px">
    <br/>

Petastorm
=========

.. image:: https://travis-ci.com/uber/petastorm.svg?branch=master
   :target: https://travis-ci.com/uber/petastorm
   :alt: Build Status (Travis CI)

.. image:: https://codecov.io/gh/uber/petastorm/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/uber/petastorm/branch/master
   :alt: Code coverage

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :alt: License

.. image:: https://badge.fury.io/py/petastorm.svg
   :target: https://pypi.org/project/petastorm
   :alt: Latest Version

.. inclusion-marker-start-do-not-remove

.. contents::


Petastorm is an open source data access library developed at Uber ATG. This library enables single machine or
distributed training and evaluation of deep learning models directly from datasets in Apache Parquet
format. Petastorm supports popular Python-based machine learning (ML) frameworks such as
`Tensorflow <http://www.tensorflow.org/>`_, `PyTorch <https://pytorch.org/>`_, and
`PySpark <http://spark.apache.org/docs/latest/api/python/pyspark.html>`_. It can also be used from pure Python code.

Documentation web site: `<https://petastorm.readthedocs.io>`_



Installation
------------

.. code-block:: bash

    pip install petastorm


There are several extra dependencies that are defined by the ``petastorm`` package that are not installed automatically.
The extras are: ``tf``, ``tf_gpu``, ``torch``, ``opencv``, ``docs``, ``test``.

For example to trigger installation of GPU version of tensorflow and opencv, use the following pip command:

.. code-block:: bash

    pip install petastorm[opencv,tf_gpu]



Generating a dataset
--------------------

A dataset created using Petastorm is stored in `Apache Parquet <https://parquet.apache.org/>`_ format.
On top of a Parquet schema, petastorm also stores higher-level schema information that makes multidimensional arrays into a native part of a petastorm dataset.

Petastorm supports extensible data codecs. These enable a user to use one of the standard data compressions (jpeg, png) or implement her own.

Generating a dataset is done using PySpark.
PySpark natively supports Parquet format, making it easy to run on a single machine or on a Spark compute cluster.
Here is a minimalistic example writing out a table with some random data.


.. code-block:: python

    HelloWorldSchema = Unischema('HelloWorldSchema', [
       UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
       UnischemaField('image1', np.uint8, (128, 256, 3), CompressedImageCodec('png'), False),
       UnischemaField('other_data', np.uint8, (None, 128, 30, None), NdarrayCodec(), False),
    ])


    def row_generator(x):
       """Returns a single entry in the generated dataset. Return a bunch of random values as an example."""
       return {'id': x,
               'image1': np.random.randint(0, 255, dtype=np.uint8, size=(128, 256, 3)),
               'other_data': np.random.randint(0, 255, dtype=np.uint8, size=(4, 128, 30, 3))}


    def generate_hello_world_dataset(output_url='file:///tmp/hello_world_dataset'):
       rows_count = 10
       rowgroup_size_mb = 256

       spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
       sc = spark.sparkContext

       # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
       # well as save petastorm specific metadata
       with materialize_dataset(spark, output_url, HelloWorldSchema, rowgroup_size_mb):

           rows_rdd = sc.parallelize(range(rows_count))\
               .map(row_generator)\
               .map(lambda x: dict_to_spark_row(HelloWorldSchema, x))

           spark.createDataFrame(rows_rdd, HelloWorldSchema.as_spark_schema()) \
               .coalesce(10) \
               .write \
               .mode('overwrite') \
               .parquet(output_url)

- ``HelloWorldSchema`` is an instance of a ``Unischema`` object.
  ``Unischema`` is capable of rendering types of its fields into different
  framework specific formats, such as: Spark ``StructType``, Tensorflow
  ``tf.DType`` and numpy ``numpy.dtype``.
- To define a dataset field, you need to specify a ``type``, ``shape``, a
  ``codec`` instance and whether the field is nullable for each field of the
  ``Unischema``.
- We use PySpark for writing output Parquet files. In this example, we launch
  PySpark on a local box (``.master('local[2]')``). Of course for a larger
  scale dataset generation we would need a real compute cluster.
- We wrap spark dataset generation code with the ``materialize_dataset``
  context manager.  The context manager is responsible for configuring row
  group size at the beginning and write out petastorm specific metadata at the
  end.
- The row generating code is expected to return a Python dictionary indexed by
  a field name. We use ``row_generator`` function for that.
- ``dict_to_spark_row`` converts the dictionary into a ``pyspark.Row``
  object while ensuring schema ``HelloWorldSchema`` compliance (shape,
  type and is-nullable condition are tested).
- Once we have a ``pyspark.DataFrame`` we write it out to a parquet
  storage. The parquet schema is automatically derived from
  ``HelloWorldSchema``.

Plain Python API
----------------
The ``petastorm.reader.Reader`` class is the main entry point for user
code that accesses the data from an ML framework such as Tensorflow or Pytorch.
The reader has multiple features such as:

- Selective column readout
- Multiple parallelism strategies: thread, process, single-threaded (for debug)
- N-grams readout support
- Row filtering (row predicates)
- Shuffling
- Partitioning for multi-GPU training
- Local caching

Reading a dataset is simple using the ``petastorm.reader.Reader`` class which can be created using the
``petastorm.make_reader`` factory method:

.. code-block:: python

   from petastorm import make_reader

    with make_reader('hdfs://myhadoop/some_dataset') as reader:
       for row in reader:
           print(row)

``hdfs://...`` and ``file://...`` are supported URL protocols.

Once a ``Reader`` is instantiated, you can use it as an iterator.

Tensorflow API
--------------

To hookup the reader into a tensorflow graph, you can use the ``tf_tensors``
function:

.. code-block:: python

    with make_reader('file:///some/localpath/a_dataset') as reader:
       row_tensors = tf_tensors(reader)
       with tf.Session() as session:
           for _ in range(3):
               print(session.run(row_tensors))

Alternatively, you can use new ``tf.data.Dataset`` API;

.. code-block:: python

    with make_reader('file:///some/localpath/a_dataset') as reader:
        dataset = make_petastorm_dataset(reader)
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            sample = sess.run(tensor)
            print(sample.id)

Pytorch API
-----------

As illustrated in
`pytorch_example.py <https://github.com/uber/petastorm/blob/master/examples/mnist/pytorch_example.py>`_,
reading a petastorm dataset from pytorch
can be done via the adapter class ``petastorm.pytorch.DataLoader``,
which allows custom pytorch collating function and transforms to be supplied.

Be sure you have ``torch`` and ``torchvision`` installed:

.. code-block:: bash

    pip install torchvision

The minimalist example below assumes the definition of a ``Net`` class and
``train`` and ``test`` functions, included in ``pytorch_example``:

.. code-block:: python

    import torch
    from petastorm.pytorch import DataLoader

    torch.manual_seed(1)
    device = torch.device('cpu')
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def _transform_row(mnist_row):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return (transform(mnist_row['image']), mnist_row['digit'])


    transform = TransformSpec(_transform_row, removed_fields=['idx'])

    with DataLoader(make_reader('file:///localpath/mnist/train', num_epochs=10,
                                transform_spec=transform), batch_size=64) as train_loader:
        train(model, device, train_loader, 10, optimizer, 1)
    with DataLoader(make_reader('file:///localpath/mnist/test', num_epochs=10,
                                transform_spec=transform), batch_size=1000) as test_loader:
        test(model, device, test_loader)

PySpark and SQL
---------------

Using the Parquet data format, which is natively supported by Spark, makes it possible to use a wide range of Spark
tools to analyze and manipulate the dataset. The example below shows how to read a Petastorm dataset
as a Spark RDD object:

.. code-block:: python

   # Create a dataframe object from a parquet file
   dataframe = spark.read.parquet(dataset_url)

   # Show a schema
   dataframe.printSchema()

   # Count all
   dataframe.count()

   # Show a single column
   dataframe.select('id').show()

SQL can be used to query a Petastorm dataset:

.. code-block:: python

   spark.sql(
      'SELECT count(id) '
      'from parquet.`file:///tmp/hello_world_dataset`').collect()

You can find a full code sample here: `pyspark_hello_world.py <https://github.com/uber/petastorm/blob/master/examples/hello_world/pyspark_hello_world.py>`_,

Non Petastorm Parquet Stores
----------------------------
Petastorm can also be used to read data directly from Apache Parquet stores. To achieve that, use
``make_batch_reader`` (and not ``make_reader``). The following table summarizes the differences
``make_batch_reader`` and ``make_reader`` functions.


==================================================================  =====================================================
``make_reader``                                                     ``make_batch_reader``
==================================================================  =====================================================
Only Petastorm datasets (created using materializes_dataset)        Any Parquet store (some native Parquet column types
                                                                    are not supported yet.
------------------------------------------------------------------  -----------------------------------------------------
The reader returns one record at a time.                            The reader returns batches of records. The size of the
                                                                    batch is not fixed and defined by Parquet row-group
                                                                    size.
------------------------------------------------------------------  -----------------------------------------------------
Predicates passed to ``make_reader`` are evaluated per single row.  Predicates passed to ``make_batch_reader`` are evaluated per batch.
==================================================================  =====================================================


Troubleshooting
---------------

See the Troubleshooting_ page and please submit a ticket_ if you can't find an
answer.


Publications
------------

1. Gruener, R., Cheng, O., and Litvin, Y. (2018) *Introducing Petastorm: Uber ATG's Data Access Library for Deep Learning*. URL: https://eng.uber.com/petastorm/


.. _Troubleshooting: docs/troubleshoot.rst
.. _ticket: https://github.com/uber/petastorm/issues/new
.. _Development: docs/development.rst

How to Contribute
=================

We prefer to receive contributions in the form of GitHub pull requests. Please send pull requests against the ``github.com/uber/petastorm`` repository.

- If you are looking for some ideas on what to contribute, check out `github issues <https://github.com/uber/petastorm/issues>`_ and comment on the issue.
- If you have an idea for an improvement, or you'd like to report a bug but don't have time to fix it please a `create a github issue <https://github.com/uber/petastorm/issues/new>`_.

To contribute a patch:

- Break your work into small, single-purpose patches if possible. It's much harder to merge in a large change with a lot of disjoint features.
- Submit the patch as a GitHub pull request against the master branch. For a tutorial, see the GitHub guides on forking a repo and sending a pull request.
- Include a detailed describtion of the proposed change in the pull request.
- Make sure that your code passes the unit tests. You can find instructions how to run the unit tests `here <https://github.com/uber/petastorm/blob/master/docs/development.rst>`_.
- Add new unit tests for your code.

Thank you in advance for your contributions!


See the Development_ for development related information.


.. inclusion-marker-end-do-not-remove
   Place contents above here if they should also appear in read-the-docs.
   Contents below are already part of the read-the-docs table of contents.
