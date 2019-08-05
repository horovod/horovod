
.. inclusion-marker-start-do-not-remove


Contributor Guide
=================

This guide covers the process of contributing to Horovod as a developer.


Environment Setup
~~~~~~~~~~~~~~~~~

Clone the repository locally:

.. code-block:: bash

    $ git clone --recursive https://github.com/horovod/horovod.git

Develop within a virtual environment to avoid dependency issues:

.. code-block:: bash

    $ virtualenv env
    $ . env/bin/activate

We recommend installing package versions that match with those under test in
`Buildkite <https://github.com/horovod/horovod/blob/master/.buildkite/gen-pipeline.sh>`__.

For example:

.. code-block:: bash

    $ pip install tensorflow==1.14.0
    $ pip install keras==2.2.4
    $ pip install torch==1.1.0 torchvision
    $ pip install pytest
    $ pip install h5py future scipy mpi4py pyspark mxnet


Build and Install
~~~~~~~~~~~~~~~~~

First, uninstall any existing version of Horovod.  Be sure to do this *outside* the Horovod root directory:

.. code-block:: bash

    $ cd $HOME
    $ pip uninstall -y horovod
    $ cd -

From *inside* the Horovod root directory, remove any previous build artifacts and then install Horovod:

.. code-block:: bash

    $ rm -rf build/ dist/
    $ HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 python setup.py install

Set ``HOROVOD_WITHOUT_[FRAMEWORK]=1`` to disable building Horovod plugins for that framework.
This is useful when you’re testing a feature of one framework in particular and wish to save time.


Testing
~~~~~~~

Horovod has unit tests for all frameworks you can run from the tests directory:

.. code-block:: bash

    $ cd test
    $ mpirun -np 2 pytest -v

**Note:** You will need PySpark and Java to run the Spark tests.

**IMPORTANT:** Some tests contain GPU-only codepaths that will be skipped if running without GPU support.


Adding Custom Operations
~~~~~~~~~~~~~~~~~~~~~~~~

Operations in Horovod are used to transform Tensors across workers.  Horovod currently supports operations that
implement Broadcast, Allreduce, and Allgather interfaces.  Gradients in Horovod are aggregated through
Allreduce operations (with the exception of sparse gradients, which use Allgather).

All data transfer operations are implemented in the
`horovod/common/ops <https://github.com/horovod/horovod/tree/master/horovod/common/ops>`__ directory.  Implementations
are organized by the collective communication library used to perform the operation (e.g.,
`mpi_operations.cc <https://github.com/horovod/horovod/blob/master/horovod/common/ops/mpi_operations.cc>`__ for MPI).

To create a new custom operation, start by defining a new class that inherits from the base operation, in the file
corresponding to the library you'll use to implement the operation:

.. code-block:: c++

    class CustomAllreduce : public AllreduceOp {
    public:
      CustomAllreduce(MPIContext* mpi_context, HorovodGlobalState* global_state);

      virtual ~CustomAllreduce() = default;

      Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

      bool Enabled(const ParameterManager& parameter_manager,
                   const std::vector<TensorTableEntry>& entries,
                   const Response& response) const override;

The ``Execute`` member function is responsible for performing the operation on a list of Tensors. The ``entries``
parameter provides access to all the Tensor buffers and metadata that need to be processed,
and the ``response`` parameter contains additional metadata including which devices are being used by different ranks.

``Enabled`` should return true if your operation can be performed on the given Tensor entries subject to the
current parameter settings and response metadata.

Once you've written the implementation for your operation, add it to the ``OperationManager`` in the
``CreateOperationManager`` function of
`operations.cc <https://github.com/horovod/horovod/blob/master/horovod/common/operations.cc>`__.  Because more than one
operation may be *enabled* at a time, but only one will be performed on a given vector of Tensor entries, consider the
order of your operation in the ``OperationManager`` vector before adding it in.

The first operations in the vector will be checked before those at the end, and the first operation that is *enabled*
will be performed. Broadly, the order of operations should be:

1. Custom operations that trigger based on parameters configured at runtime (e.g., ``NCCLHierarchicalAllreduce``).
2. Accelerated operations that take advantage of specialized hardware where available (e.g., ``NCCLAllreduce``).
3. Default operations that can run using standard CPUs and host memory (e.g., ``MPIAllreduce``).

Most custom operations that require preconditions such as runtime flags will fall into the first category.


Adding Compression Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gradient compression is used to reduce the amount of data sent over the network during an Allreduce operation.  Such
compression algorithms are implemented per framework (TensorFlow, PyTorch, MXNet, etc.) in
``horovod/[framework]/compression.py``
(see: `TensorFlow <https://github.com/horovod/horovod/blob/master/horovod/tensorflow/compression.py>`__,
`PyTorch <https://github.com/horovod/horovod/blob/master/horovod/torch/compression.py>`__).

To implement a new compression algorithm, first add a new class inheriting from ``Compressor``:

.. code-block:: python

    class CustomCompressor(Compressor):
        @staticmethod
        def compress(tensor):
            # do something here ...
            return tensor_compressed, ctx

        @staticmethod
        def decompress(tensor, ctx):
            # do something here ...
            return tensor_decompressed

The ``compress`` method takes a Tensor gradient and returns it in its compressed form, along with any additional context
necessary to decompress the tensor back to its original form.  Similarly, ``decompress`` takes in a compressed tensor
with its context and returns a decompressed tensor.  Compression can be done in pure Python, or in C++ using a custom
op (e.g., in `mpi_ops.cc <https://github.com/horovod/horovod/blob/master/horovod/tensorflow/mpi_ops.cc>`__ for
TensorFlow).

Once implemented, add your ``Compressor`` subclass to the ``Compressor`` class, which emulates an enumeration API:

.. code-block:: python

    class Compression(object):
        # ...

        custom = CustomCompressor

Finally, you can start using your new compressor by passing it to the ``DistributedOptimizer``:

.. code-block:: python

    opt = hvd.DistributedOptimizer(opt, compression=hvd.Compression.custom)


.. inclusion-marker-end-do-not-remove
