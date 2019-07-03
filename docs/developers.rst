
.. inclusion-marker-start-do-not-remove


Developer Guide
===============

This guide covers the process of contributing to Horovod as a developer.


Environment Setup
~~~~~~~~~~~~~~~~~

Clone the repository locally:

.. code-block:: bash

    $ git clone --recurse-submodules https://github.com/horovod/horovod.git

Be sure to run within a virtual environment to avoid dependency issues:

.. code-block:: bash

    $ virtualenv env
    $ ./env/bin/activate

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

**IMPORTANT:** Some tests contain GPU-only codepaths that will not execute without compiling with GPU support.


Adding Custom Operations
~~~~~~~~~~~~~~~~~~~~~~~~

Operations in Horovod are used to transform Tensors across workers.  Horovod currently supports operations that
implement Broadcast, Allreduce, and Allgather interfaces.  Gradients in Horovod are aggregated through
Allreduce operations (with the exception of sparse gradients which use Allgather).

All data transfer operations are implemented in the ``horovod/common/ops`` directory.  Implementations are organized by
the collective communication library used to perform the operation (e.g., `mpi_operations.cc` for MPI).

To create a new custom operation, start by defining a new class that inherits from the base operation, in the file
corresponding to the library you'll be using to implement the operation:

.. code-block:: c++

    class CustomAllreduce : public AllreduceOp {
    public:
      CustomAllreduce(MPIContext* mpi_context, HorovodGlobalState* global_state);

      virtual ~CustomAllreduce() = default;

      Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

      bool Enabled(const ParameterManager& param_manager,
                   const std::vector<TensorTableEntry>& entries,
                   const Response& response) const override;

The ``Execute`` member function is responsible for taking a list of Tensors and performing the transformation between
workers.  The ``entries`` parameter provides access to all the Tensor buffers and metadata that need to be transformed,
and the ``response`` parameter contains additional metadata including which devices are being used by different ranks.

``Enabled`` should return true if your operation can be used to transform the given Tensor entries subject to the
current parameter settings and response metadata.

Once you've written the implementation for your operation, add it to the ``OperationManager`` in the
``CreateOperationManager`` function of ``operations.cc``.  Because more than one operation may be *enabled* at a
time, but only one will be performed on a given vector of Tensor entries, the order of your operation in the
``OperationManager`` vector needs to be considered.

The first operations in the vector will be checked before those at the end, and the first operation that is *enabled*
will be performed. Broadly, the order of operations should be:

1. Custom operations that trigger based on parameters configured at runtime (e.g., ``NCCLHierarchicalAllreduce``).
2. Accelerated operations that take advantage of specialized hardware where available (e.g., ``NCCLAllreduce``).
3. Default operations that can run using standard CPUs and host memory (e.g., ``MPIAllreduce``).

Most custom operations that require preconditions such as runtime flags will fall into the first category.
