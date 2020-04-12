.. inclusion-marker-start-do-not-remove

Horovod Installation Guide
==========================

Frameworks
----------

You can build Horovod for TensorFlow, PyTorch, and MXNet. By default, Horovod will attempt to build
support for all of them. At least one must be enabled for Horovod to install successfully.

To ensure that framework dependencies are properly installed before attempting to install Horovod, append
extra arguments that identify the required frameworks:

.. code-block:: bash

    $ pip install horovod[tensorflow,keras,pytorch,mxnet,spark]

In addition to specifying framework requirements individually, you can require all frameworks collectively:

.. code-block:: bash

    $ pip install horovod[all-frameworks]

This is useful when building Horovod as part of a larger collection of dependencies at once, relying on the pip
compiler to determine the correct install order.

TensorFlow
~~~~~~~~~~

To ensure that Horovod is built with TensorFlow support enabled:

.. code-block:: bash

    $ HOROVOD_WITH_TENSORFLOW=1 pip install horovod[tensorflow]

To skip TensorFlow, set ``HOROVOD_WITHOUT_TENSORFLOW=1`` in your environment.

If you've installed TensorFlow from `PyPI <https://pypi.org/project/tensorflow>`__, make sure that
the ``g++-4.8.5`` or ``g++-4.9`` is installed.

PyTorch
~~~~~~~

To ensure that Horovod is built with PyTorch support enabled:

.. code-block:: bash

    $ HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]

To skip PyTorch, set ``HOROVOD_WITHOUT_PYTORCH=1`` in your environment.

If you've installed PyTorch from `PyPI <https://pypi.org/project/torch>`__, make sure that the ``g++-4.9`` or
above is installed.

MXNet
~~~~~

To ensure that Horovod is built with MXNet support enabled:

.. code-block:: bash

    $ HOROVOD_WITH_MXNET=1 pip install horovod[mxnet]

To skip MXNet, set ``HOROVOD_WITHOUT_MXNET=1`` in your environment.

Keras
~~~~~

Standalone Keras support is currently only available for the TensorFlow backend.

To ensure that Horovod is built with Keras support available:

.. code-block:: bash

    $ HOROVOD_WITH_TENSORFLOW=1 pip install horovod[tensorflow,keras]

There are no plugins built for Keras, but the TensorFlow plugin must be enabled in order to use Horovod with Keras.

Spark
~~~~~

Horovod can be used with Spark in combination with any of the frameworks above.

To ensure Horovod has all the necessary requirements in order to run on top of Spark:

.. code-block:: bash

    $ pip install horovod[spark]

Controllers
-----------

The controller is used for coordinating work between Horovod processes (determining which tensors to process). We
provide controller implementations for both MPI and Gloo. By default, Horovod will attempt to build support for both
of them. At least one must be enabled for Horovod to install successfully.

MPI
~~~

MPI is the original controller for Horovod.  It uses ``mpirun`` to launch worker processes (``horovodrun`` will use
``mpirun`` under the hood when using MPI).

To use Horovod with MPI, install `Open MPI <https://www.open-mpi.org/>`_ or another MPI implementation.
Learn how to install Open MPI `on this page <https://www.open-mpi.org/faq/?category=building#easy-build>`_.

**Note**: Open MPI 3.1.3 has an issue that may cause hangs. The recommended fix is to downgrade to Open MPI 3.1.2 or
upgrade to Open MPI 4.0.0.

* To force Horovod to install with MPI support, set ``HOROVOD_WITH_MPI=1`` in your environment.
* To force Horovod to skip building MPI support, set ``HOROVOD_WITHOUT_MPI=1``.

If both MPI and Gloo are enabled in your installation, then MPI will be the default controller.

Gloo
~~~~

Gloo is a more recent controller for Horovod that does not require additional dependencies besides CMake to install.

When used as a controller in combination with NCCL, Gloo performs almost identically to MPI on standard benchmarks.

* To force Horovod to install with Gloo support, set ``HOROVOD_WITH_GLOO=1`` in your environment.
* To force Horovod to skip building Gloo support, set ``HOROVOD_WITHOUT_GLOO=1``.

Gloo mode uses ``horovodrun`` to launch worker processes.

Gloo is required to use the elastic / fault tolerant API for Horovod.

If Horovod in unable to find the CMake binary, you may need to set ``HOROVOD_CMAKE`` in your environment before
installing.

Tensor Operations
-----------------

For running on GPUs with optimal performance, we recommend installing Horovod with NCCL support following the
`Horovod on GPU <gpus.rst>`_ guide.

For tensor data on CPU, you can use MPI, Gloo, and Intel's oneCCL. By default, the framework used by your controller
will be used for CPU operations. You can override this by setting ``HOROVOD_CPU_OPERATIONS`` in your environment.

NCCL
~~~~

NCCL is currently supported for Allreduce and Broadcast operations.  You can enable these by setting
``HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL`` during installation.

NCCL operations are supported on both Nvidia (CUDA) and AMD (ROCm) GPUs. You can set ``HOROVOD_GPU`` in your
environment to specify building with CUDA or ROCm. CUDA will be assumed if not specified.

MPI
~~~

When using an MPI controller, MPI will be used when NCCL is unavailable, or if tensors are placed in host memory prior
to the allreduce request. In cases where NCCL is unavailable, MPI has been shown to outperform Gloo for CPU tensor
operations.

MPI can also be used for GPU operations, but this is not recommended in most cases. See `Horovod on GPU <gpus.rst>`_ for
more details.

Gloo
~~~~

When using a Gloo controller, Gloo will be used in place of MPI for CPU operations by default.

oneCCL
~~~~~~

oneCCL is an Intel library for accelerated collective operations on CPU. See
`Horovod with Intel(R) oneCCL <oneccl.md>`_ for more details.

Set ``HOROVOD_CPU_OPERATIONS=CCL`` to use oneCCL.


Check Build
-----------

After successfully installing Horovod, run:

.. code-block:: bash

    $ horovodrun --check-build

Every feature that was successfully enabled will be marked with an 'X'. If you intended to install Horovod with a
feature that is not listed as enabled, you can reinstall Horovod, setting the appropriate environment variables to
diagnose failures:

.. code-block:: bash

    $ pip uninstall horovod
    $ HOROVOD_WITH_...=1 pip install --no-cache-dir horovod


Environment Variables
---------------------

Optional environment variables that can be set to configure the installation process for Horovod.

Possible values are given in curly brackets: {}.

* ``HOROVOD_BUILD_ARCH_FLAGS`` - additional C++ compilation flags to pass in for your build architecture.
* ``HOROVOD_MPICXX_SHOW`` - custom command to show MPI compilation flags (default: ``mpicxx -show``).
* ``HOROVOD_CUDA_HOME`` - path where CUDA include and lib directories can be found.
* ``HOROVOD_CUDA_INCLUDE`` - path to CUDA include directory.
* ``HOROVOD_CUDA_LIB`` - path to CUDA lib directory.
* ``HOROVOD_ROCM_HOME`` - path where ROCm include and lib directories can be found.
* ``HOROVOD_NCCL_HOME`` - path where NCCL include and lib directories can be found.
* ``HOROVOD_NCCL_INCLUDE`` - path to NCCL include directory.
* ``HOROVOD_NCCL_LIB`` - path to NCCL lib directory.
* ``HOROVOD_NCCL_LINK`` - {SHARED, STATIC}. Mode to link NCCL library. Defaults to STATIC for CUDA, SHARED for ROCm.
* ``HOROVOD_WITH_GLOO`` - {1}. Require that Horovod is built with Gloo support enabled.
* ``HOROVOD_WITHOUT_GLOO`` - {1}. Skip building with Gloo support.
* ``HOROVOD_WITH_MPI`` - {1}. Require that Horovod is built with MPI support enabled.
* ``HOROVOD_WITHOUT_MPI`` - {1}. Skip building with MPI support.
* ``HOROVOD_GPU`` - {CUDA, ROCM}. Framework to use for GPU operations.
* ``HOROVOD_GPU_ALLREDUCE`` - {NCCL, MPI, DDL}. Framework to use for GPU tensor allreduce.
* ``HOROVOD_GPU_ALLGATHER`` - {MPI}. Framework to use for GPU tensor allgather.
* ``HOROVOD_GPU_BROADCAST`` - {NCCL, MPI}. Framework to use for GPU tensor broadcast.
* ``HOROVOD_ALLOW_MIXED_GPU_IMPL`` - {1}. Allow Horovod to install with NCCL allreduce and MPI GPU allgather / broadcast.  Not recommended due to a possible deadlock.
* ``HOROVOD_CPU_OPERATIONS`` - {MPI, GLOO, CCL}. Framework to use for CPU tensor allreduce, allgather, and broadcast.
* ``HOROVOD_CMAKE`` - path to the CMake binary used to build Gloo (not required when using MPI).
* ``HOROVOD_WITH_TENSORFLOW`` - {1}. Require Horovod to install with TensorFlow support enabled.
* ``HOROVOD_WITHOUT_TENSORFLOW`` - {1}. Skip installing TensorFlow support.
* ``HOROVOD_WITH_PYTORCH`` - {1}. Require Horovod to install with PyTorch support enabled.
* ``HOROVOD_WITHOUT_PYTORCH`` - {1}. Skip installing PyTorch support.
* ``HOROVOD_WITH_MXNET`` - {1}. Require Horovod to install with MXNet support enabled.
* ``HOROVOD_WITHOUT_MXNET`` - {1}. Skip installing MXNet support.

.. inclusion-marker-end-do-not-remove
