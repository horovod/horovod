.. inclusion-marker-start-do-not-remove

Horovod Installation Guide
==========================

Requirements
------------

- GNU Linux or macOS
- Python >= 3.6
- ``g++-5`` or above, or another compiler supporting C++14
- CMake 3.13 or newer
- TensorFlow (>=1.15.0), PyTorch (>=1.5.0), or MXNet (>=1.4.1)
- (Optional) MPI

For best performance on GPU:

- `NCCL 2 <https://developer.nvidia.com/nccl>`__

To install Horovod with TensorFlow 2.10 or later you will need a compiler that supports C++17 like ``g++8`` or newer.

If Horovod cannot find CMake 3.13 or newer, the build script will attempt to pull in a recent CMake binary and run it
from a temporary location.  To select a specific binary you can also set ``HOROVOD_CMAKE`` in your environment before
installing Horovod.

Horovod does not support Windows.

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

    $ HOROVOD_WITH_TENSORFLOW={YOUR_TF_VERSION} pip install horovod[tensorflow]

To skip TensorFlow, set ``HOROVOD_WITHOUT_TENSORFLOW=1`` in your environment.

PyTorch
~~~~~~~

To ensure that Horovod is built with PyTorch support enabled:

.. code-block:: bash

    $ HOROVOD_WITH_PYTORCH={YOUR_PyTorch_VERSION} pip install horovod[pytorch]

To skip PyTorch, set ``HOROVOD_WITHOUT_PYTORCH=1`` in your environment.

MXNet
~~~~~

To ensure that Horovod is built with MXNet CPU support enabled:

.. code-block:: bash

    $ HOROVOD_WITH_MXNET={YOUR_MXNet_VERSION} pip install horovod[mxnet]

Some MXNet versions do not work with Horovod:

- MXNet 1.4.0 and earlier have `GCC incompatibility issues <https://github.com/horovod/horovod/issues/884>`__. Use MXNet 1.4.1 or later with Horovod 0.16.2 or later to avoid these incompatibilities.
- MXNet 1.5.1, 1.6.0, 1.7.0, and 1.7.0.post1 are missing MKLDNN headers, so they do not work with Horovod. Use 1.5.1.post0, 1.6.0.post0, and 1.7.0.post0, respectively.
- MXNet 1.6.0.post0 and 1.7.0.post0 are only available as mxnet-cu101 and mxnet-cu102.

To skip MXNet, set ``HOROVOD_WITHOUT_MXNET=1`` in your environment.

Keras
~~~~~

Standalone Keras support is currently only available for the TensorFlow backend.

To ensure that Horovod is built with Keras support available:

.. code-block:: bash

    $ HOROVOD_WITH_TENSORFLOW={YOUR_TF_VERSION} pip install horovod[tensorflow,keras]

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

**Note**: macOS users must install `libuv <https://github.com/libuv/libuv>`_ in order to use Gloo:

.. code-block:: bash

    $ brew install libuv

Tensor Operations
-----------------

For running on GPUs with optimal performance, we recommend installing Horovod with NCCL support following the
`Horovod on GPU <gpus.rst>`_ guide.

For tensor data on CPU, you can use MPI, Gloo, and Intel's oneCCL. By default, the framework used by your controller
will be used for CPU operations. You can override this by setting ``HOROVOD_CPU_OPERATIONS`` in your environment.

NCCL
~~~~

NCCL is supported for Allreduce, Allgather, Broadcast, and Alltoall operations.  You can enable these by setting
``HOROVOD_GPU_OPERATIONS=NCCL`` during installation.

NCCL operations are supported on both Nvidia (CUDA) and AMD (ROCm) GPUs. You can set ``HOROVOD_GPU`` in your
environment to specify building with CUDA or ROCm. CUDA will be assumed if not specified.

Note that Alltoall requires NCCL version >= 2.7.0.

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
`Horovod with Intel(R) oneCCL <oneccl.rst>`_ for more details.

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

Installing Horovod with Conda (+pip)
------------------------------------

To use Conda to install PyTorch, TensorFlow, MXNet, Horovod, as well as GPU dependencies such as
NVIDIA CUDA Toolkit, cuDNN, NCCL, etc., see `Build a Conda Environment with GPU Support for Horovod <conda.rst>`_.

Environment Variables
---------------------

Optional environment variables that can be set to configure the installation process for Horovod.

Due to `PEP-517 <https://peps.python.org/pep-0517/>`_ we can't rely on any DL library being installed into
the build env, therefore we need to tell the build env specific DL library versions we require.
This isn't the prettiest solution, however it is the most pragmatic.

Possible values are given in curly brackets: {}.

* ``HOROVOD_DEBUG`` - {1}. Install a debug build of Horovod with checked assertions, disabled compiler optimizations etc.
* ``HOROVOD_BUILD_ARCH_FLAGS`` - additional C++ compilation flags to pass in for your build architecture.
* ``HOROVOD_CUDA_HOME`` - path where CUDA include and lib directories can be found.
* ``HOROVOD_BUILD_CUDA_CC_LIST`` - List of compute capabilities to build Horovod CUDA kernels for (example: ``HOROVOD_BUILD_CUDA_CC_LIST=60,70,75``)
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
* ``HOROVOD_GPU_OPERATIONS`` - {NCCL, MPI}. Framework to use for GPU tensor allreduce, allgather, and broadcast.
* ``HOROVOD_GPU_ALLREDUCE`` - {NCCL, MPI}. Framework to use for GPU tensor allreduce.
* ``HOROVOD_GPU_ALLGATHER`` - {NCCL, MPI}. Framework to use for GPU tensor allgather.
* ``HOROVOD_GPU_BROADCAST`` - {NCCL, MPI}. Framework to use for GPU tensor broadcast.
* ``HOROVOD_GPU_ALLTOALL`` - {NCCL, MPI}. Framework to use for GPU tensor alltoall.
* ``HOROVOD_GPU_REDUCESCATTER`` - {NCCL, MPI}. Framework to use for GPU tensor reducescatter.
* ``HOROVOD_ALLOW_MIXED_GPU_IMPL`` - {1}. Allow Horovod to install with NCCL allreduce and MPI GPU allgather / broadcast / alltoall / reducescatter.  Not recommended due to a possible deadlock.
* ``HOROVOD_CPU_OPERATIONS`` - {MPI, GLOO, CCL}. Framework to use for CPU tensor allreduce, allgather, and broadcast.
* ``HOROVOD_CMAKE`` - path to the CMake binary used to build Horovod.
* ``HOROVOD_WITH_TENSORFLOW`` - {TF pypi version}. If set require Horovod to install with specific TensorFlow version support enabled.
* ``HOROVOD_WITHOUT_TENSORFLOW`` - {1}. Skip installing TensorFlow support.
* ``HOROVOD_WITH_PYTORCH`` - {PyTorch pypi version}. If set require Horovod to install with specific PyTorch version support enabled.
* ``HOROVOD_WITHOUT_PYTORCH`` - {1}. Skip installing PyTorch support.
* ``HOROVOD_WITH_MXNET`` - {MXNet pypi version}. If set require Horovod to install with specific MXNet version support enabled.
* ``HOROVOD_WITHOUT_MXNET`` - {1}. Skip installing MXNet support.

.. inclusion-marker-end-do-not-remove
