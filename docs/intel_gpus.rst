.. inclusion-marker-start-do-not-remove

Horovod on Intel GPU
====================

Intel has brought **Intel® oneAPI (DPC++ compiler/oneCCL/IMPI included)** software stack into Horovod to make TensorFlow distributed workloads run on `Intel GPU devices <https://www.intel.com/content/www/us/en/products/details/discrete-gpus.html>`__
together with `Intel® Extension for TensorFlow* <https://github.com/intel/intel-extension-for-tensorflow>`__, which is an open-source solution to run TensorFlow on Intel AI hardware.

DPC++ compiler/oneCCL/IMPI packaged into oneAPI Base Tookit are core set of tools and libraries for developing high-performance computing and cross-devices communication applications. 
**DPC++ compiler** is an extension of C++ for heterogeneous computing and based on SYCL. 
**OneCCL** (oneAPI Collective Communications Library) and **IMPI** provide high-performance communication patterns for distributed workloads in Intel® GPU cluster.

To use Horovod on Intel GPU, please follow steps below.

Install Intel GPU driver
~~~~~~~~~~~~~~~~~~~~~~~~

Refer to `Installation Guides <https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-max-series>`__
for latest driver installation. 

Recommend to install verified driver version `602 <https://dgpu-docs.intel.com/releases/stable_602_20230323.html>`__ for hardware platforms: 
Intel® Data Center GPU Max Series and Intel® Data Center GPU Flex Series 170.


Install oneAPI Base Toolkit Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are necessary 3 components for horovod on Intel GPU:

- Intel® oneAPI DPC++ Compiler.
- Intel® oneAPI Message Passing Interface (IMPI).
- Intel® oneAPI Collective Communications Library (oneCCL).

.. code-block:: bash

    $ wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/7deeaac4-f605-4bcf-a81b-ea7531577c61/l_BaseKit_p_2023.1.0.46401_offline.sh     
    $ sudo sh ./l_BaseKit_p_2023.1.0.46401_offline.sh

For any more details, follow the `procedures <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html>`__.
   
Setup environment variables

.. code-block:: bash

    $ source /path to basekit/intel/oneapi/compiler/latest/env/vars.sh
    $ source /path to basekit/intel/oneapi/mpi/latest/env/vars.sh
    $ source /path to basekit/intel/oneapi/ccl/latest/env/vars.sh

Install Deep Learning Frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To bring Intel GPU devices into **TensorFlow** community for AI workload acceleration, Intel promoted a new user visible ``XPU`` device type as a device abstraction for Intel computation architectures and implemented corresponding device runtime in **Intel® Extension for TensorFlow**.

To use **TensorFlow** and **Intel® Extension for TensorFlow**, please follow `Tensorflow 2.12.0 installation <https://www.tensorflow.org/install>`__  and `Intel® Extension for TensorFlow* 1.2 installation <https://github.com/intel/intel-extension-for-tensorflow/tree/r1.2#install>`__.

Install the Horovod pip package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   $ python setup.py sdist
   $ CC=icx CXX=icpx pip install --no-cache-dir dist/*.tar.gz

Get started with TensorFlow models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use horovod with models implemented by Tensorflow, please refer to `Horovod with TensorFlow <tensorflow.rst>`_  to modify your training script or see `tensorflow2 examples <https://github.com/horovod/horovod/blob/master/examples/tensorflow2/>`_ directory for full training examples.

.. inclusion-marker-end-do-not-remove
