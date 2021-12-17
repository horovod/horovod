.. inclusion-marker-start-do-not-remove

Build a Conda Environment with GPU Support for Horovod
======================================================

In this section we describe how to build Conda environments for deep learning projects using 
Horovod to enable distributed training across multiple GPUs (either on the same node or 
spread across multuple nodes).

Installing the NVIDIA CUDA Toolkit
----------------------------------

Install `NVIDIA CUDA Toolkit 10.1`_ (`documentation`_) which is the most recent version of NVIDIA 
CUDA Toolkit supported by all three deep learning frameworks that are currently supported by 
Horovod.

Why not just use the ``cudatoolkit`` package?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically when installing PyTorch, TensorFlow, or Apache MXNet with GPU support using Conda, you 
add the appropriate version of the ``cudatoolkit`` package to your ``environment.yml`` file. 
Unfortunately, for the moment at least, the cudatoolkit packages available via Conda do not 
include the `NVIDIA CUDA Compiler (NVCC)`_, which is required in order to build Horovod extensions 
for PyTorch, TensorFlow, or MXNet.

What about the ``cudatoolkit-dev`` package?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While there are ``cudatoolkit-dev`` packages available from ``conda-forge`` that do include NVCC, 
we have had difficulty getting these packages to consistently install properly. Some of the 
available builds require manual intervention to accept license agreements, making these builds 
unsuitable for installing on remote systems (which is critical functionality). Other builds seems 
to work on Ubuntu but not on other flavors of Linux.

Despite this, we would encourage you to try adding ``cudatoolkit-dev`` to your ``environment.yml`` 
file and see what happens! The package is well maintained so perhaps it will become more stable in 
the future.

Use the ``nvcc_linux-64`` meta-package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most robust approach to obtain NVCC and still use Conda to manage all the other dependencies 
is to install the NVIDIA CUDA Toolkit on your system and then install a meta-package 
`nvcc_linux-64`_ from conda-forge, which configures your Conda environment to use the NVCC 
installed on the system together with the other CUDA Toolkit components installed inside the Conda 
environment.

The ``environment.yml`` file
----------------------------

We prefer to specify as many dependencies as possible in the Conda ``environment.yml`` file 
and only specify dependencies in ``requirements.txt`` for install via ``pip`` that are not 
available via Conda channels. Check the Horovod `installation guide`_ for details of required 
dependencies.

Channel Priority
^^^^^^^^^^^^^^^^

Use the recommended channel priorities. Note that ``conda-forge`` has priority over 
``defaults`` and ``pytorch`` has priority over ``conda-forge``. ::

    name: null

    channels:
    - pytorch
    - conda-forge
    - defaults

Dependencies
^^^^^^^^^^^^

There are a few things worth noting about the dependencies.

1. Even though you have installed the NVIDIA CUDA Toolkit manually, you should still use Conda to 
   manage the other required CUDA components such as ``cudnn`` and ``nccl`` (and the optional 
   ``cupti``).
2. Use two meta-packages, ``cxx-compiler`` and ``nvcc_linux-64``, to make sure that suitable C, 
   and C++ compilers are installed and that the resulting Conda environment is aware of the 
   manually installed CUDA Toolkit.
3. Horovod requires some controller library to coordinate work between the various Horovod 
   processes. Typically this will be some MPI implementation such as `OpenMPI`_. However, rather 
   than specifying the ``openmpi`` package directly, you should instead opt for `mpi4py`_ Conda 
   package which provides a CUDA-aware build of OpenMPI.
4. Horovod also support the `Gloo`_ collective communications library that can be used in place of 
   MPI. Include ``cmake`` to insure that the Horovod extensions for Gloo are built.

Below are the core required dependencies. The complete ``environment.yml`` file is available 
on `GitHub`_. ::

    dependencies:
    - bokeh=1.4
    - cmake=3.16 # insures that Gloo library extensions will be built
    - cudnn=7.6
    - cupti=10.1
    - cxx-compiler=1.0 # insures C and C++ compilers are available
    - jupyterlab=1.2
    - mpi4py=3.0 # installs cuda-aware openmpi
    - nccl=2.5
    - nodejs=13
    - nvcc_linux-64=10.1 # configures environment to be "cuda-aware"
    - pip=20.0
    - pip:
        - mxnet-cu101mkl==1.6.* # MXNET is installed prior to horovod
        - -r file:requirements.txt
    - python=3.7
    - pytorch=1.5
    - tensorboard=2.1
    - tensorflow-gpu=2.1
    - torchvision=0.6

The ``requirements.txt`` file
-----------------------------

The ``requirements.txt`` file is where all of the ``pip`` dependencies, including Horovod itself, 
are listed for installation. In addition to Horovod we typically will also use ``pip`` to install 
JupyterLab extensions to enable GPU and CPU resource monitoring via `jupyterlab-nvdashboard`_ and 
Tensorboard support via `jupyter-tensorboard`_. ::

    horovod==0.19.*
    jupyterlab-nvdashboard==0.2.*
    jupyter-tensorboard==0.2.*

    # make sure horovod is re-compiled if environment is re-built
    --no-binary=horovod

Note the use of the ``--no-binary`` option at the end of the file. Including this option ensures 
that Horovod will be re-built whenever the Conda environment is re-built.

Building the Conda environment
------------------------------

After adding any necessary dependencies that should be downloaded via Conda to the 
``environment.yml`` file and any dependencies that should be downloaded via ``pip`` to the 
``requirements.txt`` file, create the Conda environment in a sub-directory ``env`` of your 
project directory by running the following commands.

.. code-block:: bash

    $ export ENV_PREFIX=$PWD/env
    $ export HOROVOD_CUDA_HOME=$CUDA_HOME
    $ export HOROVOD_NCCL_HOME=$ENV_PREFIX
    $ export HOROVOD_GPU_OPERATIONS=NCCL
    $ conda env create --prefix $ENV_PREFIX --file environment.yml --force

By default Horovod will try and build extensions for all detected frameworks. See the 
documentation on `environment variables`_ for the details on additional environment variables that 
can be set prior to building Horovod.

Once the new environment has been created you can activate the environment with the following 
command.

.. code-block:: bash

    $ conda activate $ENV_PREFIX

The ``postBuild`` file
^^^^^^^^^^^^^^^^^^^^^^

If you wish to use any JupyterLab extensions included in the ``environment.yml`` and 
``requirements.txt`` files, then you may need to rebuild the JupyterLab application.

For simplicity, we typically include the instructions for re-building JupyterLab in a 
``postBuild`` script. Here is what this script looks like in the example Horovod environments.

.. code-block:: bash

    jupyter labextension install --no-build jupyterlab-nvdashboard 
    jupyter labextension install --no-build jupyterlab_tensorboard
    jupyter lab build

Use the following commands to source the ``postBuild`` script.

.. code-block:: bash

    $ conda activate $ENV_PREFIX # optional if environment already active
    $ . postBuild

Listing the contents of the Conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To see the full list of packages installed into the environment, run the following command.

.. code-block:: bash

    $ conda activate $ENV_PREFIX # optional if environment already active
    $ conda list

Verifying the Conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After building the Conda environment, check that Horovod has been built with support for the deep 
learning frameworks TensorFlow, PyTorch, Apache MXNet, and the contollers MPI and Gloo with the 
following command.

.. code-block:: bash

    $ conda activate $ENV_PREFIX # optional if environment already active
    $ horovodrun --check-build

You should see output similar to the following.::

    Horovod v0.19.4:
    Available Frameworks:
        [X] TensorFlow
        [X] PyTorch
        [X] MXNet
    Available Controllers:
        [X] MPI
        [X] Gloo
    Available Tensor Operations:
        [X] NCCL
        [ ] DDL
        [ ] CCL
        [X] MPI
        [X] Gloo

Wrapping it all up in a Bash script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We typically wrap these commands into a shell script ``create-conda-env.sh``. Running the shell 
script will set the Horovod build variables, create the Conda environment, activate the Conda 
environment, and build JupyterLab with any additional extensions.

.. code-block:: bash

    #!/bin/bash --login

    set -e
    
    export ENV_PREFIX=$PWD/env
    export HOROVOD_CUDA_HOME=$CUDA_HOME
    export HOROVOD_NCCL_HOME=$ENV_PREFIX
    export HOROVOD_GPU_OPERATIONS=NCCL
    conda env create --prefix $ENV_PREFIX --file environment.yml --force
    conda activate $ENV_PREFIX
    . postBuild

We recommend that you put scripts inside a ``bin`` directory in your project root directory. Run 
the script from the project root directory as follows.

.. code-block:: bash

    ./bin/create-conda-env.sh # assumes that $CUDA_HOME is set properly

Updating the Conda environment
------------------------------

If you add (remove) dependencies to (from) the ``environment.yml`` file or the 
``requirements.txt`` file after the environment has already been created, then you can 
re-create the environment with the following command.

.. code-block:: bash

    $ conda env create --prefix $ENV_PREFIX --file environment.yml --force

However, whenever we add (remove) dependencies we prefer to re-run the Bash script which will re-build 
both the Conda environment and JupyterLab.

.. code-block:: bash

    $ ./bin/create-conda-env.sh

.. _NVIDIA CUDA Toolkit 10.1: https://developer.nvidia.com/cuda-10.1-download-archive-update2
.. _documentation: https://docs.nvidia.com/cuda/archive/10.1/
.. _NVIDIA CUDA Compiler (NVCC): https://docs.nvidia.com/cuda/archive/10.1/cuda-compiler-driver-nvcc/index.html
.. _nvcc_linux-64: https://github.com/conda-forge/nvcc-feedstock
.. _installation guide: https://horovod.readthedocs.io/en/latest/install_include.html
.. _OpenMPI: https://www.open-mpi.org/
.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
.. _Gloo: https://github.com/facebookincubator/gloo
.. _GitHub: https://github.com/kaust-vislab/horovod-gpu-data-science-project
.. _jupyterlab-nvdashboard: https://github.com/rapidsai/jupyterlab-nvdashboard
.. _jupyter-tensorboard: https://github.com/lspvic/jupyter_tensorboard
.. _environment variables: https://horovod.readthedocs.io/en/latest/install_include.html#environment-variables

.. inclusion-marker-end-do-not-remove