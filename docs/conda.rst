Building a Conda environment with GPU support for Horovod
=========================================================

In this section I describe how I build Conda environments for my deep learning projects when I am 
using Horovod to enable distributed training across multiple GPUs (either on the same node or 
spread across multuple nodes). If you like my approach then you can make use of the template 
repository on GitHub to get started with your next Horovod data science project!

Installing the NVIDIA CUDA Toolkit
----------------------------------

First thing you need to do is to install the appropriate version of the NVIDIA CUDA Toolkit on 
your workstation. I am using NVIDIA CUDA Toolkit 10.1 (documentation) which works with all three 
deep learning frameworks that are currently supported by Horovod.

Why not just use the cudatoolkit package?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically when installing PyTorch, TensorFlow, or Apache MXNet with GPU support using Conda you 
simply add the appropriate version of the cudatoolkit package to your environment.yml file.
Unfortunately, for the moment at least, the cudatoolkit package available from conda-forge does 
not include NVCC which is required in order to use Horovod with either PyTorch, TensorFlow, or 
MXNet as you need to compile extensions.

What about the cudatoolkit-dev package?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While there are cudatoolkit-dev packages available from conda-forge that do include NVCC, I have 
had difficult getting these packages to consistently install properly. Some of the available 
builds require manual intervention to accept license agreements making these builds unsuitable 
for installing on remote systems (which is critical functionality). Other builds seems to work 
on Ubuntu but not on other flavors of Linux.

I would encourage you to try adding cudatoolkit-dev to your environment.yml file and see what 
happens! The package is well maintained so perhaps it will become more stable in the future.

Use the nvcc_linux-64 meta-pacakge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most robust approach to obtain NVCC and still use Conda to manage all the other dependencies 
is to install the NVIDIA CUDA Toolkit on your system and then install a meta-package nvcc_linux-64 
from conda-forge which configures your Conda environment to use the NVCC installed on the system 
together with the other CUDA Toolkit components installed inside the Conda environment. For more 
details on this package I recommend reading through the issue threads on GitHub.

The environment.yml file
------------------------

I prefer to specify as many dependencies as possible in the Conda environment.yml file and only 
specify dependencies in requirements.txt that are not available via Conda channels. Check the 
official Horovod installation guide for details of required dependencies.

Channel Priority
^^^^^^^^^^^^^^^^

I use the recommended channel priorities. Note that conda-forge has priority over defaults. ::

    name: null
    channels:
    - pytorch
    - conda-forge
    - defaults

Dependencies
^^^^^^^^^^^^

There are a few things worth noting about the dependencies.

1. Even though I have installed the NVIDIA CUDA Toolkit manually I still use Conda to manage the 
   other required CUDA components such as cudnn and nccl (and the optional cupti).
2. I use two meta-pacakges, cxx-compiler and nvcc_linux-64, to make sure that suitable C, and C++ 
   compilers are installed and that the resulting Conda environment is aware of the manually 
   installed CUDA Toolkit.
3. Horovod requires some controller library to coordinate work between the various Horovod 
   processes. Typically this will be some MPI implementation such as OpenMPI. However, rather than 
   specifying the openmpi package directly I instead opt for mpi4py Conda package which provides a 
   cuda-aware build of OpenMPI (assuming it is supported by your hardware).
4. Horovod also support that Gloo collective communications library that can be used in place of 
   MPI. I include cmake in order to insure that the Horovod extensions for Gloo are built.

Below are the core required dependencies. The complete environment.yml file is available on GitHub.::

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
    - pytorch=1.4
    - tensorboard=2.1
    - tensorflow-gpu=2.1
    - torchvision=0.5

The requirements.txt file
-------------------------

The requirements.txt file is where all of the pip dependencies, including Horovod itself, are 
listed for installation. In addition to Horovod I typically will also use pip to install 
JupyterLab extensions to enable GPU and CPU resource monitoring via jupyterlab-nvdashboard and 
Tensorboard support via jupyter-tensorboard.::

    horovod==0.19.*
    jupyterlab-nvdashboard==0.2.*
    jupyter-tensorboard==0.2.*
    # make sure horovod is re-compiled if environment is re-built
    --no-binary=horovod

Note the use of the --no-binary option at the end of the file. Including this option insures that 
Horovod will be re-built whenever the Conda environment is re-built.

The complete requirements.txt file is available on GitHub.

Building Conda environment
--------------------------

After adding any necessary dependencies that should be downloaded via conda to the environment.yml 
file and any dependencies that should be downloaded via pip to the requirements.txt file you 
create the Conda environment in a sub-directory ./env of your project directory by running the 
following commands.::

    export ENV_PREFIX=$PWD/env
    export HOROVOD_CUDA_HOME=$CUDA_HOME
    export HOROVOD_NCCL_HOME=$ENV_PREFIX
    export HOROVOD_GPU_OPERATIONS=NCCL
    conda env create --prefix $ENV_PREFIX --file environment.yml --force

By default Horovod will try and build extensions for all detected frameworks. See the Horovod 
documentation on environment variables for the details on additional environment variables that 
can be set prior to building Horovod.

Once the new environment has been created you can activate the environment with the following 
command.::

    conda activate $ENV_PREFIX

The postBuild file
^^^^^^^^^^^^^^^^^^

If you wish to use any JupyterLab extensions included in the environment.yml and requirements.txt 
files, then you may need to rebuild the JupyterLab application.

For simplicity, I typically include the instructions for re-building JupyterLab in a postBuild 
script. Here is what this script looks like for my Horovod environments.::

    jupyter labextension install --no-build @pyviz/jupyterlab_pyviz
    jupyter labextension install --no-build jupyterlab-nvdashboard 
    jupyter labextension install --no-build jupyterlab_tensorboard
    jupyter serverextension enable jupyterlab_sql --py --sys-prefix
    jupyter lab build

Use the following commands to source the postBuild script.::

    conda activate $ENV_PREFIX # optional if environment already active
    . postBuild

Listing the contents of the Conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To see the full list of packages installed into the environment run the following command.::

    conda activate $ENV_PREFIX # optional if environment already active
    conda list

Verifying the Conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After building the Conda environment you can check that Horovod has been built with support for 
the deep learning frameworks TensorFlow, PyTorch, Apache MXNet, and the contollers MPI and Gloo 
with the following command.::

    conda activate $ENV_PREFIX # optional if environment already active
    horovodrun --check-build

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

I typically wrap these commands into a shell scriptcreate-conda-env.sh. Running the shell script 
will set the Horovod build variables, create the Conda environment, activate the Conda 
environment, and built JupyterLab with any additional extensions.::

    #!/bin/bash --login
    set -e
    export ENV_PREFIX=$PWD/env
    export HOROVOD_CUDA_HOME=$CUDA_HOME
    export HOROVOD_NCCL_HOME=$ENV_PREFIX
    export HOROVOD_GPU_OPERATIONS=NCCL
    conda env create --prefix $ENV_PREFIX --file environment.yml --force
    conda activate $ENV_PREFIX
    . postBuild

I typically put scripts inside a ./bin directory in my project root directory. The script should 
be run from the project root directory as follows.::

    ./bin/create-conda-env.sh # assumes that $CUDA_HOME is set properly

Updating the Conda environment
------------------------------

If you add (remove) dependencies to (from) the environment.yml file or the requirements.txt file 
after the environment has already been created, then you can re-create the environment with the 
following command.::

    conda env create --prefix $ENV_PREFIX --file environment.yml --force

However, whenever I add new dependencies I prefer to re-run the Bash script which will re-build 
both the Conda environment and JupyterLab.::

    ./bin/create-conda-env.sh

Summary
-------

Finding a reproducible process for building Horovod extensions for my deep learning projects was 
tricky. Key to my solution is the use of meta-packages from conda-forge to insure that the 
appropriate compilers are installed and that the resulting Conda environment is aware of the 
system installed NVIDIA CUDA Toolkit. The second key is to use the --no-binary flag in the 
requirements.txt file to insure that Horovod is re-built whenever the Conda environment is 
re-built.

If you like my approach then you can make use of the template repository on GitHub to get started 
with your next Horovod data science project!