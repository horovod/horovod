ARG CUDA_DOCKER_VERSION=11.2.2-devel-ubuntu20.04
FROM nvidia/cuda:${CUDA_DOCKER_VERSION}

# Arguments for the build. CUDA_DOCKER_VERSION needs to be repeated because
# the first usage only applies to the FROM tag.
ARG CUDA_DOCKER_VERSION=11.2.2-devel-ubuntu20.04
ARG CUDNN_VERSION=8.1.1.33-1+cuda11.2
ARG NCCL_VERSION=2.8.4-1+cuda11.2
ARG MPI_KIND=OpenMPI
ARG PYTHON_VERSION=3.8
ARG GPP_VERSION=7
ARG TENSORFLOW_PACKAGE=tensorflow-gpu==2.8.0
ARG KERAS_PACKAGE=keras==2.8.0
ARG PYTORCH_PACKAGE=torch==1.8.1+cu111
ARG PYTORCH_LIGHTNING_PACKAGE=pytorch_lightning==1.5.9
ARG TORCHVISION_PACKAGE=torchvision==0.9.1+cu111
ARG MXNET_PACKAGE=mxnet-cu112==1.8.0.post0
ARG PYSPARK_PACKAGE=pyspark==3.2.1
# if SPARK_PACKAGE is set, installs Spark into /spark from the tgz archive
# if SPARK_PACKAGE is a preview version, installs PySpark from the tgz archive
# see https://archive.apache.org/dist/spark/ for available packages, version must match PYSPARK_PACKAGE
ARG SPARK_PACKAGE=spark-3.2.1/spark-3.2.1-bin-hadoop2.7.tgz
ARG HOROVOD_BUILD_FLAGS="HOROVOD_GPU_OPERATIONS=NCCL"
ARG HOROVOD_MIXED_INSTALL=0

ENV PATH=/root/miniconda3/bin:$PATH

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Extract ubuntu distribution version and download the corresponding key.
# This is to fix CI failures caused by the new rotating key mechanism rolled out by Nvidia.
# Refer to https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for more details.
#RUN DIST=$(echo ${CUDA_DOCKER_VERSION#*ubuntu} | sed 's/\.//'); \
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${DIST}/x86_64/3bf863cc.pub && \
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${DIST}/x86_64/7fa2af80.pub

# Prepare to install specific g++ versions
RUN apt-get update -qq && apt-get install -y --no-install-recommends software-properties-common && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:ubuntu-toolchain-r/test

# Install essential packages.
RUN CUDNN_MAJOR=$(cut -d '.' -f 1 <<< "${CUDNN_VERSION}"); \
    apt-get update -qq && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        wget \
        ca-certificates \
        cmake \
        openssh-client \
        openssh-server \
        git \
        build-essential \
        g++-${GPP_VERSION} \
        moreutils \
        libcudnn${CUDNN_MAJOR}=${CUDNN_VERSION} && \
    rm -rf /var/lib/apt/lists/*

# setup ssh service
RUN ssh-keygen -f /root/.ssh/id_rsa -q -N ''
RUN cp -v /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys

# install miniconda Python environment
RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b && \
    rm miniconda.sh && \
    # Source conda.sh for all login and interactive shells.
    ln -s /root/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /etc/profile.d/conda.sh" >> ~/.bashrc && \
    # Set always_yes for non-interactive shells.
    conda config --system --set always_yes True && \
    # Install NVTabular
    conda install -y -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=${PYTHON_VERSION} cudatoolkit=${CUDA_DOCKER_VERSION:0:4} && \
    conda clean --all

RUN pip install --no-cache-dir -U --force requests pytest mock pytest-forked parameterized

# Add launch helper scripts
RUN echo "env SPARK_HOME=/spark SPARK_DRIVER_MEM=512m PYSPARK_PYTHON=/usr/bin/python${PYTHON_VERSION} PYSPARK_DRIVER_PYTHON=/usr/bin/python${PYTHON_VERSION} \"\$@\"" > /spark_env.sh
RUN echo "/spark_env.sh pytest -v --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.\$1.\${HOROVOD_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK}}}.\$2.xml \${@:2}" > /pytest.sh
RUN echo "/spark_env.sh pytest -v --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.\$1.standalone.\$2.xml --forked \${@:2}" > /pytest_standalone.sh
RUN chmod a+x /spark_env.sh
RUN chmod a+x /pytest.sh
RUN chmod a+x /pytest_standalone.sh

# Install Spark stand-alone cluster.
RUN if [[ -n ${SPARK_PACKAGE} ]]; then \
        wget --progress=dot:giga "https://www.apache.org/dyn/closer.lua/spark/${SPARK_PACKAGE}?action=download" -O - | tar -xzC /tmp; \
        archive=$(basename "${SPARK_PACKAGE}") bash -c "mv -v /tmp/\${archive/%.tgz/} /spark"; \
    fi

# Install PySpark.
RUN apt-get update -qq && apt install -y openjdk-8-jdk-headless && rm -rf /var/lib/apt/lists/*
RUN if [[ ${SPARK_PACKAGE} != *"-preview"* ]]; then \
        pip install --no-cache-dir ${PYSPARK_PACKAGE}; \
    else \
        apt-get update -qq && apt-get install pandoc; \
        pip install --no-cache-dir pypandoc; \
        (cd /spark/python && python setup.py sdist && pip install --no-cache-dir dist/pyspark-*.tar.gz && rm dist/pyspark-*); \
    rm -rf /var/lib/apt/lists/*; \
    fi

# Install Ray.
RUN pip install --no-cache-dir ray==1.3.0

# Install MPI.
RUN if [[ ${MPI_KIND} == "OpenMPI" ]]; then \
        wget --progress=dot:mega -O /tmp/openmpi-3.0.0-bin.tar.gz https://github.com/horovod/horovod/files/1596799/openmpi-3.0.0-bin.tar.gz && \
            cd /usr/local && tar -zxf /tmp/openmpi-3.0.0-bin.tar.gz && ldconfig && \
            echo "mpirun -allow-run-as-root -np 2 -H localhost:2 -bind-to none -map-by slot -mca mpi_abort_print_stack 1" > /mpirun_command; \
    elif [[ ${MPI_KIND} == "MPICH" ]]; then \
        apt-get update -qq && apt-get install -y mpich && \
            echo "mpirun -np 2" > /mpirun_command; \
    fi

# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf

# Install mpi4py.
# This requires SETUPTOOLS_USE_DISTUTILS=stdlib as with setuptools>=60.1.0 installing mpi4py broke
# https://github.com/mpi4py/mpi4py/issues/157#issuecomment-1001022274
RUN if [[ ${MPI_KIND} != "None" ]]; then \
        SETUPTOOLS_USE_DISTUTILS=stdlib pip install --no-cache-dir mpi4py; \
    fi

# Install TensorFlow and Keras (releases).
# Pin scipy!=1.4.0: https://github.com/scipy/scipy/issues/11237
# Pin protobuf<4 for tensorflow: https://github.com/tensorflow/tensorflow/issues/56815
RUN if [[ ${TENSORFLOW_PACKAGE} != "tf-nightly-gpu" ]]; then \
        pip install --no-cache-dir ${TENSORFLOW_PACKAGE} "protobuf<4"; \
        if [[ ${KERAS_PACKAGE} != "None" ]]; then \
            pip uninstall -y keras; \
            pip install --no-cache-dir ${KERAS_PACKAGE} "scipy!=1.4.0" "pandas<1.1.0"; \
        fi; \
        mkdir -p ~/.keras; \
        ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs; \
        python -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data()"; \
        ldconfig; \
    fi

# Pin h5py < 3 for tensorflow: https://github.com/tensorflow/tensorflow/issues/44467
RUN pip install 'h5py<3.0' --force-reinstall

# Install PyTorch (releases).
# Pin Pillow<7.0 for torchvision < 0.5.0: https://github.com/pytorch/vision/issues/1718
# Pin Pillow!=8.3.0 for torchvision: https://github.com/pytorch/vision/issues/4146
RUN if [[ ${PYTORCH_PACKAGE} != "torch-nightly-cu"* ]]; then \
        pip install --no-cache-dir ${PYTORCH_PACKAGE} ${TORCHVISION_PACKAGE} -f https://download.pytorch.org/whl/${PYTORCH_PACKAGE/*+/}/torch_stable.html; \
        if [[ "${TORCHVISION_PACKAGE/%+*/}" == torchvision==0.[1234].* ]]; then \
            pip install --no-cache-dir "Pillow<7.0" --no-deps; \
        else \
            pip install --no-cache-dir "Pillow!=8.3.0" --no-deps; \
        fi; \
    fi
RUN pip install ${PYTORCH_LIGHTNING_PACKAGE}

# Install MXNet (releases).
RUN if [[ ${MXNET_PACKAGE} != "mxnet-nightly-cu"* ]]; then \
        pip install --no-cache-dir ${MXNET_PACKAGE} ; \
    fi

# Prefetch Spark MNIST dataset.
RUN mkdir -p /work /data && wget --progress=dot:mega https://horovod-datasets.s3.amazonaws.com/mnist.bz2 -O /data/mnist.bz2

# Prefetch Spark Rossmann dataset.
RUN mkdir -p /data && wget --progress=dot:mega https://horovod-datasets.s3.amazonaws.com/rossmann.tgz -O - | tar -xzC /data

# Prefetch PyTorch datasets.
RUN wget --progress=dot:mega https://horovod-datasets.s3.amazonaws.com/pytorch_datasets.tgz -O - | tar -xzC /data

# Update pip dependencies for nvtabular and apply patch per: https://github.com/NVIDIA-Merlin/NVTabular/pull/1587
RUN CUDA_VER=$(echo ${CUDA_DOCKER_VERSION:0:4} | sed 's/\.//'); \
    pip uninstall -y cupy && pip install --no-cache-dir cupy-cuda${CUDA_VER} "numpy<=1.22" petastorm && \
    sed -i 's/warnings.warn(e)/warnings.warn(str(e))/' /root/miniconda3/lib/python3.8/site-packages/nvtabular/loader/tf_utils.py

### END OF CACHE ###
COPY . /horovod

# Install nightly packages here so they do not get cached

# Install TensorFlow and Keras (nightly).
# Pin scipy!=1.4.0: https://github.com/scipy/scipy/issues/11237
RUN if [[ ${TENSORFLOW_PACKAGE} == "tf-nightly-gpu" ]]; then \
        pip install --no-cache-dir ${TENSORFLOW_PACKAGE}; \
        if [[ ${KERAS_PACKAGE} != "None" ]]; then \
            pip uninstall -y keras-nightly; \
            pip install --no-cache-dir ${KERAS_PACKAGE} "scipy!=1.4.0" "pandas<1.1.0"; \
        fi; \
        mkdir -p ~/.keras; \
        ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs; \
        python -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data()"; \
        ldconfig; \
    fi

# Install PyTorch (nightly).
# Pin Pillow!=8.3.0 for torchvision: https://github.com/pytorch/vision/issues/4146
RUN if [[ ${PYTORCH_PACKAGE} == "torch-nightly-cu"* ]]; then \
        pip install --no-cache-dir --pre torch ${TORCHVISION_PACKAGE} -f https://download.pytorch.org/whl/nightly/${PYTORCH_PACKAGE/#torch-nightly-/}/torch_nightly.html; \
        pip install --no-cache-dir "Pillow!=8.3.0" --no-deps; \
    fi

# Install MXNet (nightly).
RUN if [[ ${MXNET_PACKAGE} == "mxnet-nightly-cu"* ]]; then \
        pip install --no-cache-dir --pre ${MXNET_PACKAGE/-nightly/} -f https://dist.mxnet.io/python/${MXNET_PACKAGE/#mxnet-nightly-/}; \
    fi

# Install Horovod.
RUN cd /horovod && \
    python setup.py sdist && \
    ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    bash -c "${HOROVOD_BUILD_FLAGS} HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 pip install --no-cache-dir -v $(ls /horovod/dist/horovod-*.tar.gz)[spark,ray]" && \
    ldconfig

# Show the effective python package version to easily spot version differences
RUN pip list --format=freeze | sort

# Export HOROVOD_MIXED_INSTALL
ENV HOROVOD_MIXED_INSTALL=${HOROVOD_MIXED_INSTALL}

