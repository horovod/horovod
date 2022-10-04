ARG RAY_DOCKER_VERSION=nightly
FROM rayproject/ray:${RAY_DOCKER_VERSION}-gpu

# Arguments for the build. RAY_DOCKER_VERSION needs to be repeated because
# the first usage only applies to the FROM tag.
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ARG RAY_DOCKER_VERSION=nightly
ARG TENSORFLOW_VERSION=2.9.2
ARG PYTORCH_VERSION=1.12.1+cu113
ARG PYTORCH_LIGHTNING_VERSION=1.5.9
ARG TORCHVISION_VERSION=0.13.1+cu113
ARG MXNET_VERSION=1.9.1

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Download the corresponding key for ubuntu1804.
# This is to fix CI failures caused by the new rotating key mechanism rolled out by Nvidia.
# Refer to https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for more details.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN sudo apt-get update && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y \
        build-essential \
        cmake \
        wget \
        git \
        curl \
        rsync \
        vim \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://download.pytorch.org/whl/${PYTORCH_VERSION/*+/}/torch_stable.html
RUN pip install --no-cache-dir pytorch_lightning==${PYTORCH_LIGHTNING_VERSION}

# Install TensorFlow and Keras
RUN pip install --no-cache-dir future typing packaging
RUN pip install --no-cache-dir \
    tensorflow==${TENSORFLOW_VERSION} \
    keras \
    h5py

USER ray
RUN sudo mkdir -p /horovod /data && sudo chown ray:users /horovod /data
WORKDIR /horovod

# Install Horovod, temporarily using CUDA stubs
COPY --chown=ray:users . .
RUN python setup.py sdist && \
    sudo ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir -v $(ls /horovod/dist/horovod-*.tar.gz)[ray] && \
    horovodrun --check-build && \
    sudo ldconfig

# Check all frameworks are working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
# when running on CPU machine
WORKDIR "/horovod/examples"
RUN sudo ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    python -c "import horovod.tensorflow as hvd; hvd.init()" && \
    python -c "import horovod.torch as hvd; hvd.init()" && \
    sudo ldconfig
