ARG RAY_DOCKER_VERSION=nightly
FROM rayproject/ray:${RAY_DOCKER_VERSION}-gpu

# Arguments for the build. RAY_DOCKER_VERSION needs to be repeated because
# the first usage only applies to the FROM tag.
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ARG RAY_DOCKER_VERSION=nightly
ARG TENSORFLOW_VERSION=2.5.0
ARG PYTORCH_VERSION=1.8.1+cu111
ARG PYTORCH_LIGHTNING_VERSION=1.2.9
ARG TORCHVISION_VERSION=0.9.1+cu111

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN sudo apt-get update && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y \
        build-essential \
        wget \
        git \
        gpg \
        curl \
        rsync \
        vim \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# Install recent CMake.
RUN pip install --no-cache-dir -U cmake~=3.13.0

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
RUN sudo mkdir -p /horovod && sudo chown ray:users /horovod
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
