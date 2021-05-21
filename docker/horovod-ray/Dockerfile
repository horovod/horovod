#
# Ludwig Docker image with Ray nightly support and full dependencies including:
#   text features
#   image features
#   audio features
#   visualizations
#   hyperparameter optimization
#   distributed training
#   model serving
#

FROM rayproject/ray:nightly-gpu

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=2.5.0
ENV PYTORCH_VERSION=1.8.1+cu111
ENV PYTORCH_LIGHTNING_VERSION=1.2.9
ENV TORCHVISION_VERSION=0.9.1+cu111

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN sudo apt-get update && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    cmake \
    rsync \
    vim

# Install TensorFlow and Keras
RUN pip install future typing packaging
RUN pip install tensorflow==${TENSORFLOW_VERSION} \
                keras \
                h5py

# Install PyTorch
RUN pip install \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://download.pytorch.org/whl/${PYTORCH_VERSION/*+/}/torch_stable.html
RUN pip install pytorch_lightning==${PYTORCH_LIGHTNING_VERSION}

USER ray
RUN mkdir -p /horovod
WORKDIR /horovod

# Install Horovod, temporarily using CUDA stubs
COPY --chown=ray:users . .
RUN python setup.py sdist && \
    sudo ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir -v $(ls /horovod/dist/horovod-*.tar.gz)[ray] && \
    horovodrun --check-build && \
    sudo ldconfig

WORKDIR "/horovod/examples"
