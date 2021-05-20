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

RUN sudo apt-get update && DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    cmake \
    rsync \
    vim

# Install Horovod, temporarily using CUDA stubs
WORKDIR /horovod
COPY . .
RUN sudo python setup.py sdist && \
    sudo ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    sudo bash -c "HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 pip install --no-cache-dir -v $(ls /horovod/dist/horovod-*.tar.gz)[all-frameworks]" && \
    sudo ldconfig

WORKDIR "/horovod/examples"
