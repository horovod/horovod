# Launcher-only Horovod image (CPU).
#
# This image packages the `horovodrun` / `horovod.runner` process launcher with
# Open MPI and passwordless SSH so it can launch single-node and multi-node jobs.
# The TensorFlow/PyTorch/MXNet integrations and the C++ allreduce core have been
# removed from Horovod, so there is nothing to compile; distributed communication
# in the launched processes is expected to be handled by the user's framework of
# choice (e.g. PyTorch `torch.distributed`). Install your own framework packages on
# top of this image as needed.
FROM ubuntu:20.04

ARG PYTHON_VERSION=3.8

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        openssh-client \
        openssh-server \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Open MPI
RUN wget --progress=dot:mega -O /tmp/openmpi-4.1.4-bin.tar.gz https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz && \
    cd /tmp && tar -zxf /tmp/openmpi-4.1.4-bin.tar.gz && \
    mkdir openmpi-4.1.4/build && cd openmpi-4.1.4/build && ../configure --prefix=/usr/local && \
    make -j all && make install && ldconfig && \
    mpirun --version

# Allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install the Horovod launcher. It is pure Python, so no compiler or CMake is needed.
WORKDIR /horovod
COPY . .
RUN pip install --no-cache-dir . && \
    horovodrun --check-build
