ARG UBUNTU_VERSION=16.04
FROM ubuntu:${UBUNTU_VERSION}

# Arguments for the build. UBUNTU_VERSION needs to be repeated because
# the first usage only applies to the FROM tag.
ARG UBUNTU_VERSION=16.04
ARG MPI_KIND=OpenMPI
ARG PYTHON_VERSION=2.7
ARG TENSORFLOW_PACKAGE=tensorflow==1.14.0
ARG KERAS_PACKAGE=keras==2.2.4
ARG PYTORCH_PACKAGE=torch==1.1.0
ARG TORCHVISION_PACKAGE=https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp27-cp27mu-linux_x86_64.whl
ARG MXNET_PACKAGE=mxnet==1.4.1
ARG PYSPARK_PACKAGE=pyspark==2.4.0

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install essential packages.
RUN apt-get update -qq
RUN apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        cmake \
        openssh-client \
        git \
        build-essential \
        g++-4.8

# install g++-7
RUN apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update -qq
RUN apt-get install -y --no-install-recommends g++-7

# Install Python.
RUN apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev
RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
        apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi
RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py
RUN pip install -U --force pip setuptools requests pytest

# Install PySpark.
RUN apt install -y openjdk-8-jdk-headless
RUN pip install ${PYSPARK_PACKAGE}

# Install MPI.
RUN if [[ ${MPI_KIND} == "OpenMPI" ]]; then \
        wget -O /tmp/openmpi-3.0.0-bin.tar.gz https://github.com/horovod/horovod/files/1596799/openmpi-3.0.0-bin.tar.gz && \
            cd /usr/local && tar -zxf /tmp/openmpi-3.0.0-bin.tar.gz && ldconfig && \
            echo "mpirun -allow-run-as-root -np 2 -H localhost:2 -bind-to none -map-by slot -mca mpi_abort_print_stack 1" > /mpirun_command; \
    elif [[ ${MPI_KIND} == "MLSL" ]]; then \
        wget -O /tmp/l_mlsl_2018.3.008.tgz https://github.com/intel/MLSL/releases/download/IntelMLSL-v2018.3-Preview/l_mlsl_2018.3.008.tgz && \
            cd /tmp && tar -zxf /tmp/l_mlsl_2018.3.008.tgz && \
            /tmp/l_mlsl_2018.3.008/install.sh -s -d /usr/local/mlsl && \
            wget https://raw.githubusercontent.com/intel/MLSL/master/mpirt_2019/include/mpi.h -P /usr/local/mlsl/intel64/include && \
            wget https://raw.githubusercontent.com/intel/MLSL/master/mpirt_2019/include/mpio.h -P /usr/local/mlsl/intel64/include && \
            wget https://raw.githubusercontent.com/intel/MLSL/master/mpirt_2019/include/mpicxx.h -P /usr/local/mlsl/intel64/include && \
            wget https://raw.githubusercontent.com/AlekseyMarchuk/MLSL/master/mpirt_2019/bin/mpicc -P /usr/local/mlsl/intel64/bin && \
            chmod +x /usr/local/mlsl/intel64/bin/mpicc && \
            wget https://raw.githubusercontent.com/AlekseyMarchuk/MLSL/master/mpirt_2019/bin/mpicxx -P /usr/local/mlsl/intel64/bin && \
            chmod +x /usr/local/mlsl/intel64/bin/mpicxx && \
            wget https://raw.githubusercontent.com/AlekseyMarchuk/MLSL/master/mpirt_2019/bin/mpigcc -P /usr/local/mlsl/intel64/bin && \
            chmod +x /usr/local/mlsl/intel64/bin/mpigcc && \
            wget https://raw.githubusercontent.com/AlekseyMarchuk/MLSL/master/mpirt_2019/bin/mpigxx -P /usr/local/mlsl/intel64/bin && \
            chmod +x /usr/local/mlsl/intel64/bin/mpigxx && \
            wget https://raw.githubusercontent.com/AlekseyMarchuk/MLSL/master/mpirt_2019/lib/libmpicxx.so -P /usr/local/mlsl/intel64/lib && \
            chmod +x /usr/local/mlsl/intel64/lib/libmpicxx.so && \
            echo ". /usr/local/mlsl/intel64/bin/mlslvars.sh \"thread\"; \
                   echo \"mpirun is \$(which mpirun)\"; \
                   echo \"this file is \$(cat /mpirun_command_script)\"; \
                   echo \"LD_LIBRARY_PATH is \$(echo \$LD_LIBRARY_PATH)\"; \
                   echo \"mlsl links with \$(ldd /usr/local/mlsl/intel64/lib/libmlsl.so)\"; \
                   mpirun -np 2 -ppn 2 -hosts localhost \$@" > /mpirun_command_script && \
            chmod +x /mpirun_command_script && \
            echo "-L/usr/local/mlsl/intel64/lib/thread -lmpi -I/usr/local/mlsl/intel64/include" > /mpicc_mlsl && \
            chmod +x /mpicc_mlsl && \
            echo "/mpirun_command_script" > /mpirun_command; \
    else \
        apt-get install -y mpich && \
            echo "mpirun -np 2" > /mpirun_command; \
    fi

# Install mpi4py.
RUN if [[ ${MPI_KIND} == "MLSL" ]]; then \
        export I_MPI_ROOT=/usr/local/mlsl; \
        export MPICC=/usr/local/mlsl/intel64/bin/mpicc; \
    fi; \
    pip install mpi4py

### END OF CACHE ###
COPY . /horovod

# Install TensorFlow.
RUN pip install ${TENSORFLOW_PACKAGE}

# Install Keras.
RUN pip install ${KERAS_PACKAGE} h5py scipy pandas
RUN mkdir -p ~/.keras
RUN python -c "from keras.datasets import mnist; mnist.load_data()"

# Install PyTorch.
RUN pip install future typing
RUN if [[ ${PYTORCH_PACKAGE} == "torch-nightly" ]]; then \
        pip install torch_nightly -v -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html; \
    else \
        pip install ${PYTORCH_PACKAGE}; \
    fi
RUN pip install ${TORCHVISION_PACKAGE} Pillow --no-deps

# Install MXNet.
RUN pip install ${MXNET_PACKAGE}

# Install Horovod.
RUN if [[ ${MPI_KIND} == "MLSL" ]]; then \
      if [ -z "${LD_LIBRARY_PATH:-}" ]; then \
          export LD_LIBRARY_PATH=""; \
      fi; \
      if [ -z "${PYTHONPATH:-}" ]; then \
          export PYTHONPATH=""; \
      fi; \
      source /usr/local/mlsl/intel64/bin/mlslvars.sh "thread"; \
      export I_MPI_ROOT=/usr/local/mlsl; \
      export PIP_HOROVOD_MPICXX_SHOW=/usr/local/mlsl/intel64/bin/mpicxx; \
      echo "horovod python setup.py sdist, mpicxx is $(which mpicxx)"; \
      cd /horovod && python setup.py sdist; \
    else \
      cd /horovod && python setup.py sdist; \
    fi

RUN if [[ ${MPI_KIND} == "MLSL" ]]; then \
      if [ -z "${LD_LIBRARY_PATH:-}" ]; then \
          export LD_LIBRARY_PATH=""; \
      fi; \
      if [ -z "${PYTHONPATH:-}" ]; then \
          export PYTHONPATH=""; \
      fi; \
      source /usr/local/mlsl/intel64/bin/mlslvars.sh "thread"; \
      echo "pip install horovod, mpicxx is $(which mpicxx)"; \
      pip install -v /horovod/dist/horovod-*.tar.gz; \
    else \
      pip install -v /horovod/dist/horovod-*.tar.gz; \
    fi

# Hack for compatibility of MNIST example with TensorFlow 1.1.0.
RUN if [[ ${TENSORFLOW_PACKAGE} == "tensorflow==1.1.0" ]]; then \
        sed -i "s/from tensorflow import keras/from tensorflow.contrib import keras/" /horovod/examples/tensorflow_mnist.py; \
    fi

# Hack TensorFlow MNIST example to be smaller.
RUN sed -i "s/last_step=20000/last_step=100/" /horovod/examples/tensorflow_mnist.py

# Hack TensorFlow Eager MNIST example to be smaller.
RUN sed -i "s/dataset.take(20000/dataset.take(100/" /horovod/examples/tensorflow_mnist_eager.py

# Hack TensorFlow 2.0 example to be smaller.
RUN sed -i "s/dataset.take(10000/dataset.take(100/" /horovod/examples/tensorflow2_mnist.py

# Hack Keras MNIST advanced example to be smaller.
RUN sed -i "s/epochs = .*/epochs = 9/" /horovod/examples/keras_mnist_advanced.py
RUN sed -i "s/model.add(Conv2D(32, kernel_size=(3, 3),/model.add(Conv2D(1, kernel_size=(3, 3),/" /horovod/examples/keras_mnist_advanced.py
RUN sed -i "s/model.add(Conv2D(64, (3, 3), activation='relu'))//" /horovod/examples/keras_mnist_advanced.py

# Hack TensorFlow 2.0 Keras MNIST advanced example to be smaller.
RUN sed -i "s/epochs = .*/epochs = 9/" /horovod/examples/tensorflow2_keras_mnist.py
RUN sed -i "s/tf.keras.layers.Conv2D(32, \\[3, 3\\],/tf.keras.layers.Conv2D(1, [3, 3],/" /horovod/examples/tensorflow2_keras_mnist.py
RUN sed -i "s/tf.keras.layers.Conv2D(64, \\[3, 3\\], activation='relu')),//" /horovod/examples/tensorflow2_keras_mnist.py

# Hack PyTorch MNIST example to be smaller.
RUN sed -i "s/'--epochs', type=int, default=10,/'--epochs', type=int, default=2,/" /horovod/examples/pytorch_mnist.py
RUN sed -i "s/self.fc1 = nn.Linear(320, 50)/self.fc1 = nn.Linear(784, 50)/" /horovod/examples/pytorch_mnist.py
RUN sed -i "s/x = F.relu(F.max_pool2d(self.conv1(x), 2))//" /horovod/examples/pytorch_mnist.py
RUN sed -i "s/x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))//" /horovod/examples/pytorch_mnist.py
RUN sed -i "s/x = x.view(-1, 320)/x = x.view(-1, 784)/" /horovod/examples/pytorch_mnist.py
