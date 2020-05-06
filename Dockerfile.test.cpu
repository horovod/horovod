ARG UBUNTU_VERSION=16.04
FROM ubuntu:${UBUNTU_VERSION}

# Arguments for the build. UBUNTU_VERSION needs to be repeated because
# the first usage only applies to the FROM tag.
ARG UBUNTU_VERSION=16.04
ARG MPI_KIND=OpenMPI
ARG PYTHON_VERSION=2.7
ARG TENSORFLOW_PACKAGE=tensorflow==1.14.0
ARG KERAS_PACKAGE=keras==2.2.4
ARG PYTORCH_PACKAGE=torch==1.2.0+cpu
ARG TORCHVISION_PACKAGE=torchvision==0.4.0+cpu
ARG MXNET_PACKAGE=mxnet==1.5.0
ARG PYSPARK_PACKAGE=pyspark==2.4.0

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install essential packages.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
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
RUN apt-get update -qq && apt-get install -y --no-install-recommends g++-7

# Install Python.
RUN apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev
RUN if [[ "${PYTHON_VERSION}" == "3."* ]]; then \
        apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi
RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py
RUN pip install -U --force pip 'setuptools<46.0.0' requests pytest mock pytest-forked

# Install PySpark.
RUN apt-get update -qq && apt install -y openjdk-8-jdk-headless
RUN pip install ${PYSPARK_PACKAGE}
RUN if [[ "${PYTHON_VERSION}" == "2."* ]]; then \
    pip install 'pyarrow<0.16.0'; \
fi

# Install MPI.
RUN if [[ ${MPI_KIND} == "OpenMPI" ]]; then \
        wget -O /tmp/openmpi-3.0.0-bin.tar.gz https://github.com/horovod/horovod/files/1596799/openmpi-3.0.0-bin.tar.gz && \
            cd /usr/local && tar -zxf /tmp/openmpi-3.0.0-bin.tar.gz && ldconfig && \
            echo "mpirun -allow-run-as-root -np 2 -H localhost:2 -bind-to none -map-by slot -mca mpi_abort_print_stack 1" > /mpirun_command; \
    elif [[ ${MPI_KIND} == "ONECCL" ]]; then \
        wget -O /tmp/oneccl.tar.gz https://github.com/oneapi-src/oneCCL/archive/master.tar.gz && \
            cd /tmp && tar -zxf oneccl.tar.gz && \
            mkdir oneCCL-master/build && cd oneCCL-master/build && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/oneccl -DCMAKE_BUILD_TYPE=Release && make -j install && \
            cp /tmp/oneCCL-master/mpi/include/*.h /usr/local/oneccl/include && \
            cp /tmp/oneCCL-master/mpi/bin/mpicc /usr/local/oneccl/bin && \
            chmod +x /usr/local/oneccl/bin/mpicc && \
            cp /tmp/oneCCL-master/mpi/bin/mpicxx /usr/local/oneccl/bin && \
            chmod +x /usr/local/oneccl/bin/mpicxx && \
            cp /tmp/oneCCL-master/mpi/bin/mpigcc /usr/local/oneccl/bin && \
            chmod +x /usr/local/oneccl/bin/mpigcc && \
            cp /tmp/oneCCL-master/mpi/bin/mpigxx /usr/local/oneccl/bin && \
            chmod +x /usr/local/oneccl/bin/mpigxx && \
            cp /tmp/oneCCL-master/mpi/lib/libmpicxx.so /usr/local/oneccl/lib && \
            chmod +x /usr/local/oneccl/lib/libmpicxx.so && \
            sed -i 's/if \[ -z \"\${I_MPI_ROOT}\" \]/if [ -z \"${I_MPI_ROOT:-}\" ]/g' /usr/local/oneccl/env/setvars.sh && \
            sed -i 's/ \$1/ \${1:-}/g' /usr/local/oneccl/env/setvars.sh && \
            echo ". /usr/local/oneccl/env/setvars.sh" > /oneccl_env && \
            chmod +x /oneccl_env && \
            echo "export CCL_ATL_TRANSPORT=ofi; export CCL_ATL_SHM=0; \
                  echo \"\$(env)\"; \
                  echo \"mpirun is \$(which mpirun)\"; \
                  echo \"LD_LIBRARY_PATH is \$(echo \$LD_LIBRARY_PATH)\"; \
                  echo \"oneCCL links with \$(ldd /usr/local/oneccl/lib/libccl.so)\"; \
                  mpirun -np 2 -hosts localhost \$@" > /mpirun_command_ofi && \
            chmod +x /mpirun_command_ofi && \
            cp /mpirun_command_ofi /mpirun_command_mpi && \
            sed -i 's/export CCL_ATL_TRANSPORT=ofi;//g' /mpirun_command_mpi && \
            sed -i 's/export CCL_ATL_SHM=0;//g' /mpirun_command_mpi && \
            echo "-L/usr/local/oneccl/lib -lmpi -I/usr/local/oneccl/include" > /mpicc_oneccl && \
            chmod +x /mpicc_oneccl && \
            echo "/mpirun_command_mpi" > /mpirun_command; \
    elif [[ ${MPI_KIND} == "MPICH" ]]; then \
        apt-get install -y mpich && \
            echo "mpirun -np 2" > /mpirun_command; \
    fi

# Install mpi4py.
RUN if [[ ${MPI_KIND} != "None" ]]; then \
        if [[ ${MPI_KIND} == "ONECCL" ]]; then \
            export I_MPI_ROOT=/usr/local/oneccl; \
            export MPICC=/usr/local/oneccl/bin/mpicc; \
        fi; \
        pip install mpi4py; \
    fi

### END OF CACHE ###
COPY . /horovod

# Install TensorFlow.
RUN pip install ${TENSORFLOW_PACKAGE}

# Install Keras.
# Pin scipy<1.4.0: https://github.com/scipy/scipy/issues/11237
RUN pip install ${KERAS_PACKAGE} h5py "scipy<1.4.0" pandas
RUN mkdir -p ~/.keras
RUN python -c "from keras.datasets import mnist; mnist.load_data()"

# Install PyTorch.
RUN pip install future typing
RUN if [[ ${PYTORCH_PACKAGE} == "torch-nightly" ]]; then \
        pip install --pre torch ${TORCHVISION_PACKAGE} -v -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html; \
    else \
        pip install ${PYTORCH_PACKAGE} ${TORCHVISION_PACKAGE} -f https://download.pytorch.org/whl/torch_stable.html; \
    fi
# Pin Pillow<7.0: https://github.com/pytorch/vision/issues/1718
RUN pip install "Pillow<7.0" --no-deps

# Install MXNet.
RUN if [[ ${MXNET_PACKAGE} == "mxnet-nightly" && ${PYTHON_VERSION} != "2.7" ]]; then \
        pip install --pre mxnet-mkl -f https://dist.mxnet.io/python/all; \
    else \
        pip install ${MXNET_PACKAGE} ; \
    fi

# Install Horovod.
RUN if [[ ${MPI_KIND} == "ONECCL" ]]; then \
      if [ -z "${LD_LIBRARY_PATH:-}" ]; then \
          export LD_LIBRARY_PATH=""; \
      fi; \
      if [ -z "${PYTHONPATH:-}" ]; then \
          export PYTHONPATH=""; \
      fi; \
      . /usr/local/oneccl/env/setvars.sh; \
      export I_MPI_ROOT=/usr/local/oneccl; \
      export PIP_HOROVOD_MPICXX_SHOW=/usr/local/oneccl/bin/mpicxx; \
      echo "horovod python setup.py sdist, mpicxx is $(which mpicxx)"; \
      cd /horovod && HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 python setup.py sdist; \
    else \
      cd /horovod && HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 python setup.py sdist; \
    fi

RUN if [[ ${MPI_KIND} == "ONECCL" ]]; then \
      if [ -z "${LD_LIBRARY_PATH:-}" ]; then \
          export LD_LIBRARY_PATH=""; \
      fi; \
      if [ -z "${PYTHONPATH:-}" ]; then \
          export PYTHONPATH=""; \
      fi; \
      . /usr/local/oneccl/env/setvars.sh; \
      echo "pip install horovod, mpicxx is $(which mpicxx)"; \
      pip install -v $(ls /horovod/dist/horovod-*.tar.gz)[spark]; \
    else \
      pip install -v $(ls /horovod/dist/horovod-*.tar.gz)[spark]; \
    fi

# Prefetch Spark MNIST dataset.
RUN mkdir -p /work
RUN mkdir -p /data && wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 -O /data/mnist.bz2

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
RUN sed -i "s/'--epochs', type=int, default=24,/'--epochs', type=int, default=9,/" /horovod/examples/keras_mnist_advanced.py
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

# Prefetch Spark Rossmann dataset.
RUN mkdir -p /work
RUN mkdir -p /data && wget http://files.fast.ai/part2/lesson14/rossmann.tgz -O - | tar -xzC /data

# Hack Keras Spark Rossmann Run example to be smaller.
RUN sed -i "s/x = Dense(1000,/x = Dense(100,/g" /horovod/examples/keras_spark_rossmann_run.py
RUN sed -i "s/x = Dense(500,/x = Dense(50,/g" /horovod/examples/keras_spark_rossmann_run.py

# Hack Keras Spark Rossmann Estimator example to be smaller.
RUN sed -i "s/x = Dense(1000,/x = Dense(100,/g" /horovod/examples/keras_spark_rossmann_estimator.py
RUN sed -i "s/x = Dense(500,/x = Dense(50,/g" /horovod/examples/keras_spark_rossmann_estimator.py
