.. inclusion-marker-start-do-not-remove

Horovod in Docker
=================

To streamline the installation process on GPU machines, we have published the reference `Dockerfile <https://github.com/horovod/horovod/blob/master/Dockerfile.gpu>`__ so
you can get started with Horovod in minutes. The container includes `Examples <https://github.com/horovod/horovod/tree/master/examples>`__ in the ``/examples``
directory.

Pre-built Docker containers with Horovod are available on `DockerHub <https://hub.docker.com/r/horovod/horovod>`__.


Building
~~~~~~~~
Before building, you can modify ``Dockerfile.gpu`` to your liking, e.g. select a different CUDA, TensorFlow or Python version.

.. code-block:: bash

    $ mkdir horovod-docker-gpu
    $ wget -O horovod-docker-gpu/Dockerfile https://raw.githubusercontent.com/horovod/horovod/master/Dockerfile.gpu
    $ docker build -t horovod:latest horovod-docker-gpu

For users without GPUs available in their environments, we've also published a `CPU Dockerfile <https://github.com/horovod/horovod/blob/master/Dockerfile.cpu>`__
you can build and run similarly.


Running on a single machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the container is built, run it using `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__.

**Note**: You can replace ``horovod:latest`` with the `specific <https://hub.docker.com/r/horovod/horovod/tags>`__ pre-build
Docker container with Horovod instead of building it by yourself.

.. code-block:: bash

    $ nvidia-docker run -it horovod:latest
    root@c278c88dd552:/examples# horovodrun -np 4 -H localhost:4 python keras_mnist_advanced.py


If you don't run your container in privileged mode, you may see the following message:

.. code-block:: bash

    [a8c9914754d2:00040] Read -1, expected 131072, errno = 1


You can ignore this message.


Running on multiple machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we describe a simple example involving a shared filesystem ``/mnt/share`` using a common port number ``12345`` for the SSH
daemon that will be run on all the containers. ``/mnt/share/ssh`` would contain a typical ``id_rsa`` and ``authorized_keys``
pair that allows `passwordless authentication <http://www.linuxproblem.org/art_9.html>`__.

**Note**: These are not hard requirements but they make the example more concise. A shared filesystem can be replaced by ``rsyncing``
SSH configuration and code across machines, and a common SSH port can be replaced by machine-specific ports
defined in ``/root/.ssh/ssh_config`` file.

Primary worker:

.. code-block:: bash

    host1$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest
    root@c278c88dd552:/examples# horovodrun -np 16 -H host1:4,host2:4,host3:4,host4:4 -p 12345 python keras_mnist_advanced.py


Secondary workers:

.. code-block:: bash

    host2$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest \
        bash -c "/usr/sbin/sshd -p 12345; sleep infinity"


.. code-block:: bash

    host3$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest \
        bash -c "/usr/sbin/sshd -p 12345; sleep infinity"


.. code-block:: bash

    host4$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest \
        bash -c "/usr/sbin/sshd -p 12345; sleep infinity"


Adding Mellanox RDMA support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have Mellanox NICs, we recommend that you mount your Mellanox devices (``/dev/infiniband``) in the container
and enable the IPC_LOCK capability for memory registration:

.. code-block:: bash

   $ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh --cap-add=IPC_LOCK --device=/dev/infiniband horovod:latest 
   root@c278c88dd552:/examples# ...


You need to specify these additional configuration options on primary and secondary workers.


.. inclusion-marker-end-do-not-remove
