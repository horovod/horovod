## Horovod in Docker

To streamline the installation process on GPU machines, we have published the reference [Dockerfile](../Dockerfile) so
you can get started with Horovod in minutes. The container includes [Examples](../examples) in the `/examples`
directory.

Pre-built docker containers with Horovod are available on [DockerHub](https://hub.docker.com/r/uber/horovod).

### Building

Before building, you can modify `Dockerfile` to your liking, e.g. select a different CUDA, TensorFlow or Python version.

```bash
$ mkdir horovod-docker
$ wget -O horovod-docker/Dockerfile https://raw.githubusercontent.com/uber/horovod/master/Dockerfile
$ docker build -t horovod:latest horovod-docker
```

### Running on a single machine

After the container is built, run it using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

```bash
$ nvidia-docker run -it horovod:latest
root@c278c88dd552:/examples# mpirun -np 4 -H localhost:4 python keras_mnist_advanced.py
```

This command does not have options recommended in other parts of the documentation. 
`-bind-to none -map-by slot -x NCCL_DEBUG=INFO` options are already set by default in the Docker container so
you don't need to repeat them in the command.  Options `-x LD_LIBRARY_PATH -x PATH` are not necessary because we assume
that all the software is installed in the default system location in this Docker image.

If you don't run your container in privileged mode, you may see the following message:

```
[a8c9914754d2:00040] Read -1, expected 131072, errno = 1
```

You can ignore this message.

### Running on multiple machines

Here we describe a simple example involving a shared filesystem `/mnt/share` using a common port number `12345` for the SSH
daemon that will be run on all the containers. `/mnt/share/ssh` would contain a typical `id_rsa` and `authorized_keys`
pair that allows [passwordless authentication](http://www.linuxproblem.org/art_9.html).

**Note**: These are not hard requirements but they make the example more concise. A shared filesystem can be replaced by
`rsync`ing SSH configuration and code across machines, and a common SSH port can be replaced by machine-specific ports
defined in `/root/.ssh/ssh_config` file.

Primary worker:

```bash
host1$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest
root@c278c88dd552:/examples# mpirun -np 16 -H host1:4,host2:4,host3:4,host4:4 \
    -mca plm_rsh_args "-p 12345" python keras_mnist_advanced.py
```

Secondary workers:

```bash
host2$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest \
    bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
```

```bash
host3$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest \
    bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
```

```bash
host4$ nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh horovod:latest \
    bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
```
