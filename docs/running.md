## Running Horovod

The examples below are for Open MPI. Check your MPI documentation for arguments to the `mpirun` command on your system.

Typically one GPU will be allocated per process, so if a server has 4 GPUs, you would run 4 processes. In Open MPI,
the number of processes is specified with the `-np` flag.

Starting with the Open MPI 3, it's important to add the `-bind-to none` and `-map-by slot` arguments. `-bind-to none`
specifies Open MPI to not bind a training process to a single CPU core (which would hurt performance). `-map-by slot`
allows you to have a mixture of different NUMA configurations because the default behavior is to bind to the socket.

`-mca pml ob1` and `-mca btl ^openib` flags force the use of TCP for MPI communication.  This avoids many multiprocessing
issues that Open MPI has with RDMA which typically result in segmentation faults.  Using TCP for MPI does not have
noticeable performance impact since most of the heavy communication is done by NCCL, which will use RDMA via RoCE or
InfiniBand if they're available (see [Horovod on GPU](gpus.md)).  Notable exceptions from this rule are models that heavily
use `hvd.broadcast()` and `hvd.allgather()` operations.  To make those operations use RDMA, read the [Open MPI with RDMA](#open-mpi-with-rdma)
section below.

With the `-x` option you can specify (`-x NCCL_DEBUG=INFO`) or copy (`-x LD_LIBRARY_PATH`) an environment variable to all
the workers.

1. To run on a machine with 4 GPUs:

```bash
$ mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```

2. To run on 4 machines with 4 GPUs each:

```bash
$ mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```

### Open MPI with RDMA

As noted above, using TCP for MPI communication does not have any significant effects on performance in the majority of cases.
Models that make heavy use of `hvd.broadcast()` and `hvd.allgather()` operations are exceptions to that rule.

Default Open MPI `openib` BTL that provides RDMA functionality does not work well with MPI multithreading.  In order to use
RDMA with `openib`, multithreading must be disabled via `-x HOROVOD_MPI_THREADS_DISABLE=1` option.  See the example below:

```bash
$ mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x PATH \
    -mca pml ob1 \
    python train.py
```

Other MPI RDMA implementations may or may not benefit from disabling multithreading, so please consult vendor documentation.

### Hangs due to SSH issues

The host where `mpirun` is executed must be able to SSH to all other hosts without any prompts.

If `mpirun` hangs without any output, verify that you can ssh to every other server without entering a password or
answering questions like this:

```
The authenticity of host '<hostname> (<ip address>)' can't be established.
RSA key fingerprint is xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx.
Are you sure you want to continue connecting (yes/no)?
```

To learn more about setting up passwordless authentication, see [this page](http://www.linuxproblem.org/art_9.html).

To avoid `The authenticity of host '<hostname> (<ip address>)' can't be established` prompts, add all the hosts to
the `~/.ssh/known_hosts` file using `ssh-keyscan`:

```bash
$ ssh-keyscan -t rsa,dsa server1 server2 > ~/.ssh/known_hosts
```

### Hangs due to non-routed network interfaces

Having network interfaces that are not routed can cause Open MPI to hang. An example of such interface is `docker0`.

If you see non-routed interfaces (like `docker0`) in the output of `ifconfig`, you should tell Open MPI and NCCL to not
use them via the `-mca btl_tcp_if_exclude <interface>[,<interface>]` and `NCCL_SOCKET_IFNAME=^<interface>` parameters.

```bash
$ ifconfig
docker0   Link encap:Ethernet  HWaddr 02:42:2d:17:ea:66
          inet addr:172.17.0.1  Bcast:0.0.0.0  Mask:255.255.0.0
          UP BROADCAST MULTICAST  MTU:1500  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)

eth0      Link encap:Ethernet  HWaddr 24:8a:07:b3:7d:8b
          inet addr:10.0.0.1  Bcast:10.0.0.255  Mask:255.255.255.0
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:900002410 errors:0 dropped:405 overruns:0 frame:0
          TX packets:1521598641 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:376184431726 (350.3 GiB)  TX bytes:954933846124 (889.3 GiB)

eth1      Link encap:Ethernet  HWaddr 24:8a:07:b3:7d:8a
          inet addr:192.168.0.1  Bcast:192.168.0.255  Mask:255.255.255.0
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:2410141 errors:0 dropped:0 overruns:0 frame:0
          TX packets:2312177 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:698398061 (666.0 MiB)  TX bytes:458504418 (437.2 MiB)

lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:497075633 errors:0 dropped:0 overruns:0 frame:0
          TX packets:497075633 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1
          RX bytes:72680421398 (67.6 GiB)  TX bytes:72680421398 (67.6 GiB)
```

For example:

```bash
$ mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_SOCKET_IFNAME=^docker0 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    python train.py
```
