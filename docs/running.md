## Running Horovod

The examples below are for Open MPI. Check your MPI documentation for arguments to the `mpirun` command on your system.

Typically one GPU will be allocated per process, so if a server has 4 GPUs, you would run 4 processes. In Open MPI,
the number of processes is specified with the `-np` flag.

Starting with the Open MPI 3, it's important to add the `-bind-to none` and `-oversubscribe` arguments. `-bind-to none`
specifies Open MPI to not bind a training process to a single CPU core (which would hurt performance). `-oversubscribe`
enables you to to run multiple training processes on a single core.

With the `-x` option you can specify (`-x NCCL_DEBUG=INFO`) or copy (`-x LD_LIBRARY_PATH`) an environment variable to all
the workers.

1. To run on a machine with 4 GPUs:

```bash
$ mpirun -np 4 \
    -bind-to none -oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    python train.py
```

2. To run on 4 machines with 4 GPUs each:

```bash
$ mpirun -np 16 \
    -bind-to none -oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    -H server1:4,server2:4,server3:4,server4:4 \
    python train.py
```

3. If you have RoCE or InfiniBand, we found that the `pml` and `btl_openib_receive_queues` parameters improve
performance a lot:

```bash
$ mpirun -np 16 \
    -bind-to none -oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    -mca pml ob1 -mca btl_openib_receive_queues P,128,32:P,2048,32:P,12288,32:P,65536,32 \
    -H server1:4,server2:4,server3:4,server4:4 \
    python train.py
```

### Hangs due to non-routed network interfaces

Having network interfaces that are not routed can cause Open MPI to hang. An example of such interface is `docker0`.

If you see non-routed interfaces (like `docker0`) in the output of `ifconfig`, you should tell Open MPI to not use them
via the `-mca btl_tcp_if_exclude <interface>[,<interface>]` parameter.

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
$ mpirun -mca btl_tcp_if_exclude docker0 -np 16 -bind-to none -oversubscribe -x LD_LIBRARY_PATH -H server1:4,server2:4,server3:4,server4:4 python train.py
```
