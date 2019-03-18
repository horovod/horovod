## Running Horovod

The examples below are for Open MPI and use `horovodrun`. Check your MPI documentation for arguments to the `mpirun`
command on your system.

Typically one GPU will be allocated per process, so if a server has 4 GPUs, you would run 4 processes. In `horovodrun`,
the number of processes is specified with the `-np` flag.

1. To run on a machine with 4 GPUs:

```bash
$ horovodrun -np 4 -H localhost:4 python train.py
```

2. To run on 4 machines with 4 GPUs each:

```bash
$ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py
```

### Failures due to SSH issues

The host where `horovodrun` is executed must be able to SSH to all other hosts without any prompts.

If `horovodrun` fails with permission error, verify that you can ssh to every other server without entering a password or
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
