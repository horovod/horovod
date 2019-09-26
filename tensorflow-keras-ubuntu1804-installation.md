**Installing Horovod on Ubuntu 18.04 with Tensorflow-gpu & Keras - Including AWS EC2**

Contrary to how it may seem setting up Horovod on Ubuntu is pretty straight forward as long you make sure you get correct versions of each package and configure the system correctly.

I can't stress enough if you any of the Prerequisite components installed from previous attempts remove them, start fresh - a great number of the errors you can get from mpirun/horovod at the time of writing, don't actually represent the problem which is often mix ups between library versions.

Please note this is a cobbling together of [this](https://lambdalabs.com/blog/horovod-keras-for-multi-gpu-training/), [this](https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/) and [this](http://ubuntuhandbook.org/index.php/2019/03/install-nvidia-418-43-g-sync-support-ubuntu-18-04/) article along with some extra information that isn't immediately apparent and often the cause of problems with your Horovod setup.

To start with you need to get a clean install of Ubuntu 18.04 - on AWS you should select Ubuntu Server 18.04 LTS when picking an AMI.

**Prerequisite**

**Cuda v10.0** - this can be hard work as no matter how you install it you often end up with the latest version, Tensorflow doesn't yet support versions above it, don't worry the answers lay below!

**CUDNN v7.5.0** - this is a straight forward download although you'll need to sign up for NVIDIA's developer account.

**NVIDIA Driver v418.43**


**Lets go!**

First up lets grab Cuda 10.0 as this can be fiddly(unless you're here). NOTE: I've added -y to the install commands for those of us that just want to copy paste, if you get problems just take them out and step through.

```
sudo apt-get install build-essential -y
sudo apt-get install cmake git unzip zip -y
sudo apt-get install python-dev python3-dev python-pip python3-pip -y
sudo apt-get install linux-headers-$(uname -r) -y
sudo apt-get purge nvidia* -y
sudo apt-get autoremove -y
sudo apt-get autoclean -y
sudo rm -rf /usr/local/cuda* 
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update 
sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-10-0 cuda-drivers -y
```

**Reboot!** (No really, sudo reboot now)

```
echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

source ~/.bashrc
sudo ldconfig
```

**CUDNN**

CUDNN can be downloaded from [here](https://developer.nvidia.com/cudnn) note if you're redirected to the NVIDIA home page you either need to sign in, or create an account, come back here click that link again it'll take you to the download page.

Select the one that says 'Download cuDNN v7.6.3 (August 23, 2019), for CUDA 10.0' **make sure it's for version 10.0**

Quick tip if you've had to download it from another PC because you don't have a GUI on your GPU instance(some AWS EC2) you can use scp to transfer it something like:

```
scp -i ~/.ssh/your_key.pem ./cudnn-10.0-linux-x64-v7.6.3.30.tgz ubuntu@1.2.3.4.5:~/
```

Once you've transferred it head back to the file location and then:

```
tar -xzvf cudnn-10.0-linux-x64-v7.6.3.30.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

**NVIDIA Drivers**

Quick an easy one.

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-418 nvidia-settings
```

Thats the basic stuff done, now we need:

**NCCL2**

Follow [this](https://developer.nvidia.com/nccl/nccl-download) link again you might to sign into NVIDIA. Download *"NCCL v2.4.8, for CUDA 10.0, July 31, 2019, O/S agnostic local installer"*

Once you've got it you need to run the following in the download location:

```
tar -vxf ~/Downloads/nccl_2.4.8-1+cuda10.0_x86_64.txz -C ~/Downloads/
sudo cp ~/Downloads/nccl_2.4.8-1+cuda10.0_x86_64/lib/libnccl* /usr/lib/x86_64-linux-gnu/
sudo cp ~/Downloads/nccl_2.4.8-1+cuda10.0_x86_64/include/nccl.h  /usr/include/
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc
source ~/.bashrc
```

**OpenMPI**

First up lets get rid of any existing installs, sometimes Ubuntu has an old version of OpenMPI:

```
sudo mv /usr/bin/mpirun /usr/bin/bk_mpirun
sudo mv /usr/bin/mpirun.openmpi /usr/bin/bk_mpirun.openmpi
```

Then install OpenMPI(this takes a little while):

```
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz -P ~/Downloads
tar -xvf ~/Downloads/openmpi-4.0.1.tar.gz -C ~/Downloads
cd ~/Downloads/openmpi-4.0.1
./configure --prefix=$HOME/openmpi
make -j 8 all
make install

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/openmpi/lib' >> ~/.bashrc
echo 'export PATH=$PATH:~/openmpi/bin' >> ~/.bashrc
source ~/.bashrc
```

**Horovod**

Pretty straight forward:


```
sudo apt install g++-4.8
sudo apt-get install python3-pip
sudo pip3 install virtualenv 
virtualenv -p /usr/bin/python3.6 horovod
activate horovod
pip3 install tensorflow-gpu keras
HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --no-cache-dir horovod
```

At this point if you want to try out a trying script you can launch horovod by doing the following(just to test on one GPU)

```
activate horovod
horovodrun --verbose -np 1 -H localhost:1 python3 train.py
```

This should now fire up and after a little time begin training.

**For multi node training we need to add a few steps.**

First up ALL instances need to be able to connect to ALL other instances via SSH.

A) You need to setup SSH keys so that you can SSH to and from each server you want to use.

***on each host***

B) ssh-keyscan -t rsa,dsa server1 server2 > ~/.ssh/known_hosts  

The reason for step B is because if horovod encounters any of the SSH prompts asking for (yes/no) answers it'll fail.

C) Now most importantly you also need to open TCP communcation between your instances for mpirun to function properly. On AWS you can edit your security group ingress rules and simply add the subnet or various IP's.

[Running with the generated MPI run when you get orted error]
