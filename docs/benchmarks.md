## Benchmarks

![128-GPU Benchmark](https://user-images.githubusercontent.com/16640218/31681220-7453e760-b32b-11e7-9ba3-6d01f83b7748.png)

The above benchmark was done on 32 servers with 4 Pascal GPUs each connected by RoCE-capable 25 Gbit/s network. Horovod
achieves 90% scaling efficiency for both Inception V3 and ResNet-101, and 79% scaling efficiency for VGG-16.

To reproduce the benchmarks:

1. Install Horovod using the instructions provided on the [Horovod on GPU](gpus.md) page.

2. Clone branch `horovod_v2` of [https://github.com/alsrgv/benchmarks](https://github.com/alsrgv/benchmarks):

```bash
$ git clone https://github.com/alsrgv/benchmarks
$ cd benchmarks
$ git checkout horovod_v2
```

3. Run the benchmark. Examples below are for Open MPI.

    1. If you have a network with RoCE / InfiniBand (recommended):
    
    ```bash
    $ mpirun -np 16 \
        -bind-to none -oversubscribe \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
        -mca pml ob1 -mca btl_openib_receive_queues P,128,32:P,2048,32:P,12288,32:P,65536,32 \
        -H server1:4,server2:4,server3:4,server4:4 \
        \
        python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
            --model resnet101 \
            --batch_size 64 \
            --variable_update horovod
    ```

    2. If you have a plain TCP network:
    
    ```bash
    $ mpirun -np 16 \
        -bind-to none -oversubscribe \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
        -H server1:4,server2:4,server3:4,server4:4 \
        \
        python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
            --model resnet101 \
            --batch_size 64 \
            --variable_update horovod
    ```

4. At the end of the run, you will see the number of images processed per second:

```
total images/sec: 1656.82
```

### Real data benchmarks

The benchmark instructions above are for the synthetic data benchmark.

To run the benchmark on a real data, you need to download the [ImageNet dataset](http://image-net.org/download-images)
and convert it using the TFRecord [preprocessing script](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_imagenet.sh).

Now, simply add `--data_dir /path/to/imagenet/tfrecords --data_name imagenet --num_batches=2000` to your training command:

1. If you have a network with RoCE / InfiniBand (recommended):

```bash
$ mpirun -np 16 \
    -bind-to none -oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    -mca pml ob1 -mca btl_openib_receive_queues P,128,32:P,2048,32:P,12288,32:P,65536,32 \
    -H server1:4,server2:4,server3:4,server4:4 \
    \
    python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
        --model resnet101 \
        --batch_size 64 \
        --variable_update horovod \
        --data_dir /path/to/imagenet/tfrecords \
        --data_name imagenet \
        --num_batches=2000
```

2. If you have a plain TCP network:

```bash
$ mpirun -np 16 \
    -bind-to none -oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    -H server1:4,server2:4,server3:4,server4:4 \
    \
    python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
        --model resnet101 \
        --batch_size 64 \
        --variable_update horovod \
        --data_dir /path/to/imagenet/tfrecords \
        --data_name imagenet \
        --num_batches=2000
```
