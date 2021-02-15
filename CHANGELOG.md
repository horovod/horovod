# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - YYYY-MM-DD

### Added

### Changed

### Deprecated

### Removed

### Fixed

## [v0.21.3] - 2021-02-15

### Added

- Add `groups` parameter in `DistributedOptimizer` for custom allreduce groups. ([#2523](https://github.com/horovod/horovod/pull/2523))

### Removed

- Removed `num_groups` parameter in `DistributedOptimizer`, replaced with `groups`. ([#2523](https://github.com/horovod/horovod/pull/2523))

### Fixed

- Fixed worker desynchronization deadlock issue in TensorFlow 2.4. ([#2647](https://github.com/horovod/horovod/pull/2647))

- Deduped Keras `LearningRateWarmupCallback` log after gradual learning rate warmup. ([#2661](https://github.com/horovod/horovod/pull/2661))

## [v0.21.2] - 2021-02-08

### Added

- Added support for Intel(R) MPI in horovodrun. ([#2374](https://github.com/horovod/horovod/pull/2374))

- Add support for callbacks in Ray Elastic Executor. ([#2639](https://github.com/horovod/horovod/pull/2639))

- Added forwarding of stdout/stderr captured to driver over Gloo. ([#2646](https://github.com/horovod/horovod/pull/2646))

### Fixed

- Fixed broadcast_optimizer_state to handle NoneType params for PyTorch 1.8. ([#2624](https://github.com/horovod/horovod/pull/2624))

- Fixed `local_rank` support for Ray. ([#2596](https://github.com/horovod/horovod/pull/2596))

- Fixed DL estimators to obtain the output df schema without sampling the input. ([#2611](https://github.com/horovod/horovod/pull/2611))

- Fixed wrong default for horovod.tensorflow.keras.allreduce average ([#2627](https://github.com/horovod/horovod/pull/2627))

## [v0.21.1] - 2021-01-06

### Added

- Added in-memory dataset caching param to `TorchEstimator`. ([#2434](https://github.com/horovod/horovod/pull/2434))

- Added `val_batch_size` param to the Estimator API. ([#2505](https://github.com/horovod/horovod/pull/2505))

- Added support for TorchScript modules when using `TorchEstimator`. ([#2494](https://github.com/horovod/horovod/pull/2494))

### Changed

- Migrated to oneCCL aligned with oneAPI specification v1.0. ([#2513](https://github.com/horovod/horovod/pull/2513))

- Added knob to set cache hint for oneCCL allreduce. ([#2560](https://github.com/horovod/horovod/pull/2560))

- Renamed `horovodrun` arg `--ccl-bgt-affinity` to `--thread-affinity`. ([#2562](https://github.com/horovod/horovod/pull/2562))

- Changed default build parallelism from `-j8` to `-j1` to address potential race condition. ([#2572](https://github.com/horovod/horovod/pull/2572))

### Fixed

- Fixed building Horovod for ROCm PyTorch with newer hipify script. ([#2360](https://github.com/horovod/horovod/pull/2360))

- Fixed "Executable class" support for Ray. ([#2510](https://github.com/horovod/horovod/pull/2510))

- Fixed TorchEstimator returning model without switching to eval mode. ([#2517](https://github.com/horovod/horovod/pull/2517))

- Remove ssh reliance for Ray elastic training. ([#2528](https://github.com/horovod/horovod/pull/2528))

- Fixed error handling for changing framework without reinstalling horovod. ([#2529](https://github.com/horovod/horovod/pull/2529))

- Fixed "Intermediate path does not exist" error with DBFSLocalStore. ([#2526](https://github.com/horovod/horovod/pull/2526))

- Avoid synchronization if workers are only shrinked in elastic mode. ([#2514](https://github.com/horovod/horovod/pull/2514))

- Fixed Ray resource test. ([#2575](https://github.com/horovod/horovod/pull/2575))

- Fixed usage of env variable `HOROVOD_GLOO_TIMEOUT_SECONDS` with `horovodrun`. ([#2571](https://github.com/horovod/horovod/pull/2571))

## [v0.21.0] - 2020-11-23

### Added

- Added support for backward_passes_per_step > 1 for TF Keras graph mode. ([#2346](https://github.com/horovod/horovod/pull/2346))

- Added support for backward_passes_per_step > 1 for TF Keras eager execution. ([#2371](https://github.com/horovod/horovod/pull/2371))

- Added support for backward_passes_per_step > 1 for TF LegacyOptimizer in graph mode. ([#2401](https://github.com/horovod/horovod/pull/2401))

- Added grouped allreduce to enable more efficient tensor fusion and deterministic training. ([#2453](https://github.com/horovod/horovod/pull/2453))

- Add support for specifying `op` and `compression` in `horovod.tensorflow.keras.allreduce()`. ([#2423](https://github.com/horovod/horovod/pull/2423))

- Adding support for batched D2D memcopy kernel on GPU. ([#2435](https://github.com/horovod/horovod/pull/2435))

- Added schema inference in Spark Estimator without sampling. ([#2373](https://github.com/horovod/horovod/pull/2373))

- Added `Store.create("dbfs:/")` mapping to `DBFSLocalStore("/dbfs/...")`. ([#2376](https://github.com/horovod/horovod/pull/2376))

### Changed

- Changed Keras callbacks to require parameter `initial_lr` of `LearningRateScheduleCallback` and `LearningRateWarmupCallback`. ([#2459](https://github.com/horovod/horovod/pull/2459))

- Changed default cycle time from 5ms to 1ms and fusion threshold from 64MB to 128MB. ([#2468](https://github.com/horovod/horovod/pull/2468))

### Fixed

- Fixed support for TensorFlow v2.4.0. ([#2381](https://github.com/horovod/horovod/pull/2381))

- Fixed averaging using CUDA half2 implementation one element half buffers. ([#2375](https://github.com/horovod/horovod/pull/2375))

- Fixed `HOROVOD_THREAD_AFFINITY` when using oneCCL. ([#2350](https://github.com/horovod/horovod/pull/2350))

- Added timeout to SSH check in horovodrun to prevent hanging. ([#2448](https://github.com/horovod/horovod/pull/2448))

- Added `HOROVOD_GLOO_TIMEOUT_SECONDS` value to error messages. ([#2436](https://github.com/horovod/horovod/pull/2436))

- Fixed race condition in dynamic timeline API. ([#2341](https://github.com/horovod/horovod/pull/2341))

- Fixed --log-hide-timestamp to apply to driver logs with Gloo. ([#2388](https://github.com/horovod/horovod/pull/2388))

- Fixed the search order of Eigen and Flatbuffers paths. ([#2473](https://github.com/horovod/horovod/pull/2473))

- Fixed type checks in `TorchEstimator` to correctly use `isinstance()`. ([#2480](https://github.com/horovod/horovod/pull/2480))

## [0.20.3] - 2020-10-01

### Added

- Added Elastic Ray integration. ([#2291](https://github.com/horovod/horovod/pull/2291))

### Changed

- Removed dependency on SSH access for Ray. ([#2275](https://github.com/horovod/horovod/pull/2275))

## [0.20.2] - 2020-09-25

### Fixed

- Fixed building Horovod without HOROVOD_WITHOUT_MXNET when MXNet is not installed. ([#2334](https://github.com/horovod/horovod/pull/2334))

## [0.20.1] - 2020-09-25

### Added

- Added Databricks storage `DBFSLocalStore` and support for GPU-aware scheduling to horovod.spark Estimator. ([#2234](https://github.com/horovod/horovod/pull/2234))

- Added ElasticSampler and PyTorch Elastic ImageNet example. ([#2297](https://github.com/horovod/horovod/pull/2297))

- Added ability to dynamically start and stop timeline programmatically. ([#2215](https://github.com/horovod/horovod/pull/2215))

- Added support for Gloo on macOS. ([#2254](https://github.com/horovod/horovod/pull/2254))

- Exposed name argument to TensorFlow allreduce operation. ([#2325](https://github.com/horovod/horovod/pull/2325))

- Added option to strip outer name scope from Horovod ops in TensorFlow. ([#2328](https://github.com/horovod/horovod/pull/2328))

### Fixed

- Fixed usage of VERBOSE=1 when setting custom MAKEFLAGS. ([#2239](https://github.com/horovod/horovod/pull/2239))

- Fixed bugs in Keras Elastic Callback classes. ([#2289](https://github.com/horovod/horovod/pull/2289))

- Fixed RelWithDebInfo build and made it the default with -03 optimizations. ([#2305](https://github.com/horovod/horovod/pull/2305))

- Fixed usage of tf.cond in TensorFlow alltoall gradient. ([#2327](https://github.com/horovod/horovod/pull/2327))

- Fixed allreduce averaging for TF IndexedSlices in ROCm path. ([#2279](https://github.com/horovod/horovod/pull/2279))

- Include stdexcept to handle certain compiler / frameworks that don't include it already. ([#2238](https://github.com/horovod/horovod/pull/2238))

- Fixed Debug builds by setting compiler options based on CMake build type. ([#2263](https://github.com/horovod/horovod/pull/2263))

- Skipped launching zero-sized send/recvs for NCCLAlltoall. ([#2273](https://github.com/horovod/horovod/pull/2273))

- Fixed missing run in tf keras elastic mode. ([#2272](https://github.com/horovod/horovod/pull/2272))

- Fixed loss function in TensorFlow2 elastic synthetic benchmark. ([#2265](https://github.com/horovod/horovod/pull/2265))

- Fixed usage of HOROVOD_MIXED_INSTALL env var in alltoall tests. ([#2266](https://github.com/horovod/horovod/pull/2266))

- Removed keras requirement from Ray example. ([#2262](https://github.com/horovod/horovod/pull/2262))

## [0.20.0] - 2020-09-02

### Added

- Added bare-metal elastic mode implementation to enable auto-scaling and fault tolerance. ([#1849](https://github.com/horovod/horovod/pull/1849))

- Added Elastic Horovod support for Spark auto-scaling. ([#1956](https://github.com/horovod/horovod/pull/1956))

- Added All-to-All operation for TensorFlow, PyTorch, and MXNet. ([#2143](https://github.com/horovod/horovod/pull/2143))

- Added support for `gradient_predivide_factor` and averaging in Horovod backend. ([#1949](https://github.com/horovod/horovod/pull/1949))

- Added NCCL implementation of the allgather operation. ([#1952](https://github.com/horovod/horovod/pull/1952))

- Added `HOROVOD_GPU_OPERATIONS` installation variable to simplify enabling NCCL support for all GPU operations. ([#1960](https://github.com/horovod/horovod/pull/1960))

- Added TensorFlow implementation of `SyncBatchNormalization` layer. ([#2075](https://github.com/horovod/horovod/pull/2075))

- Added `hvd.is_initialized()` method. ([#2020](https://github.com/horovod/horovod/pull/2020))

- Added `hvd.allgather_object` function for TensorFlow, PyTorch, and MXNet. ([#2166](https://github.com/horovod/horovod/pull/2166))

- Added `hvd.broadcast_object` function for MXNet. ([#2122](https://github.com/horovod/horovod/pull/2122))

- Added `label_shapes` parameter to KerasEstimator and TorchEstimator. ([#2140](https://github.com/horovod/horovod/pull/2140))

- Added optional `modelCheckPoint` callback to KerasEstimator params. ([#2124](https://github.com/horovod/horovod/pull/2124))

- Added `ssh_identity_file` argument to `horovodrun`. ([#2201](https://github.com/horovod/horovod/pull/2201))

- Added support for `horovodrun` on `kubeflow/mpi-job`. ([#2199](https://github.com/horovod/horovod/pull/2199))

- Added Ray integration. ([#2218](https://github.com/horovod/horovod/pull/2218))

### Changed

- Moved `horovod.run.runner.run` to `horovod.run`. ([#2099](https://github.com/horovod/horovod/pull/2099))

- HOROVOD_THREAD_AFFINITY accepts multiple values, one for every Horovod rank ([#2131](https://github.com/horovod/horovod/pull/2131))

- Migrated build system for native libraries to CMake ([#2009](https://github.com/horovod/horovod/pull/2009))

### Deprecated

- HOROVOD_CCL_BGT_AFFINITY is deprected. Use HOROVOD_THREAD_AFFINITY instead ([#2131](https://github.com/horovod/horovod/pull/2131))

### Removed

- Dropped support for Python 2. ([#1954](https://github.com/horovod/horovod/pull/1954))

- Dropped support for TensorFlow < 1.15. ([#2169](https://github.com/horovod/horovod/pull/2169))

- Dropped support for PyTorch < 1.2. ([#2086](https://github.com/horovod/horovod/pull/2086))

### Fixed

- Fixed MXNet allgather implementation to correctly handle resizing the output buffer. ([#2092](https://github.com/horovod/horovod/pull/2092))

- Fixed Keras Spark Estimator incompatibility with TensorFlow 1.15 due to `tf.autograph`. ([#2069](https://github.com/horovod/horovod/pull/2069))

- Fixed API compatibility with PyTorch 1.6. ([#2051](https://github.com/horovod/horovod/pull/2051))

- Fixed Keras API compatibility with TensorFlow 2.4.0. ([#2178](https://github.com/horovod/horovod/pull/2178))

- Fixed allgather gradient for TensorFlow 2 in cases where the tensor shape is not known during graph construction. ([#2121](https://github.com/horovod/horovod/pull/2121))

- Fixed running using Gloo with an imbalanced number of workers per host. ([#2212](https://github.com/horovod/horovod/pull/2212))
