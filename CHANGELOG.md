# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - YYYY-MM-DD

### Added

### Changed

### Deprecated

### Removed

### Fixed

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

- Added ability to dynamically start and stop timeline programmatically. ([#2215](https://github.com/horovod/horovod/pull/2215))

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
