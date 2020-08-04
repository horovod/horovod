# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - YYYY-MM-DD

### Added

- Added bare-metal elastic mode implementation to enable auto-scaling and fault tolerance. ([#1849](https://github.com/horovod/horovod/pull/1849))

- Added NCCL implementation of the allgather operation. ([#1952](https://github.com/horovod/horovod/pull/1952))

- Added `HOROVOD_GPU_OPERATIONS` installation variable to simplify enabling NCCL support for all GPU operations. ([#1960](https://github.com/horovod/horovod/pull/1960))

- Added TensorFlow implementation of `SyncBatchNormalization` layer. ([#2075](https://github.com/horovod/horovod/pull/2075))

- Added `hvd.is_initialized()` method. ([#2020](https://github.com/horovod/horovod/pull/2020))

### Changed

- Moved `horovod.run.runner.run` to `horovod.run`. ([#2099](https://github.com/horovod/horovod/pull/2099))

- HOROVOD_THREAD_AFFINITY accepts multiple values, one for every Horovod rank ([#2131](https://github.com/horovod/horovod/pull/2131))

### Deprecated

- HOROVOD_CCL_BGT_AFFINITY is deprected. Use HOROVOD_THREAD_AFFINITY instead ([#2131](https://github.com/horovod/horovod/pull/2131))

### Removed

- Dropped support for Python 2. ([#1954](https://github.com/horovod/horovod/pull/1954))

### Fixed

- Fixed MXNet allgather implementation to correctly handle resizing the output buffer. ([#2092](https://github.com/horovod/horovod/pull/2092))

- Fixed Keras Spark Estimator incompatibility with TensorFlow 1.15 due to `tf.autograph`. ([#2069](https://github.com/horovod/horovod/pull/2069))

- Fixed API compatibility with PyTorch 1.6. ([#2051](https://github.com/horovod/horovod/pull/2051))
