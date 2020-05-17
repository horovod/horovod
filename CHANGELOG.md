# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - YYYY-MM-DD

### Added

- Added bare-metal elastic mode implementation to enable auto-scaling and fault tolerance. ([#1849](https://github.com/horovod/horovod/pull/1849))

- Added NCCL implementation of the allgather operation. ([#1952](https://github.com/horovod/horovod/pull/1952))

- Added `HOROVOD_GPU_OPERATIONS` installation variable to simplify enabling NCCL support for all GPU operations. ([#1960](https://github.com/horovod/horovod/pull/1960))

### Changed

### Deprecated

### Removed

- Dropped support for Python 2. ([#1954](https://github.com/horovod/horovod/pull/1954))

### Fixed
