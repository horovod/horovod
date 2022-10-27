# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - YYYY-MM-DD

### Added

### Changed

- Improved NCCL performance for fused allgather operations through padding for better memory alignment. ([#3727](https://github.com/horovod/horovod/pull/3727))
- Improved look-ahead tensor fusion buffer size estimates when allgather and other operations are mixed. ([#3727](https://github.com/horovod/horovod/pull/3727))

### Deprecated

### Removed

### Fixed

- Fixed memory leak in MPI_GPUAllgather. ([#3727](https://github.com/horovod/horovod/pull/3727))


## [v0.26.1] - 2022-10-14

### Fixed

- Fixed packaging import during install to occur after install_requires. ([#3741](https://github.com/horovod/horovod/pull/3741))

## [v0.26.0] - 2022-10-13

### Added

- Spark Estimator: Added support for custom data loaders in KerasEstimator. ([#3603](https://github.com/horovod/horovod/pull/3603))
- Spark Estimator: Added NVTabular data loader for KerasEstimator. ([#3603](https://github.com/horovod/horovod/pull/3603))
- Spark Estimator: Added gradient accumulation support to Spark torch estimator. ([#3681](https://github.com/horovod/horovod/pull/3681))
- TensorFlow: Added `register_local_var` functionality to distributed optimizers and local gradient aggregators. ([#3695](https://github.com/horovod/horovod/pull/3695))
- TensorFlow: Added support for local variables for `BroadcastGlobalVariablesCallback`. ([#3703](https://github.com/horovod/horovod/pull/3703))
- Enabled use of native `ncclAvg` op for NCCL allreduces. ([#3646](https://github.com/horovod/horovod/pull/3646))
- Added support for additional reduction operations for `allreduce` (min, max, product). ([#3660](https://github.com/horovod/horovod/pull/3660))
- Added 2D torus `allreduce` using NCCL. ([#3608](https://github.com/horovod/horovod/pull/3608))
- Added support for Petastorm reader level parallel shuffling. ([#3665](https://github.com/horovod/horovod/pull/3665))
- Added random seed support for Lightning datamodule to generate reproducible data loading outputs. ([#3665](https://github.com/horovod/horovod/pull/3665))
- Added support for `int8` and `uint8` `allreduce` and `grouped_allreduce` in TensorFlow. ([#3649](https://github.com/horovod/horovod/pull/3649))
- Added support for batched memory copies in `GPUAllgather`. ([#3590](https://github.com/horovod/horovod/pull/3590))
- Added support for batched memory copies in `GPUReducescatter`. ([#3621](https://github.com/horovod/horovod/pull/3621))
- Added `hvd.grouped_allgather()` and `hvd.grouped_reducescatter()` operations. ([#3594](https://github.com/horovod/horovod/pull/3594))
- Added warning messages if output tensor memory allocations fail. ([#3594](https://github.com/horovod/horovod/pull/3594))
- Added `register_local_source` and `use_generic_names` funtionality to `DistributedGradientTape`. ([#3628](https://github.com/horovod/horovod/pull/3628))
- Added `PartialDistributedGradientTape()` API for model parallel use cases. ([#3643](https://github.com/horovod/horovod/pull/3643))
- Spark/Lightning: Added `reader_worker_count` and `reader_pool_type`. ([#3612](https://github.com/horovod/horovod/pull/3612))
- Spark/Lightning: Added `transformation_edit_fields` and `transformation_removed_fields` param for `EstimatorParams`. ([#3651](https://github.com/horovod/horovod/pull/3651))
- TensorFlow: Added doc string for `hvd.grouped_allreduce()`. ([#3594](https://github.com/horovod/horovod/pull/3594))
- ROCm: Enabled `alltoall`. ([#3654](https://github.com/horovod/horovod/pull/3654))

### Changed

- Default Petastorm reader pool is changed from `process` to `thread` for lower memory usage. ([#3665](https://github.com/horovod/horovod/pull/3665))
- Keras: Support only legacy optimizers in Keras 2.11+. ([#3725](https://github.com/horovod/horovod/pull/3725))
- Gloo: When negotiating, use `gather` rather than `allgather`. ([#3633](https://github.com/horovod/horovod/pull/3633))
- Use `packaging.version` instead of `distutils` version classes. ([#3700](https://github.com/horovod/horovod/pull/3700))

### Deprecated

- Deprecated field `shuffle_buffer_size` from `EstimatorParams`. Use `shuffle` to enable shuffle or not. ([#3665](https://github.com/horovod/horovod/pull/3665))

### Removed

- Build: Removed std::regex use for better cxxabi11 compatibility. ([#3584](https://github.com/horovod/horovod/pull/3584))

### Fixed

- TensorFlow: Fixed the optimizer iteration increments when `backward_passes_per_step > 1`. ([#3631](https://github.com/horovod/horovod/pull/3631))
- Fixed `FuseResponses()` on `BATCHED_D2D_PADDING` edge cases for Reducescatter and/or ROCm. ([#3621](https://github.com/horovod/horovod/pull/3621))
- PyTorch: Fixed Reducescatter functions to raise `HorovodInternalError` rather than `RuntimeError`. ([#3594](https://github.com/horovod/horovod/pull/3594))
- PyTorch on GPUs without GPU operations: Fixed grouped allreduce to set CPU device in tensor table. ([#3594](https://github.com/horovod/horovod/pull/3594))
- Fixed race condition in PyTorch allocation handling. ([#3639](https://github.com/horovod/horovod/pull/3639))
- Build: Fixed finding `nvcc` (if not in `$PATH`) with older versions of CMake. ([#3682](https://github.com/horovod/horovod/pull/3682))
- Fixed `reducescatter()` and `grouped_reducescatter()` to raise clean exceptions for scalar inputs. ([#3699](https://github.com/horovod/horovod/pull/3699))
- Updated Eigen submodule to fix build on macOS with aarch64. ([#3619](https://github.com/horovod/horovod/pull/3619))
- Build: Correctly select files in `torch/` directory to be hipified. ([#3588](https://github.com/horovod/horovod/pull/3588))
- Build: Modify regex match for CUDA|ROCm in `FindPytorch.cmake`. ([#3593](https://github.com/horovod/horovod/pull/3593))
- Build: Fixed ROCm-specific build failure. ([#3630](https://github.com/horovod/horovod/pull/3630))

## [v0.25.0] - 2022-06-20

### Added

- Added `hvd.reducescatter()` operation with implementations in NCCL, MPI, and Gloo. ([#3299](https://github.com/horovod/horovod/pull/3299), [#3574](https://github.com/horovod/horovod/pull/3574))
- Added AMD GPU XLA Op Implementation. ([#3486](https://github.com/horovod/horovod/pull/3486))
- Added Horovod job to spin up distributed TensorFlow Data Service. ([#3525](https://github.com/horovod/horovod/pull/3525))
- Spark: Expose random seed as an optional parameter. ([#3517](https://github.com/horovod/horovod/pull/3517))
- Add Helm Chart. ([#3546](https://github.com/horovod/horovod/pull/3546))
- Elastic: Add elastic run API. ([#3503](https://github.com/horovod/horovod/pull/3503))
- Spark Estimator: Expose random seed for model training reproducibility. ([#3517](https://github.com/horovod/horovod/pull/3517))
- Spark Estimator: Add option whether to use GPUs at all. ([#3526](https://github.com/horovod/horovod/pull/3526))
- Spark Estimator: Expose parameter to set start method for `multiprocessing`. ([#3580](https://github.com/horovod/horovod/pull/3580))

### Changed

- MXNet: Updated allreduce functions to newer `op` API. ([#3299](https://github.com/horovod/horovod/pull/3299))
- TensorFlow: Make TensorFlow output allocations asynchronous when using NCCL backend. ([#3464](https://github.com/horovod/horovod/pull/3464))
- TensorFlow: Clear locally accumulated gradient by assigning with `zeros_like` to avoid infinite gradient not correctly cleared up. ([#3505](https://github.com/horovod/horovod/pull/3505))
- Make `HorovodVersionMismatchError` subclass `ImportError` instead of just a standard `Exception`. ([#3549](https://github.com/horovod/horovod/pull/3549))
- Elastic: Catch any exception to prevent the discovery thread from silently dying. ([#3436](https://github.com/horovod/horovod/pull/3436))
- Horovodrun: Exit check_build (`--check-build`) via `sys.exit` to flush stdout. ([#3272](https://github.com/horovod/horovod/pull/3272))
- Spark: Use `env` to set environment vars in remote shell. ([#3489](https://github.com/horovod/horovod/pull/3489))
- Build: Avoid redundant ptx generation for maximum specified compute capability. ([#3509](https://github.com/horovod/horovod/pull/3509))

### Deprecated

- MXNet: Deprecated `average` argument of allreduce functions. ([#3299](https://github.com/horovod/horovod/pull/3299))
- Public and internal APIs: deprecate use of np, min_np, max_np. Use num_proc, min_num_proc, and max_num_proc, respectively, instead. ([#3409](https://github.com/horovod/horovod/pull/3409))
- Horovodrun: Providing multiple NICS as comma-separated string via `--network-interface` is deprecated,
  use `--network-interface` multiple times or `--network-interfaces` instead. ([#3506](https://github.com/horovod/horovod/pull/3506))
- horovod.run: Argument `network_interface` with comma-separated string is deprecated,
  use `network_interfaces` with `Iterable[str]` instead. ([#3506](https://github.com/horovod/horovod/pull/3506))

### Fixed

- Fallback to NCCL shared lib if static one is not found. ([#3500]((https://github.com/horovod/horovod/pull/3500))
- Spark/Lightning: Added missing `tranform_spec` for Petastorm datamodule. ([#3543](https://github.com/horovod/horovod/pull/3543))
- Spark/Lightning: Fixed PTL Spark example with checkpoint usage by calling `save_hyperparameters()`. ([#3527](https://github.com/horovod/horovod/pull/3527))
- Elastic: Fixed empty hostname returned from `HostDiscoveryScript`. ([#3490](https://github.com/horovod/horovod/pull/3490))
- TensorFlow 2.9: Fixed build for API change related to `tensorflow_accelerator_device_info`. ([#3513](https://github.com/horovod/horovod/pull/3513))
- TensorFlow 2.10: Bumped build partially to C++17. ([#3558](https://github.com/horovod/horovod/pull/3558))
- TensorFlow: Fixed gradient update timing in TF `AggregationHelperEager`. ([#3496](https://github.com/horovod/horovod/pull/3496))
- TensorFlow: Fixed resource `NotFoundError` in TF `AggregationHelper`. ([#3499](https://github.com/horovod/horovod/pull/3499))

## [v0.24.3] - 2022-04-21

### Fixed

- Make DBFSLocalStore support "file:/dbfs/...", implement get_localized_path. ([#3510](https://github.com/horovod/horovod/pull/3510))

## [v0.24.2] - 2022-03-10

### Fixed

- Setup: Require fsspec >= 2010.07.0 ([#3451](https://github.com/horovod/horovod/pull/3451))
- Fix ignored cuda arch flags ([#3462]((https://github.com/horovod/horovod/pull/3462))

## [v0.24.1] - 2022-03-03

### Fixed

- Extended CMake build script to often find CUDA even if `nvcc` is not in `$PATH`. ([#3444](https://github.com/horovod/horovod/pull/3444))

## [v0.24.0] - 2022-03-01

### Added

- Ray: Added elastic keyword parameters to RayExecutor API: This API supports both static (non-elastic) and elastic Horovod jobs. ([#3190](https://github.com/horovod/horovod/issues/3190))
- TensorFlow: Added in-place broadcasting of variables. ([#3128](https://github.com/horovod/horovod/pull/3128))
- Elastic: Added support for resurrecting blacklisted hosts. ([#3319](https://github.com/horovod/horovod/pull/3319))
- MXNet: Added support for MXNet async dependency engine. ([#3242](https://github.com/horovod/horovod/pull/3242), [#2963](https://github.com/horovod/horovod/pull/2963))
- Spark/Lightning: Added history to lightning estimator. ([#3214](https://github.com/horovod/horovod/pull/3214))

### Changed

- Moved to CMake version 3.13 with first-class CUDA language support and re-enabled parallelized builds. Uses a temporary installation of CMake if CMake 3.13 is not found. ([#3261](https://github.com/horovod/horovod/pull/3261), [#3371](https://github.com/horovod/horovod/pull/3371))
- Moved released Docker image `horovod` and `horovod-cpu` to Ubuntu 20.04 and Python 3.8. ([#3393](https://github.com/horovod/horovod/pull/3393))
- Spark Estimator: Don't shuffle row groups if training data requires non-shuffle ([#3369](https://github.com/horovod/horovod/pull/3369))
- Spark/Lightning: Reduced memory footprint of async dataloader. ([#3239](https://github.com/horovod/horovod/pull/3239))
- Elastic: Improved handling NCCL errors under elastic scenario. ([#3112](https://github.com/horovod/horovod/pull/3112))
- Spark/Lightning: Do not overwrite model with checkpoint by default. ([#3201](https://github.com/horovod/horovod/pull/3201))
- Make checkpoint name optional so that user can save to h5 format. ([#3411](https://github.com/horovod/horovod/pull/3411))

### Deprecated

- Deprecated ElasticRayExecutor APIs in favor of the new RayExecutor API. ([#3190](https://github.com/horovod/horovod/issues/3190))

### Removed

- Spark: Removed `h5py<3` constraint as this is not needed anymore for Tensorflow >2.5.0. ([#3301](https://github.com/horovod/horovod/pull/3301))

### Fixed

- Elastic Spark: Fixed indices in initial task-to-task registration. ([#3410](https://github.com/horovod/horovod/pull/3410))
- PyTorch: Fixed GIL-related deadlock with PyTorch 1.10.1. ([#3352](https://github.com/horovod/horovod/issues/3352))
- PyTorch: Fixed finalization of ProcessSetTable. ([#3351](https://github.com/horovod/horovod/pull/3351))
- Fixed remote trainers to point to the correct shared lib path. ([#3258](https://github.com/horovod/horovod/pull/3258))
- Fixed imports from `tensorflow.python.keras` with tensorflow 2.6.0+. ([#3403](https://github.com/horovod/horovod/pull/3403))
- Fixed Adasum communicator init logic. ([#3379](https://github.com/horovod/horovod/pull/3379))
- Lightning: Fixed resume logger. ([#3375](https://github.com/horovod/horovod/pull/3375))
- Fixed the checkpoint directory structure for pytorch and pytorch lightning. ([#3362](https://github.com/horovod/horovod/pull/3362))
- Fixed possible integer overflow in multiplication. ([#3368](https://github.com/horovod/horovod/pull/3368))
- Fixed the `pytorch_lightning_mnist.py` example. ([#3245](https://github.com/horovod/horovod/pull/3245), [#3290](https://github.com/horovod/horovod/pull/3290))
- Fixed barrier segmentation fault. ([#3313](https://github.com/horovod/horovod/pull/3313))
- Fixed `hvd.barrier()` tensor queue management. ([#3300](https://github.com/horovod/horovod/pull/3300))
- Fixed PyArrow "list index out of range" IndexError. ([#3274](https://github.com/horovod/horovod/pull/3274))
- Elastic: Fixed all workers sometimes failing on elastic Horovod failure. ([#3264](https://github.com/horovod/horovod/issues/3264))
- Spark/Lightning: Fixed setting `limit_train_batches` and `limit_val_batches`. ([#3237](https://github.com/horovod/horovod/pull/3237))
- Elastic: Fixed ElasticSampler and `hvd.elastic.state` losing some indices of processed samples when nodes dropped. ([#3143](https://github.com/horovod/horovod/issues/3143))
- Spark/Lightning: Fixed history metrics for estimator serialization. ([#3216](https://github.com/horovod/horovod/pull/3216))
- Ray: Fixed RayExecutor to fail when `num_workers=0` and `num_hosts=None`. ([#3210](https://github.com/horovod/horovod/pull/3210))
- Spark/Lightning: Fixed checkpoint callback `dirpath` typo. ([#3204](https://github.com/horovod/horovod/pull/3204))

## [v0.23.0] - 2021-10-06

### Added

- Added process sets to concurrently run collective operations on subsets of Horovod processes in TensorFlow, PyTorch, and MXNet. ([#2839](https://github.com/horovod/horovod/pull/2839), [#3042](https://github.com/horovod/horovod/pull/3042), [#3043](https://github.com/horovod/horovod/pull/3043), [#3054](https://github.com/horovod/horovod/pull/3054), [#3083](https://github.com/horovod/horovod/pull/3083), [#3090](https://github.com/horovod/horovod/pull/3090))
- Added XLA support for Allreduce via `tf.function(jit_compile=True)`. ([#3053](https://github.com/horovod/horovod/pull/3053))
- Added fused buffer scaling and unpack/pack kernels on GPU. ([#2973](https://github.com/horovod/horovod/pull/2973))
- Added support for NCCL on CUDA 11.4. ([#3182](https://github.com/horovod/horovod/issues/3182))
- Added fp16 compression for MXNet. ([#2987](https://github.com/horovod/horovod/issues/2987))
- Added terminate_on_nan flag to Spark Lightning estimator. ([#3088](https://github.com/horovod/horovod/issues/3088))
- Added barrier() API to torch module to support simple synchronization among ranks and to achieve parity with PyTorch DDP and similar frameworks. [#3139](https://github.com/horovod/horovod/pull/3139)
- Added params for customizing Tensorboard callback. ([#3153](https://github.com/horovod/horovod/issues/3153))
- Added `hvd.cross_rank()` for keras. ([#3008](https://github.com/horovod/horovod/issues/3008))
- Added barrier() API to torch module to support simple synchronization among ranks and to achieve parity with PyTorch DDP and similar frameworks. [#3139](https://github.com/horovod/horovod/pull/3139)

### Changed

- Implemented more asynchronous dependency handling on GPU. ([#2963](https://github.com/horovod/horovod/pull/2963))
- Ray: RayExecutor will now use the current placement group instead of always creating a new one. ([#3134](https://github.com/horovod/horovod/pull/3134))
- Lightning: turned off shuffling for validation dataset. ([#2974](https://github.com/horovod/horovod/pull/2974))
- Ray: RayExecutor will use the current placement group if one exists. ([#3134](https://github.com/horovod/horovod/pull/3134))
- Extended `hvd.join()` to return the last rank that joined. ([#3097](https://github.com/horovod/horovod/pull/3097)

### Deprecated

### Removed

- Spark/Keras: remove bare Keras support. ([#3191](https://github.com/horovod/horovod/pull/3191))

### Fixed

- Fix Horovod develop/editable install mode and incremental builds. ([#3074](https://github.com/horovod/horovod/pull/3074))
- Estimator/Lightning: use lightning datamodule. ([#3084](https://github.com/horovod/horovod/pull/3084))
- Fix Horovod Spark StringType and numpy type mapping issue. ([#3146](https://github.com/horovod/horovod/pull/3146))
- Fixed error in Keras LearningRateScheduler. ([#3135](https://github.com/horovod/horovod/pull/3135))
- Fixed bug in Lightning Profiler on Ray. ([#3122](https://github.com/horovod/horovod/pull/3122))
- Fixed torch op lazy release to prevent OOM in elastic training. ([#3110](https://github.com/horovod/horovod/pull/3110))
- Lightning: Fixed usage of the checkpoint callback. ([#3186](https://github.com/horovod/horovod/pull/3186))
- Fixed MPICH support to use Intel MPI's implementation. ([#3148](https://github.com/horovod/horovod/pull/3148))
- Fixed race condition in PyTorch async dataloader. ([#3120](https://github.com/horovod/horovod/pull/3120))
- Keras: Fixed learning rate scheduler. ([#3142](https://github.com/horovod/horovod/pull/3142), [#3135](https://github.com/horovod/horovod/pull/3135))

## [v0.22.1] - 2021-06-10

### Added

- Estimator: added support for loading data from S3, GCS, ADLS, and other remote filesystems. ([#2927](https://github.com/horovod/horovod/issues/2927))
- Estimator: added custom Spark data loader interface. ([#2938](https://github.com/horovod/horovod/issues/2923))
- LightningEstimator: added support to supply a logger and associated parameter to control the frequency of logging. ([#2926](https://github.com/horovod/horovod/pull/2926))
- Estimator: added check to ensure all ranks have the same device type. ([#2942](https://github.com/horovod/horovod/pull/2942))

### Changed

- Changed behavior from using TensorBoardLogger to now using it as a fallback if a logger is not supplied. ([#2926](https://github.com/horovod/horovod/pull/2926))
- Ray: disabled capturing child tasks in placement group. ([#2920](https://github.com/horovod/horovod/pull/2920))

### Fixed

- Fixed `hvd.tensorflow.keras.Compression`, accidentally removed in v0.22.0. ([#2945](https://github.com/horovod/horovod/pull/2945))
- TorchEstimator: fixed usage of `validation_steps` in place of `validation_steps_per_epoch`. ([#2918](https://github.com/horovod/horovod/pull/2918))
- TensorFlow: fixed C++ API for TF v2.6.0. ([#2932](https://github.com/horovod/horovod/pull/2932))
- PyTorch: fixed `sparse_allreduce_async` for PyTorch v0.10.0. ([#2965](https://github.com/horovod/horovod/pull/2965))

## [v0.22.0] - 2021-05-18

### Added

- Added pytorch_lightning spark estimator which enables training pytorch_lightning models. ([#2713](https://github.com/horovod/horovod/pull/2713))
- Added NVTX tracing hooks for profiling with Nsight Systems. ([#2723](https://github.com/horovod/horovod/pull/2723))
- Added a generic `num_workers` API for ``RayExecutor`` ([#2870](https://github.com/horovod/horovod/pull/2870))
- Supports Ray Client without code changes. ([#2882](https://github.com/horovod/horovod/pull/2882))
- Supports inmemory cache option for Keras Estimator. ([#2896](https://github.com/horovod/horovod/pull/2896))
- Added FP16 support for GPU tensor in mxnet. ([#2915](https://github.com/horovod/horovod/pull/2915))
- Added response caching for allgather operations. ([#2872](https://github.com/horovod/horovod/pull/2872))
- Estimator: add petastorm reader_pool_type into constructor ([#2903](https://github.com/horovod/horovod/pull/2903))

### Changed

- Changed `alltoall` to return the received splits as a second return value if non-uniform splits are sent. ([#2631](https://github.com/horovod/horovod/pull/2631))
- Changed ``RayExecutor`` to use [Ray Placement Groups](https://docs.ray.io/en/master/placement-group.html) for worker colocation. ([#2824](https://github.com/horovod/horovod/pull/2824))
- Changed ``Inmemory dataloader`` usage for Torch Estimator with petastorm v0.11.0 release. ([#2896](https://github.com/horovod/horovod/pull/2896))

### Fixed

- Changed RayExecutor to use Ray node ID to enable multi-container:single-host setups. ([#2883](https://github.com/horovod/horovod/pull/2882))
- Support sparse gradients aggregation in TF1 Keras. ([#2879](https://github.com/horovod/horovod/pull/2879))
- Respect `global_step` parameter for LegacyOptimizers when aggregating gradients.  ([#2879](https://github.com/horovod/horovod/pull/2879))
- Fixed compatibility with PyTorch 1.9.0. ([#2829](https://github.com/horovod/horovod/pull/2829))

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
