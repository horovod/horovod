
// TODO: Migrate reducescatter to mpi_opsv2.cc (1)
template <DataType DT, DeviceType Dev, class T>
int DoReducescatter(T* tensor, T* output, char* name, int reduce_op_int) {
  ThrowIfError(common::CheckInitialized());

  // For ReduceOp::AVERAGE, we do SUM reduction then divide on the device.
  auto reduce_op = static_cast<ReduceOp>(reduce_op_int);
  auto request_op = reduce_op == ReduceOp::AVERAGE ? ReduceOp::SUM : reduce_op;

  auto handle = handle_manager.AllocateHandle();
  auto device = TensorUtil::GetDevice<DT, Dev>(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor<DT, Dev, T>>(tensor);
  auto hvd_context =
      std::make_shared<TorchOpContext<DT, Dev, T>>(device, output);

  auto enqueue_result = EnqueueTensorReducescatter(
      hvd_context, hvd_tensor, ready_event,
      GetOpName("reducescatter", name, handle), device,
      [handle, reduce_op, output](const Status& status) {
        if (reduce_op == ReduceOp::AVERAGE) {
          TensorUtil::DivideTensorInPlace<DT, Dev, T>(output, horovod_size());
        }
        handle_manager.MarkDone(handle, status);
      }, request_op);
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
template <DataType DT, class TC, class T>
int DoReducescatterCudaOnCPU(TC* tensor, TC* output, char* name, int reduce_op_int) {
  ThrowIfError(common::CheckInitialized());

  // For ReduceOp::AVERAGE, we do SUM reduction then divide on the device.
  auto reduce_op = static_cast<ReduceOp>(reduce_op_int);
  auto request_op = reduce_op == ReduceOp::AVERAGE ? ReduceOp::SUM : reduce_op;

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice<DT, DeviceType::GPU>(tensor);
  auto hvd_cpu_tensor =
      std::make_shared<TorchTemporaryBuffer<DT, DeviceType::CPU, T>>(
          CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU<DT>(tensor, hvd_cpu_tensor->tensor());
  auto ready_event = RecordReadyEvent(device);

  auto hvd_cpu_output =
      std::make_shared<TorchTemporaryBuffer<DT, DeviceType::CPU, T>>(
          CPU_DEVICE_ID);
  auto hvd_context = std::make_shared<TorchOpContext<DT, DeviceType::CPU, T>>(
      CPU_DEVICE_ID, hvd_cpu_output->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorReducescatter(
      hvd_context, hvd_cpu_tensor, ready_event,
      GetOpName("reducescatter", name, handle), CPU_DEVICE_ID,
      [handle, reduce_op, hvd_cpu_output, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<DT>(hvd_cpu_output->tensor(), output);
        if (reduce_op == ReduceOp::Average) {
          TensorUtil::DivideTensorInPlace<DT, DeviceType::GPU>(output,
                                                               horovod_size());
        }
        handle_manager.MarkDone(handle, status);
      }, request_op);
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

// TODO: Migrate reducescatter to mpi_opsv2.cc (2)

#define REDUCESCATTER(torch_Tensor, HorovodType, DeviceType, THTensor)         \
  extern "C" int horovod_torch_reducescatter_async_##torch_Tensor(             \
      THTensor* tensor, THTensor* output, char* name, int reduce_op_int) {     \
    return DoReducescatter<HorovodType, DeviceType>(tensor, output, name,      \
                                                    reduce_op_int);            \
  }

REDUCESCATTER(torch_IntTensor, DataType::HOROVOD_INT32, DeviceType::CPU,
              THIntTensor)
REDUCESCATTER(torch_LongTensor, DataType::HOROVOD_INT64, DeviceType::CPU,
              THLongTensor)
REDUCESCATTER(torch_FloatTensor, DataType::HOROVOD_FLOAT32, DeviceType::CPU,
              THFloatTensor)
REDUCESCATTER(torch_DoubleTensor, DataType::HOROVOD_FLOAT64, DeviceType::CPU,
              THDoubleTensor)

#if HOROVOD_GPU_REDUCESCATTER
REDUCESCATTER(torch_cuda_IntTensor, DataType::HOROVOD_INT32, DeviceType::GPU,
              THCudaIntTensor)
REDUCESCATTER(torch_cuda_LongTensor, DataType::HOROVOD_INT64, DeviceType::GPU,
              THCudaLongTensor)
REDUCESCATTER(torch_cuda_FloatTensor, DataType::HOROVOD_FLOAT32, DeviceType::GPU,
              THCudaTensor)
REDUCESCATTER(torch_cuda_DoubleTensor, DataType::HOROVOD_FLOAT64,
              DeviceType::GPU, THCudaDoubleTensor)
#endif

#define REDUCESCATTER_CUDA_ON_CPU(torch_Tensor, HorovodType, THCTensor, THTensor) \
  extern "C" int horovod_torch_reducescatter_async_##torch_Tensor(                \
      THCTensor* tensor, THCTensor* output, char* name, int reduce_op_int) {      \
    return DoReducescatterCudaOnCPU<HorovodType, THCTensor, THTensor>(            \
        tensor, output, name, reduce_op_int);                                     \
  }

#if !HOROVOD_GPU_REDUCESCATTER && HAVE_CUDA
REDUCESCATTER_CUDA_ON_CPU(torch_cuda_IntTensor, DataType::HOROVOD_INT32,
                          THCudaIntTensor, THIntTensor)
REDUCESCATTER_CUDA_ON_CPU(torch_cuda_LongTensor, DataType::HOROVOD_INT64,
                          THCudaLongTensor, THLongTensor)
REDUCESCATTER_CUDA_ON_CPU(torch_cuda_FloatTensor, DataType::HOROVOD_FLOAT32,
                          THCudaTensor, THFloatTensor)
REDUCESCATTER_CUDA_ON_CPU(torch_cuda_DoubleTensor, DataType::HOROVOD_FLOAT64,
                          THCudaDoubleTensor, THDoubleTensor)
#endif
