
// TODO: reducescatter -- migrate somewhere

#define REDUCESCATTER_H(torch_Tensor, THTensor)                                \
  extern "C" int horovod_torch_reducescatter_async_##torch_Tensor(             \
      THTensor* tensor, THTensor* output, char* name, int reduce_op_int);

REDUCESCATTER_H(torch_IntTensor, THIntTensor)
REDUCESCATTER_H(torch_LongTensor, THLongTensor)
REDUCESCATTER_H(torch_FloatTensor, THFloatTensor)
REDUCESCATTER_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
REDUCESCATTER_H(torch_cuda_IntTensor, THCudaIntTensor)
REDUCESCATTER_H(torch_cuda_LongTensor, THCudaLongTensor)
REDUCESCATTER_H(torch_cuda_FloatTensor, THCudaTensor)
REDUCESCATTER_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif
