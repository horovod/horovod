
// TODO: reducescatter -- migrate somwhere?

int horovod_torch_reducescatter_async_torch_cuda_IntTensor(THCudaIntTensor* tensor,
                                                           THCudaIntTensor* output,
                                                           char* name,
                                                           int reduce_op_int);
int horovod_torch_reducescatter_async_torch_cuda_LongTensor(
    THCudaLongTensor* tensor, THCudaLongTensor* output, char* name,
    int reduce_op_int);
int horovod_torch_reducescatter_async_torch_cuda_FloatTensor(THCudaTensor* tensor,
                                                             THCudaTensor* output,
                                                             char* name,
                                                             int reduce_op_int);
int horovod_torch_reducescatter_async_torch_cuda_DoubleTensor(
    THCudaDoubleTensor* tensor, THCudaDoubleTensor* output, char* name,
    int reduce_op_int);
