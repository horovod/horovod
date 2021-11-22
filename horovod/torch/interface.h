

// TODO: reducescatter -- migrate somewhere?
int horovod_torch_reducescatter_async_torch_IntTensor(THIntTensor* tensor,
                                                      THIntTensor* output,
                                                      char* name,
                                                      int reduce_op_int);
int horovod_torch_reducescatter_async_torch_LongTensor(THLongTensor* tensor,
                                                       THLongTensor* output,
                                                       char* name,
                                                       int reduce_op_int);
int horovod_torch_reducescatter_async_torch_FloatTensor(THFloatTensor* tensor,
                                                        THFloatTensor* output,
                                                        char* name,
                                                        int reduce_op_int);
int horovod_torch_reducescatter_async_torch_DoubleTensor(THDoubleTensor* tensor,
                                                         THDoubleTensor* output,
                                                         char* name,
                                                         int reduce_op_int);


