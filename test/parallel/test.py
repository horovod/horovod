import horovod.torch as hvd
hvd.init()
print(hvd.rank())
