{
  global:
    *horovod*;
    # PyTorch binding
    *PyInit*;
    *initmpi_lib_v2*;
    # Legacy PyTorch binding
    *init_mpi_lib*;
    *init_mpi_lib_impl*;
  local: *;
};
