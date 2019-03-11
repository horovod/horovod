// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mpi_context.h"

namespace horovod {
namespace common {

MPI_Datatype MPIContext::GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
  return GetMPIDataType(tensor->dtype());
}

MPI_Datatype MPIContext::GetMPIDataType(const DataType dtype) {
  switch (dtype) {
    case HOROVOD_UINT8:
      return MPI_UINT8_T;
    case HOROVOD_INT8:
      return MPI_INT8_T;
    case HOROVOD_UINT16:
      return MPI_UINT16_T;
    case HOROVOD_INT16:
      return MPI_INT16_T;
    case HOROVOD_INT32:
      return MPI_INT32_T;
    case HOROVOD_INT64:
      return MPI_INT64_T;
    case HOROVOD_FLOAT16:
      return mpi_float16_t;
    case HOROVOD_FLOAT32:
      return MPI_FLOAT;
    case HOROVOD_FLOAT64:
      return MPI_DOUBLE;
    case HOROVOD_BOOL:
      return MPI_C_BOOL;
    case HOROVOD_BYTE:
      return MPI_BYTE;
    default:
      throw std::logic_error("Type " + DataType_Name(dtype) +
                             " is not supported in MPI mode.");
  }
}

MPI_Op MPIContext::GetMPISumOp(DataType dtype) {
  return dtype == HOROVOD_FLOAT16 ? mpi_float16_sum : MPI_SUM;
}

MPI_Comm MPIContext::GetMPICommunicator(Communicator comm) {
  switch (comm) {
    case GLOBAL:
      return mpi_comm;
    case LOCAL:
      return local_comm;
    case CROSS:
      return cross_comm;
    default:
      throw std::logic_error("Communicator " + CommunicatorName(comm) +
                             " is not supported in MPI mode.");
  }
}

int MPIContext::GetMPITypeSize(DataType dtype) {
  int out;
  MPI_Type_size(GetMPIDataType(dtype), &out);
  return out;
}

} // namespace common
} // namespace horovod
