// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Microsoft Corp.
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

#ifndef HOROVOD_P2P_OPERATIONS_H
#define HOROVOD_P2P_OPERATIONS_H

#include <iostream>

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "../mpi_context.h"
#include "collective_operations.h"


namespace horovod {
namespace common {

class PointToPointOp : public AllreduceOp {
public:
  PointToPointOp(MPIContext* mpi_context, HorovodGlobalState* global_state);

  virtual ~PointToPointOp() = default;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
  template<class T>
  void PointToPointSend(T* input_data_buffer,
                        int64_t buffer_length,
                        int dest_rank,
                        int tag,
                        Communicator communicator) {
    int status;                       
    if (!global_state_->msg_chunk_enabled) {
        LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" begin p2p send for tag: "<<tag;
        status = MPI_Send(input_data_buffer,
                          (int)buffer_length,
                          MPI_CHAR,
                          dest_rank,
                          tag,
                          mpi_context_->GetMPICommunicator(communicator));
        LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" end p2p send for tag: "<<tag;

    }
    else {
          const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / sizeof(T);
          for (int buf_index = 0; buf_index < buffer_length; buf_index += chunk_size) {
            status = MPI_Send((uint8_t *)input_data_buffer + buf_index,
                              std::min((int)buffer_length - buf_index, chunk_size) * sizeof(T),
                              MPI_CHAR,
                              dest_rank,
                              tag,
                              mpi_context_->GetMPICommunicator(communicator));
            status &= status;
          }
    }

    if (status != MPI_SUCCESS) {
      throw std::logic_error("MPI_Send failed, see MPI output for details.");
    }
  }

  template<class T>
  void PointToPointRecv(T* output_data_buffer,
                        int64_t buffer_length,
                        int src_rank,
                        int tag,
                        Communicator communicator) {
    int status;
    if (!global_state_->msg_chunk_enabled) {
        LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" begin p2p recv for tag: "<<tag;
        status = MPI_Recv(output_data_buffer,
                          (int)buffer_length,
                          MPI_CHAR,
                          src_rank,
                          tag,
                          mpi_context_->GetMPICommunicator(communicator),
                          MPI_STATUS_IGNORE);
        LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" end p2p recv for tag: "<<tag;
    }
    else {
          const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / sizeof(T);
          for (int buf_index = 0; buf_index < buffer_length; buf_index += chunk_size) {
            status = MPI_Recv((uint8_t *)output_data_buffer + buf_index,
                              std::min((int)buffer_length - buf_index, chunk_size) * sizeof(T),
                              MPI_CHAR,
                              src_rank,
                              tag,
                              mpi_context_->GetMPICommunicator(communicator),
                              MPI_STATUS_IGNORE);
            status &= status;
          }
    }

    if (status != MPI_SUCCESS) {
      throw std::logic_error("MPI_Recv failed, see MPI output for details.");
    }
  }

};

} // namespace common
} // namespace horovod

#endif // HOROVOD_P2P_OPERATIONS_H
