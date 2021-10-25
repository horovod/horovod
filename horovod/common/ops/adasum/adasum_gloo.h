// Copyright 2021 Predibase. All Rights Reserved.
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

#ifndef HOROVOD_ADASUM_GLOO_H
#define HOROVOD_ADASUM_GLOO_H

#include "../../controller.h"
#include "adasum.h"

namespace horovod {
namespace common {

class AdasumGloo : public Adasum<gloo::Context> {
public:
  AdasumGloo(GlooContext* gloo_context, HorovodGlobalState* global_state);

  ~AdasumMPI();

protected:
  void InitializeVHDDReductionComms();

  void PointToPointSendRecv(void* input_data_buffer,
                            int64_t input_buffer_length,
                            void* output_data_buffer,
                            int64_t output_buffer_length,
                            DataType horovod_datatype, int dst_src_rank,
                            int tag, gloo::Context communicator,
                            HorovodGlobalState* global_state) override;

  int GetLocalRankWithComm(gloo::Context local_comm) override;

  int GetSizeWithComm(gloo::Context comm) override;

  void SumAllreduceWithComm(std::vector<TensorTableEntry>& entries, void* data,
                            int num_elements, DataType horovod_datatype,
                            gloo::Context comm,
                            HorovodGlobalState* global_state) override;

  GlooContext* gloo_context_;
  // Gloo communicators used to do adasum
  gloo::Context* reduction_comms_ = nullptr;
  // Flag to indicate if reduction comms have been initialized
  bool reduction_comms_initialized = false;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_ADASUM_MPI_H
