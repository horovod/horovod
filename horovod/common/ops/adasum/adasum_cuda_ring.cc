// Copyright 2019 Microsoft. All Rights Reserved.
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

#include "adasum_cuda_ring.h"

namespace horovod {
namespace common {

void Ring::InitRing(int tmp[], bool _isFat, int rank, int size, cudaStream_t _adasum_stream) {
  load = 0;
  isFat = _isFat;
  for (int i = 0; i < 8; i++)
    loop[i] = tmp[i];

  for (int j = 0; j < size; j++) { // go through allranks
    if (rank == loop[j]) {
      prevGPU = loop[(j-1+size) % size];
      nextGPU = loop[(j+1+size) % size];
    }
  }
  adasum_stream = _adasum_stream;
}

int Ring::GetAfterLoad(int message_len) {
  if (!isFat)
    return 2*(load+message_len);
  else
    return (load+message_len);
}

void Ring::AddLoad(int message_len) {
  load += message_len;
}

void Ring::ReduceLoad(int message_len) {
  load -= message_len;
}

Message::Message(MPIContext* mpi_context)
  : mpi_context(mpi_context) {
}

void Message::InitMessage(Ring* _ring, int _rank, int _ring_starter_rank, int _count, void* _grad_buf, void* _recv_buf, DataType _datatype, MPI_Comm _comm, int _tag) {
  comm = _comm;
  count = _count;
  tag = _tag;
  ring = _ring;
  rank = _rank;
  ring_starter_rank = _ring_starter_rank;
  leg = 0;
  grad_buf = _grad_buf;
  recv_buf = _recv_buf;
  datatype = _datatype;
  cuda_stream_sync = false;
  Start();
}

ReduceMessage::ReduceMessage(MPIContext* mpi_context)
  : Message(mpi_context) {
}

void ReduceMessage::Start() {

  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);
  if (rank == ring_starter_rank) {
    MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
  } else {
    MPI_Irecv(recv_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
  }
}

bool ReduceMessage::Test() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);

  int flag = 0;
  if (leg == 2)
    return true;
  if (!cuda_stream_sync) {
    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
  }
  if (flag == 1 || cuda_stream_sync) {
    leg++;
    if (leg == 2) {
      ring->ReduceLoad(count);
      return true;
    }
    if (leg == 1) {
      if (rank == ring_starter_rank) {
        MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
      } else if (!cuda_stream_sync) {
        // call the cuda kernel
        switch(datatype) {
          case HOROVOD_FLOAT16:
            AdasumCudaReduce(count, (uint16_t*)grad_buf, (uint16_t*)recv_buf, ring->adasum_stream);
            break;
          case HOROVOD_FLOAT32:
            AdasumCudaReduce(count, (float*)grad_buf, (float*)recv_buf, ring->adasum_stream);
            break;
          case HOROVOD_FLOAT64:
            AdasumCudaReduce(count, (double*)grad_buf, (double*)recv_buf, ring->adasum_stream);
            break;
          default:
            throw std::logic_error("Message::Test: Unsupported data type.");
        }
        cuda_stream_sync = true;
        leg--;
      } else {
        cudaStreamSynchronize(ring->adasum_stream);
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
        cuda_stream_sync = false;
      }
    }
  }
  return false;
}

BroadcastMessage::BroadcastMessage(MPIContext* mpi_context)
  : Message(mpi_context) {
}

void BroadcastMessage::Start() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);
  if (rank == ring_starter_rank) {
    MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
    leg = 1;
  } else {
    MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
    if (ring->nextGPU == ring_starter_rank)
      leg = 1;
  }
}

bool BroadcastMessage::Test() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);

  int flag;
  if (leg == 2)
    return true;
  MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
  if (flag == 1) {
    leg++;
    if (leg == 2) {
      ring->ReduceLoad(count);
      return true;
    }
    if (leg == 1) {
      if (rank != ring_starter_rank) {
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
      }
    }
  }
  return false;
}

std::vector<std::vector<int>> AllRings::GetTopologyMatrix(int size) {
  std::vector<std::vector<int>> topology_matrix(size, std::vector<int>(size));
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (i == j) {
        topology_matrix[i][j] = -1;
        continue;
      }
      int accessSupported = 0;
      cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported,
                                i, j);
      if (!accessSupported) {
        topology_matrix[i][j] = -1;
        continue;
      }
      int perfRank = 0;
      cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, i, j);
      topology_matrix[i][j] = perfRank;
    }
  }
  return topology_matrix;
}

std::vector<int>
AllRings::GetRingFromTopologyMatrix(int size,
                                    const std::vector<std::vector<int>>& topology_matrix,
                                    int performance_rank) {
  // Every node in topology matrix
  // must have 2 connections for each performance rank.
  //
  // Typical topology matrix example:
  //
  // {{-1, 1, 1, 0, 0, -1, -1, -1},
  //  {1, -1, 0, 1, -1, 0, -1, -1},
  //  {1, 0, -1, 0, -1, -1, 1, -1},
  //  {0, 1, 0, -1, -1, -1, -1, 1},
  //  {0, -1, -1, -1, -1, 1, 1, 0},
  //  {-1, 0, -1, -1, 1, -1, 0, 1},
  //  {-1, -1, 1, -1, 1, 0, -1, 0},
  //  {-1, -1, -1, 1, 0, 1, 0, -1}}

  assert(topology_matrix.size() == size);
  assert(topology_matrix[0].size() == size);

  // Start point doesn't matter, let's start with 0.
  std::vector<int> ring;
  int curr = 0;
  int prev = -1;
  do {
    for (int i = 0; i < size; i++) {
      if (i != prev && topology_matrix[curr][i] == performance_rank) {
        ring.push_back(curr);
        prev = curr;
        curr = i;
        break;
      }
    }
  } while (curr != 0);
  return ring;
}

AllRings::~AllRings() {
  for (int i = 0; i < messages.size(); i++)
    delete messages[i];
  delete[] rings;
}

AllRings::AllRings(int rank, int size) {
  rings = new Ring[num_rings];
  auto topology_matrix = GetTopologyMatrix(size);

  int greatest_priority;
  cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority);
  cudaStreamCreateWithPriority(&adasum_stream, cudaStreamNonBlocking, greatest_priority);

  auto fat_ring = GetRingFromTopologyMatrix(size, topology_matrix, 0);
  if (fat_ring.empty()) {
    LOG(INFO) << "No rings created from topology matrix, use defaults.";
    {
      // fat ring 1
      int tmp[8] = {0, 3, 2, 1, 5, 6, 7, 4};
      rings[0].InitRing(tmp, true, rank, size, adasum_stream);
    }
    {
      // fat ring 2
      int tmp[8] = {0, 4, 7, 6, 5, 1, 2, 3};
      rings[1].InitRing(tmp, true, rank, size, adasum_stream);
    }
    {
      // skinny ring 1
      int tmp[8] = {0, 2, 6, 4, 5, 7, 3, 1};
      rings[2].InitRing(tmp, false, rank, size, adasum_stream);
    }
    {
      // skinny ring 2
      int tmp[8] = {0, 1, 3, 7, 5, 4, 6, 2};
      rings[3].InitRing(tmp, false, rank, size, adasum_stream);
    }
  } else {
    rings[0].InitRing(fat_ring.data(), true, rank, size, adasum_stream);
    std::reverse(fat_ring.begin(), fat_ring.end());
    rings[1].InitRing(fat_ring.data(), true, rank, size, adasum_stream);

    auto skinny_ring = GetRingFromTopologyMatrix(size, topology_matrix, 1);
    assert(!skinny_ring.empty());
    rings[2].InitRing(skinny_ring.data(), false, rank, size, adasum_stream);
    std::reverse(skinny_ring.begin(), skinny_ring.end());
    rings[3].InitRing(skinny_ring.data(), false, rank, size, adasum_stream);
  }
};

Ring* AllRings::PickRing(int count) {
  int min_load = (1<<30); // INF
  Ring* ret_ring = NULL;
  for (int i = 0; i < num_rings; i++) {
    Ring* ring = &rings[i];
    int cur_ring_after_load = ring->GetAfterLoad(count);
    if (cur_ring_after_load < min_load) {
      ret_ring = ring;
      min_load = cur_ring_after_load;
    }
  }
  ret_ring->AddLoad(count);
  assert(ret_ring != NULL);
  return ret_ring;
}

void AllRings::InitMessageInRing(Message* message, void* grad_buf, void* recv_buf, int size, DataType datatype, MPI_Comm comm, int grad_tag, int rank) {
  int count = -1;
  switch(datatype) {
    case HOROVOD_FLOAT16:
      count = size / sizeof(uint16_t);
      break;
    case HOROVOD_FLOAT32:
      count = size / sizeof(float);
      break;
    case HOROVOD_FLOAT64:
      count = size / sizeof(double);
      break;
    default:
      throw std::logic_error("AllRings::InitMessageInRing: Unsupported data type.");
  }
  messages.push_back(message);
	Ring*	ring = PickRing(count);
  message->InitMessage(ring, rank, grad_tag % 8, count, grad_buf, recv_buf, datatype, comm, grad_tag);
}

void AllRings::WaitAllMessages() {
  
  bool all_done = false;
  while (!all_done) {
    all_done = true;
    for (int i = 0; i < messages.size(); i++) {
      if (!messages.at(i)->Test())
        all_done = false;
    }
  }
  for (int i = 0; i < messages.size(); i++)
    delete messages[i];
  messages.clear();
}
} // common
} // horovod