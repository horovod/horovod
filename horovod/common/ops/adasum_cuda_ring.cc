// TODO license
#include "adasum_cuda_ring.h"

namespace horovod {
namespace common {

void Ring::InitRing(int tmp[], bool _isFat, int rank, int size) {
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
  Start();
}

AllreduceMessage::AllreduceMessage(MPIContext* mpi_context)
  : Message(mpi_context) {
}

void AllreduceMessage::Start() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);
  if (rank == ring_starter_rank) {
    MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
  } else {
    MPI_Irecv(recv_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
  }
}

bool AllreduceMessage::Test() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);

  int flag;
  if (leg == 4)
    return true;
  MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
  if (flag == 1) {
    leg++;
    if (leg == 4) {
      ring->ReduceLoad(count);
      return true;
    }
    if (leg == 1) {
      if (rank == ring_starter_rank) {
        MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
      } else {
        // call the cuda kernel
        switch(datatype) {
          case HOROVOD_FLOAT16:
            AdasumCudaReduce(count, (uint16_t*)grad_buf, (uint16_t*)recv_buf);
            break;
          case HOROVOD_FLOAT32:
            AdasumCudaReduce(count, (float*)grad_buf, (float*)recv_buf);
            break;
          case HOROVOD_FLOAT64:
            AdasumCudaReduce(count, (double*)grad_buf, (double*)recv_buf);
            break;
          default:
            throw std::logic_error("Message::Test: Unsupported data type.");
        }
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
      }
    } else if (leg == 2) {
      if (rank == ring_starter_rank) {
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
      } else {
        MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
      }
    } else if (leg == 3) {
      if (rank == ring_starter_rank) {
        MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
      } else {
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
      }
    }
  }

  return false;
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
      if (rank == ring_starter_rank) {
        MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
      } else {
        // call the cuda kernel
        switch(datatype) {
          case HOROVOD_FLOAT16:
            AdasumCudaReduce(count, (uint16_t*)grad_buf, (uint16_t*)recv_buf);
            break;
          case HOROVOD_FLOAT32:
            AdasumCudaReduce(count, (float*)grad_buf, (float*)recv_buf);
            break;
          case HOROVOD_FLOAT64:
            AdasumCudaReduce(count, (double*)grad_buf, (double*)recv_buf);
            break;
          default:
            throw std::logic_error("Message::Test: Unsupported data type.");
        }
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
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

AllRings::~AllRings() {
  for (int i = 0; i < messages.size(); i++)
    delete messages[i];
  delete[] rings;
}

AllRings::AllRings(int rank, int size) {
  rings = new Ring[num_rings];
  {
    // fat ring 1
    int tmp[8] = {0, 3, 2, 1, 5, 6, 7, 4};
    rings[0].InitRing(tmp, true, rank, size);
  }
  {
    // fat ring 2
    int tmp[8] = {0, 4, 7, 6, 5, 1, 2, 3};
    rings[1].InitRing(tmp, true, rank, size);
  }
  {
    // skinny ring 1
    int tmp[8] = {0, 2, 6, 4, 5, 7, 3, 1};
    rings[2].InitRing(tmp, false, rank, size);
  }
  {
    // skinny ring 2
    int tmp[8] = {0, 1, 3, 7, 5, 4, 6, 2};
    rings[3].InitRing(tmp, false, rank, size);
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
    for (auto& message : messages) {
      if (!message->Test())
        all_done = false;
    }
  }
  for (int i = 0; i < messages.size(); i++)
    delete messages[i];
  messages.clear();
}

} // common
} // horovod