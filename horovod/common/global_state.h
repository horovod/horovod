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

#ifndef HOROVOD_GLOBAL_STATE_H
#define HOROVOD_GLOBAL_STATE_H

#include <queue>
#include <thread>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include "fusion_buffer_manager.h"
#include "parameter_manager.h"
#include "response_cache.h"
#include "tensor_queue.h"
#include "timeline.h"
#include "utils/env_parser.h"
#include "logging.h"
// TODO: This will need to be moved somewhere else to remove dependency on MPI
#include "mpi.h"

namespace horovod {
namespace common {

// Forward declaration
class Controller;

// The global state shared by threads.
//
// MPI is a library that stores a lot of global per-program state and often
// requires running on a single thread. As a result, we have to have a single
// background thread responsible for all MPI operations, and communicate with
// that background thread through global state.
struct HorovodGlobalState {
  // An atomic boolean which is set to true when background thread is started.
  // This ensures that only one background thread is spawned.
  std::atomic_flag initialize_flag = ATOMIC_FLAG_INIT;

  // Tensors waiting to be allreduced or allgathered.
  TensorTable tensor_table;  

  // Thread pool
  boost::asio::thread_pool* background_thread_pool;
  
  //flag to indicate usage of AdaSum reduction algorithm
  bool adasum_enabled = false;
  
  // Counter used to keep track of how many of the parallel reductions finished
  // TODO do we need this?
  std::atomic_int finished_parallel_reductions;

  // Encapsulates the temp buffers used for AdaSum.
  std::queue<FusionBufferManager> temp_buffers;

  // Mutex to be used when accessing the queue of temp buffers
  std::mutex buffer_lock;

  // threads to be used for AdaSum operations
  int num_adasum_threads;

  HorovodGlobalState() {
    auto horovod_number_of_threads = std::getenv(HOROVOD_NUM_OF_ADASUM_REDUCTION_THREADS);
    auto adasum = std::getenv(HOROVOD_ADASUM_ENABLE);
    if (adasum != nullptr) {
      int adasum_value = std::strtol(adasum, nullptr, 10);
      adasum_enabled = adasum_value == 1;
    }
    if (adasum_enabled == true) {
      int num_threads;
      if (horovod_number_of_threads != nullptr){
        num_threads = std::strtol(horovod_number_of_threads, nullptr, 10);
        LOG(INFO)<<"HOROVOD_NUM_OF_ADASUM_REDUCTION_THREADS is set to "<<num_threads;
        if (num_threads <= 0){
          throw std::logic_error("Number of threads must be greater or equal to 1 when AdaSum is used.");
        }
      }
      else {
        LOG(INFO)<<"HOROVOD_NUM_OF_ADASUM_REDUCTION_THREADS is not set. Creating threadpool with 1 thread by default. ";
        num_threads = 1;
      }
      //Making this static so that this pool is preverved throughout the lifetime of the program
      LOG(INFO)<<"Starting "<<num_threads<<" MPI threads for threadpool.";
      static boost::asio::thread_pool pool(num_threads);
      num_adasum_threads = num_threads;
      // Create a buffer manager for temp buffers for each thread
      for (int i = 0; i < num_threads; ++i) {
        temp_buffers.emplace();
      }
      background_thread_pool = &pool;
    }
  }
  
  // Background thread running MPI communication.
  std::thread background_thread;

  // MPI communicators used to do msallreduction
  // TODO put this in a better place
  MPI_Comm* reduction_comms;

  //TODO find a better place
  int rank_log_size = 0;
  
  // TODO find a better place
  MPI_Comm local_comm;

  // TODO better place
  bool msg_chunk_enabled = false;

  // Whether the background thread should shutdown.
  std::atomic_bool shut_down{false};

  // Timeline writer.
  Timeline timeline;

  // Flag indicating whether timeline enabled.
  bool timeline_enabled = false;

  // Flag indicating whether to mark cycles in the timeline.
  bool mark_cycles_in_timeline = false;

  ParameterManager parameter_manager;

  // Encapsulates the fusion buffers, handles resizing and auto-tuning of buffer
  // size.
  FusionBufferManager fusion_buffer;
  
  // Time point when last cycle started.
  std::chrono::steady_clock::time_point last_cycle_start;

  // Whether collective context has been completed on the background thread.
  std::atomic_bool initialization_done{false};

  std::shared_ptr<Controller> controller;

  TensorQueue tensor_queue;

  // Pointer to shared buffer for allgather
  void* shared_buffer = nullptr;

  // Current shared buffer size
  int64_t shared_buffer_size = 0;

  // LRU cache of Responses
  ResponseCache response_cache;

  // Number of responses that can be cached
  uint32_t cache_capacity = 1024;

  // Number of CUDA streams to use
  int num_nccl_streams = 1;

  // Index of current CUDA stream to use
  int current_nccl_stream = 0;

  // A LibType indicating what framework we are using to perform CPU operations.
  LibType cpu_operation;

  // A LibType indicating what framework we are using to perform controller
  // operations.
  LibType control_operation;

  ~HorovodGlobalState() {
    // Make sure that the destructor of the background thread is safe to
    // call. If a thread is still joinable (not detached or complete) its
    // destructor cannot be called.
    if (background_thread.joinable()) {
      shut_down = true;
      background_thread.join();
    }
    //TODO merge this with background thread
    if(background_thread_pool != nullptr){
      background_thread_pool->stop();
    }

    delete reduction_comms;
  }
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_GLOBAL_STATE_H
