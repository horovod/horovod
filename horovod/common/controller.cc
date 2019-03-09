// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

#include <atomic>
#include <map>
#include <queue>
#include <set>
#include <unordered_set>

#include "controller.h"
#include "logging.h"
#include "operations.h"

namespace horovod {
namespace common {

// This function performs all the preparation work for workers to agree
// on what tensors to be all-reduced or all-gathered. The output is a
// response list that includes all tensors that are ready.
//
// The coordinator follows a master-worker paradigm. Rank zero acts
// as the master (the "coordinator"), whereas all other ranks are simply
// workers. Each worker maintains a cache of tensors that are previously
// broadcasted as ready by other ranks. If the cache covers all incoming
// messages, there's no need for workers to do additional communications.
// Otherwise, workers will communicate with each other to agree on what
// tensors to be processed. The communication performs as following:
//
//      a) The workers send a Request to the coordinator, indicating what
//      they would like to do (which tensor they would like to gather and
//      reduce, as well as their shape and type). They repeat this for every
//      tensor that they would like to operate on.
//
//      b) The workers send an empty "DONE" message to the coordinator to
//      indicate that there are no more tensors they wish to operate on.
//
//      c) The coordinator receives the Requests from the workers, as well
//      as from its own TensorFlow ops, and stores them in a request table.
//      The coordinator continues to receive Request messages until it has
//      received GLOBAL_SIZE number of empty "DONE" messages.
//
//      d) The coordinator finds all tensors that are ready to be reduced,
//      gathered, or all operations that result in an error. For each of
//      those, it sends a Response to all the workers. When no more
//      Responses are available, it sends a "DONE" response to the workers.
//      If the process is being shutdown, it instead sends a "SHUTDOWN"
//      response.
//
//      e) The workers listen for Response messages, processing each one by
//      doing the required reduce or gather, until they receive a "DONE"
//      response from the coordinator. At that point, the tick ends.
//      If instead of "DONE" they receive "SHUTDOWN", they mark it in the
//      response list.
ResponseList Controller::PrepareForOps(
    bool timeline_enabled, Timeline& timeline, ResponseCache& response_cache,
    ParameterManager& param_manager, std::queue<Request>& state_message_queue,
    std::mutex& mutex, std::atomic_bool& shut_down,
    std::shared_ptr<MessageTable> message_table, TensorTable& tensor_table) {

  // Update cache capacity if autotuning is active.
  if (param_manager.IsAutoTuning()) {
    response_cache.set_capacity((int)param_manager.CacheEnabled() *
                                cache_capacity_);
  }

  // Copy the data structures out from parameters.
  // However, don't keep the lock for the rest of the loop, so that
  // enqueued stream callbacks can continue.

  CacheCoordinator cache_coordinator(response_cache.num_active_bits());

  std::queue<Request> message_queue;
  {
    std::lock_guard<std::mutex> guard(mutex);
    while (!state_message_queue.empty()) {
      Request message = state_message_queue.front();
      state_message_queue.pop();
      message_queue.push(message);

      // Keep track of cache hits
      if (response_cache.capacity() > 0) {
        auto cache_ = response_cache.cached(message);
        if (cache_ == ResponseCache::CacheState::HIT) {
          uint32_t cache_bit = response_cache.peek_cache_bit(message);
          cache_coordinator.record_hit(cache_bit);

          // Record initial time cached tensor is encountered in queue.
          if (stall_inspector_.perform_stall_check &&
              stall_inspector_.cache_tensor_start.find(message.tensor_name()) ==
                  stall_inspector_.cache_tensor_start.end()) {
            stall_inspector_.cache_tensor_start[message.tensor_name()] =
                std::chrono::steady_clock::now();
          }

        } else {
          if (cache_ == ResponseCache::CacheState::INVALID) {
            uint32_t cache_bit = response_cache.peek_cache_bit(message);
            cache_coordinator.record_invalid_bit(cache_bit);
          }
          cache_coordinator.set_uncached_in_queue(true);

          // Remove timing entry if uncached or marked invalid.
          if (stall_inspector_.perform_stall_check) {
            stall_inspector_.cache_tensor_start.erase(message.tensor_name());
          }
        }
      }
    }
  }

  // Flag indicating that the background thread should shut down.
  bool should_shut_down = shut_down;

  // Check for stalled tensors.
  if (stall_inspector_.perform_stall_check &&
      std::chrono::steady_clock::now() - stall_inspector_.last_stall_check >
          std::chrono::seconds(stall_inspector_.stall_warning_time_seconds)) {
    if (is_coordinator_) {
      should_shut_down |= CheckForStalledTensors(message_table);
    }

    if (response_cache.capacity() > 0) {
      InvalidateStalledCachedTensors(cache_coordinator, response_cache);
    }
    stall_inspector_.last_stall_check = std::chrono::steady_clock::now();
  }

  cache_coordinator.set_should_shut_down(should_shut_down);

  if (response_cache.capacity() > 0) {
    // Obtain common cache hits and cache invalidations across workers. Also,
    // determine if any worker has uncached messages in queue or requests
    // a shutdown. This function removes any invalid cache entries, if they
    // exist.
    CoordinateCacheAndState(cache_coordinator, timeline_enabled, timeline,
                            response_cache);

    {
      // Get lock in order to safely replace messages to global queue
      std::lock_guard<std::mutex> guard(mutex);

      // Remove uncommon cached tensors from queue and replace to state
      // queue for next cycle. Skip adding common cached tensors to
      // queue as they are handled separately.
      size_t num_messages = message_queue.size();
      for (size_t i = 0; i < num_messages; ++i) {
        auto message = message_queue.front();
        if (response_cache.cached(message) == ResponseCache::CacheState::HIT) {
          uint32_t cache_bit = response_cache.peek_cache_bit(message);
          if (cache_coordinator.cache_hits().find(cache_bit) ==
              cache_coordinator.cache_hits().end()) {
            // Try to process again in next cycle.
            state_message_queue.push(std::move(message));
          } else if (stall_inspector_.perform_stall_check) {
            // Remove timing entry for messages being handled this cycle.
            stall_inspector_.cache_tensor_start.erase(message.tensor_name());
          }
        } else {
          // Remove timing entry for messages being handled this cycle.
          if (stall_inspector_.perform_stall_check) {
            stall_inspector_.cache_tensor_start.erase(message.tensor_name());
          }
          message_queue.push(std::move(message));
        }
        message_queue.pop();
      }
    }
  }

  if (!message_queue.empty()) {
    LOG(DEBUG, rank_) << "Sent " << message_queue.size()
                      << " messages to coordinator.";
  }

  ResponseList response_list;
  response_list.set_shutdown(cache_coordinator.should_shut_down());

  bool need_communication = true;
  if (response_cache.capacity() > 0 && !cache_coordinator.uncached_in_queue()) {
    // if cache is enabled and no uncached new message coming in, no need for
    // additional communications
    need_communication = false;

    // If no messages to send, we can simply return an empty response list;
    if (cache_coordinator.cache_hits().empty()) {
      return response_list;
    }
    // otherwise we need to add cached messages to response list.
  }

  if (!need_communication) {
    // If all messages in queue have responses in cache, use fast path with
    // no additional coordination.

    std::deque<Response> responses;
    // Convert cache hits to responses. Populate so that least
    // recently used responses get priority. All workers call the code
    // here so we use the get method here to consistently update the cache
    // order.
    for (auto bit : cache_coordinator.cache_hits()) {
      responses.push_back(response_cache.get_response(bit));
    }

    // Fuse responses as normal.
    response_list =
        FuseResponses(responses, mutex, tensor_table, param_manager);
  } else {
    // There are uncached messages coming in, need communication to figure out
    // whether those are ready to be reduced.

    // Collect all tensors that are ready to be reduced. Record them in the
    // tensor count table (rank zero) or send them to rank zero to be
    // recorded (everyone else).
    std::vector<std::string> ready_to_reduce;

    if (is_coordinator_) {
      while (!message_queue.empty()) {
        // Pop the first available message
        Request message = message_queue.front();
        message_queue.pop();

        bool reduce =
            IncrementTensorCount(message_table, message, size_, timeline);
        if (reduce) {
          ready_to_reduce.push_back(message.tensor_name());
        }
      }

      // Receive ready tensors from other ranks
      bool others_shut_down =
          RecvReadyTensors(ready_to_reduce, message_table, timeline);
      should_shut_down |= others_shut_down;

      // At this point, rank zero should have a fully updated tensor count
      // table and should know all the tensors that need to be reduced or
      // gathered, and everyone else should have sent all their information
      // to rank zero. We can now do reductions and gathers; rank zero will
      // choose which ones and in what order, and will notify the other ranks
      // before doing each reduction.
      std::deque<Response> responses;

      if (response_cache.capacity() > 0) {
        // Prepopulate response list with cached responses. Populate so that
        // least recently used responses get priority. Since only the
        // coordinator rank calls this code, use peek instead of get here to
        // preserve cache order across workers.
        for (auto bit : cache_coordinator.cache_hits()) {
          responses.push_back(response_cache.peek_response(bit));
        }
      }

      for (auto& tensor_name : ready_to_reduce) {
        Response response = ConstructResponse(message_table, tensor_name);
        responses.push_back(std::move(response));
      }

      response_list =
          FuseResponses(responses, mutex, tensor_table, param_manager);
      response_list.set_shutdown(should_shut_down);

      // Broadcast final results to other tensors
      SendFinalTensors(response_list);

    } else {
      RequestList message_list;
      message_list.set_shutdown(should_shut_down);
      while (!message_queue.empty()) {
        message_list.add_request(message_queue.front());
        message_queue.pop();
      }

      // Send ready tensors to rank zero
      SendReadyTensors(message_list);

      // Receive final tensors to be processed from rank zero
      RecvFinalTensors(response_list);
    }
  }

  if (!response_list.responses().empty()) {
    std::string tensors_ready;
    for (const auto& r : response_list.responses()) {
      tensors_ready += r.tensor_names_string() + "; ";
    }
    LOG(TRACE) << "Sending ready responses as " << tensors_ready;
  }

  // If need_communication is false, meaning no uncached message coming in,
  // thus no need to update cache.
  if (need_communication && response_cache.capacity() > 0) {
    std::lock_guard<std::mutex> guard(mutex);
    // All workers add supported responses to cache. This updates the cache
    // order consistently across workers.
    for (auto& response : response_list.responses()) {
      if (response.response_type() == Response::ResponseType::ALLREDUCE &&
          (int)response.devices().size() == size_) {
        response_cache.put(response, tensor_table);
      }
    }
  }

  // Reassign cache bits based on current cache order.
  response_cache.update_cache_bits();

  return response_list;
}

int64_t
Controller::TensorFusionThresholdBytes(ParameterManager& param_manager) {
  int64_t proposed_fusion_threshold =
      param_manager.TensorFusionThresholdBytes();

  // If the cluster is homogeneous and hierarchical allreduce is enabled,
  // adjust buffer size to make sure it is divisible by local_size to improve
  // performance.
  if (is_homogeneous_ && param_manager.HierarchicalAllreduce()) {
    // Assume the worst-case data type float64, since if it is divisible with
    // float64, it will be divisible for other types too.

    // Ensuring that fusion buffer can hold a number of elements divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for performance
    int double_size = GetTypeSize(HOROVOD_FLOAT64);
    int64_t div = local_size_ * double_size * FUSION_BUFFER_ATOMIC_UNIT;
    return ((proposed_fusion_threshold + div - 1) / div) * div;
  }
  return proposed_fusion_threshold;
}

// Once a tensor is ready to be reduced, the coordinator sends a Response
// instructing all ranks to start the reduction to all ranks. The Response
// also contains error messages in case the submitted Requests were not
// valid (for example, contained mismatched shapes or types).
//
// Constructing the Response, thus, requires a whole lot of error checking.
Response
Controller::ConstructResponse(std::shared_ptr<MessageTable> message_table,
                              std::string& name) {
  bool error = false;
  auto it = message_table->find(name);
  assert(it != message_table->end());

  std::vector<Request>& requests = std::get<0>(it->second);
  assert(!requests.empty());

  std::ostringstream error_message_stream;

  // Check that all data types of tensors being reduced, gathered or broadcasted
  // are identical.
  auto data_type = requests[0].tensor_type();
  for (unsigned int i = 1; i < requests.size(); ++i) {
    auto request_type = requests[i].tensor_type();
    if (data_type != request_type) {
      error = true;
      error_message_stream << "Mismatched data types: One rank had type "
                           << DataType_Name(data_type)
                           << ", but another rank had type "
                           << DataType_Name(request_type) << ".";
      break;
    }
  }

  // Check that all requested operations are the same
  auto message_type = requests[0].request_type();
  for (unsigned int i = 1; i < requests.size(); ++i) {
    if (error) {
      break;
    }

    auto request_type = requests[i].request_type();
    if (message_type != request_type) {
      error = true;
      error_message_stream << "Mismatched operations: One rank did an "
                           << Request::RequestType_Name(message_type)
                           << ", but another rank did an "
                           << Request::RequestType_Name(request_type) << ".";
      break;
    }
  }

  // If we are doing an allreduce or broadcast, check that all tensor shapes are
  // identical.
  if (message_type == Request::ALLREDUCE ||
      message_type == Request::BROADCAST) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }
    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape != request_shape) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of shape "
            << tensor_shape.DebugString()
            << ", but another rank sent a tensor of shape "
            << request_shape.DebugString() << ".";
        break;
      }
    }
  }

  // If we are doing an allgather, make sure all but the first dimension are
  // the same. The first dimension may be different and the output tensor is
  // the sum of the first dimension. Collect the sizes by rank.
  std::vector<int64_t> tensor_sizes(requests.size());
  if (message_type == Request::ALLGATHER) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }

    if (tensor_shape.dims() == 0) {
      error = true;
      error_message_stream << "Rank zero tried to "
                           << Request::RequestType_Name(message_type)
                           << " a rank-zero tensor.";
    } else {
      tensor_sizes[requests[0].request_rank()] = tensor_shape.dim_size(0);
    }

    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape.dims() != request_shape.dims()) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of rank "
            << tensor_shape.dims()
            << ", but another rank sent a tensor of rank "
            << request_shape.dims() << ".";
        break;
      }

      bool dim_mismatch = false;
      for (int dim = 1; dim < tensor_shape.dims(); ++dim) {
        if (tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
          error = true;
          error_message_stream
              << "Mismatched " << Request::RequestType_Name(message_type)
              << " tensor shapes: One rank sent a tensor with dimension " << dim
              << " equal to " << tensor_shape.dim_size(dim)
              << ", but another rank sent a tensor with dimension " << dim
              << " equal to " << request_shape.dim_size(dim) << ".";
          dim_mismatch = true;
          break;
        }
      }
      if (dim_mismatch) {
        break;
      }

      tensor_sizes[requests[i].request_rank()] = request_shape.dim_size(0);
    }
  }

  // If we are doing a broadcast, check that all root ranks are identical.
  if (message_type == Request::BROADCAST) {
    int first_root_rank = requests[0].root_rank();
    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      int this_root_rank = requests[i].root_rank();
      if (first_root_rank != this_root_rank) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " root ranks: One rank specified root rank " << first_root_rank
            << ", but another rank specified root rank " << this_root_rank
            << ".";
        break;
      }
    }
  }

  bool first_device_is_cpu = requests[0].device() == CPU_DEVICE_ID;
  for (unsigned int i = 1; i < requests.size(); ++i) {
    if (error) {
      break;
    }

    bool this_device_is_cpu = requests[i].device() == CPU_DEVICE_ID;
    if (first_device_is_cpu != this_device_is_cpu) {
      error = true;
      error_message_stream
          << "Mismatched " << Request::RequestType_Name(message_type)
          << " CPU/GPU device selection: One rank specified device "
          << (first_device_is_cpu ? "CPU" : "GPU")
          << ", but another rank specified device "
          << (this_device_is_cpu ? "CPU" : "GPU") << ".";
      break;
    }
  }
  std::vector<int32_t> devices(requests.size());
  for (auto& request : requests) {
    devices[request.request_rank()] = request.device();
  }

  Response response;
  response.add_tensor_name(name);
  if (error) {
    std::string error_message = error_message_stream.str();
    response.set_response_type(Response::ERROR);
    response.set_error_message(error_message);
  } else if (message_type == Request::ALLGATHER) {
    response.set_response_type(Response::ALLGATHER);
    for (auto dim : tensor_sizes) {
      response.add_tensor_size(dim);
    }
  } else if (message_type == Request::ALLREDUCE) {
    response.set_response_type(Response::ALLREDUCE);
  } else if (message_type == Request::BROADCAST) {
    response.set_response_type(Response::BROADCAST);
  }
  response.set_devices(devices);

  // Clear all queued up requests for this name. They are now taken care of
  // by the constructed response.
  message_table->erase(it);

  return response;
}

// Report Tensors that were submitted to be reduced, gathered or broadcasted by
// some ranks but not others and are waiting for long time to get processed.
bool Controller::CheckForStalledTensors(
    std::shared_ptr<MessageTable> message_table) {
  bool should_shut_down = false;
  auto now = std::chrono::steady_clock::now();
  std::map<int32_t, std::set<std::string>> missing_ranks;
  std::unordered_set<int32_t> shutdown_ranks;
  std::chrono::seconds stall_warning_time(
      stall_inspector_.stall_warning_time_seconds);
  std::chrono::seconds stall_shutdown_time(
      stall_inspector_.stall_shutdown_time_seconds);

  if (stall_shutdown_time > std::chrono::seconds(0) &&
      stall_shutdown_time < stall_warning_time) {
    LOG(WARNING) << "HOROVOD_STALL_SHUTDOWN_TIME_SECONDS is less than "
                    "HOROVOD_STALL_CHECK_TIME_SECONDS, will not shutdown.";
    stall_shutdown_time = std::chrono::seconds(0);
  }

  for (auto& m : *message_table) {
    auto tensor_name = m.first;
    std::vector<Request>& messages = std::get<0>(m.second);
    std::chrono::steady_clock::time_point start_at = std::get<1>(m.second);
    auto lag = now - start_at;

    if (lag > stall_warning_time) {
      std::unordered_set<int32_t> ready_ranks;
      for (auto msg_iter = messages.begin(); msg_iter != messages.end();
           ++msg_iter) {
        ready_ranks.insert(msg_iter->request_rank());
      }

      for (int32_t rank = 0; rank < size_; ++rank) {
        if (ready_ranks.find(rank) == ready_ranks.end()) {
          missing_ranks[rank].insert(tensor_name);
          if (stall_shutdown_time > std::chrono::seconds(0) &&
              lag > stall_shutdown_time) {
            shutdown_ranks.insert(rank);
            should_shut_down = true;
          }
        }
      }
    }
  }

  if (!missing_ranks.empty()) {
    std::stringstream message;
    message << "One or more tensors were submitted to be "
               "reduced, gathered or broadcasted by subset of ranks and "
               "are waiting for remainder of ranks for more than "
            << stall_warning_time.count() << " seconds. "
            << "This may indicate that different ranks are trying to "
               "submit different tensors or that only subset of ranks is "
               "submitting tensors, which will cause deadlock. "
            << std::endl
            << "Stalled ranks:";
    for (auto& kv : missing_ranks) {
      message << std::endl << kv.first;
      if (shutdown_ranks.find(kv.first) != shutdown_ranks.end()) {
        message << "!";
      }

      message << ": [";
      auto it = kv.second.begin();
      message << *it;
      int count = 0;
      while (++it != kv.second.end()) {
        message << ", " << *it;
        if (++count == 5) {
          message << " ...";
          break;
        }
      }

      message << "]";
    }

    if (should_shut_down) {
      message
          << std::endl
          << "One or more rank (marked by \"!\") is stalled for longer than "
          << stall_shutdown_time.count() << " seconds. Will shutdown.";
      LOG(ERROR) << message.str();
    } else {
      LOG(WARNING) << message.str();
    }
  }

  return should_shut_down;
}

// Routine to sync cache hit and invalid bit sets across workers.
// Also determines global shutdown state and whether uncached requests
// exist on any worker.
void Controller::CoordinateCacheAndState(CacheCoordinator& cache_coordinator,
                                         bool timeline_enabled,
                                         Timeline& timeline,
                                         ResponseCache& response_cache) {

  // Sync cache and state information across workers.
  cache_coordinator.sync(this, timeline_enabled);

  // If invalid cache entries exist, erase associated entries.
  if (!cache_coordinator.invalid_bits().empty()) {
    for (auto bit : cache_coordinator.invalid_bits()) {
      response_cache.erase_response(bit);
    }
  }

  if (timeline_enabled) {
    // Start/continue negotiation phase on timeline bit entries.
    for (auto bit : cache_coordinator.timeline_bits()) {
      auto& response = response_cache.peek_response(bit);
      timeline.NegotiateStart(response.tensor_names()[0],
                              (Request::RequestType)response.response_type());
    }

    // End negotation phase for synced cache hit set entries.
    for (auto bit : cache_coordinator.cache_hits()) {
      auto& response = response_cache.peek_response(bit);
      timeline.NegotiateEnd(response.tensor_names()[0]);
    }
  }
}

// Invalidate cached tensors that have been pending for a long time.
void Controller::InvalidateStalledCachedTensors(
    CacheCoordinator& cache_coordinator, ResponseCache& response_cache) {
  auto now = std::chrono::steady_clock::now();
  std::chrono::seconds stall_warning_time(
      stall_inspector_.stall_warning_time_seconds);

  for (auto& entry : stall_inspector_.cache_tensor_start) {
    // If pending time for cached tensor exceeds stall_warning_time, mark entry
    // for global removal from cache to trigger stall messaging.
    if (now - entry.second > stall_warning_time) {
      uint32_t cache_bit = response_cache.peek_cache_bit(entry.first);
      cache_coordinator.record_invalid_bit(cache_bit);
      cache_coordinator.set_uncached_in_queue(true);
    }
  }
}

ResponseList Controller::FuseResponses(std::deque<Response>& responses,
                                       std::mutex& mutex,
                                       TensorTable& tensor_table,
                                       ParameterManager& param_manager) {
  ResponseList response_list;
  {
    // Protect access to tensor table.
    std::lock_guard<std::mutex> guard(mutex);
    while (!responses.empty()) {

      auto response = responses.front();
      assert(response.tensor_names().size() == 1);
      responses.pop_front();
      int64_t tensor_size = 0;
      if (response.response_type() == Response::ResponseType::ALLREDUCE) {
        // Attempt to add more responses to this fused response.
        auto& entry = tensor_table[response.tensor_names()[0]];
        tensor_size = entry.tensor->size();

        std::deque<Response> skipped_responses;
        int64_t skipped_size = 0;
        while (!responses.empty()) {
          auto new_response = responses.front();
          assert(new_response.tensor_names().size() == 1);
          auto& new_entry = tensor_table[new_response.tensor_names()[0]];
          int64_t new_tensor_size = new_entry.tensor->size();

          if (response.response_type() == new_response.response_type() &&
              response.devices() == new_response.devices() &&
              entry.tensor->dtype() == new_entry.tensor->dtype() &&
              tensor_size + new_tensor_size <=
                  TensorFusionThresholdBytes(param_manager)) {
            // These tensors will fuse together well.
            tensor_size += new_tensor_size;
            response.add_tensor_name(new_response.tensor_names()[0]);
            responses.pop_front();
          } else {
            // In general, don't try to fuse additional tensors since they are
            // usually computed in order of requests and skipping tensors may
            // mean that the batch will have to wait longer while skipped
            // tensors could be reduced at that time. However, mixed-precision
            // training may yield requests of various dtype in a mixed-up
            // sequence causing breakups in fusion. To counter this some look
            // ahead is allowed.

            skipped_size += new_tensor_size;
            if (tensor_size + skipped_size <=
                TensorFusionThresholdBytes(param_manager)) {
              // Skip response and look ahead for more to fuse.
              skipped_responses.push_back(std::move(responses.front()));
              responses.pop_front();
            } else {
              break;
            }
          }
        }

        // Replace any skipped responses.
        while (!skipped_responses.empty()) {
          responses.push_front(std::move(skipped_responses.back()));
          skipped_responses.pop_back();
        }

      } else if (response.response_type() ==
                 Response::ResponseType::ALLGATHER) {
        // Attempt to add more responses to this fused response.
        auto& entry = tensor_table[response.tensor_names()[0]];

        // This is size of first dimension.
        int64_t total_byte_size_of_output =
            TotalByteSizeOfAllgatherOutput(response.tensor_sizes(), entry);

        std::deque<Response> skipped_responses;
        int64_t skipped_size = 0;
        while (!responses.empty()) {

          auto new_response = responses.front();
          assert(new_response.tensor_names().size() == 1);
          auto& new_entry = tensor_table[new_response.tensor_names()[0]];

          int64_t new_total_byte_size_of_output =
              TotalByteSizeOfAllgatherOutput(new_response.tensor_sizes(),
                                             new_entry);

          if (response.response_type() == new_response.response_type() &&
              response.devices() == new_response.devices() &&
              entry.tensor->dtype() == new_entry.tensor->dtype() &&
              total_byte_size_of_output + new_total_byte_size_of_output <=
                  TensorFusionThresholdBytes(param_manager)) {

            // These tensors will fuse together well.
            total_byte_size_of_output += new_total_byte_size_of_output;
            response.add_allgather_response(new_response);
            responses.pop_front();

          } else {
            // In general, don't try to fuse additional tensors since they are
            // usually computed in order of requests and skipping tensors may
            // mean that the batch will have to wait longer while skipped
            // tensors could be reduced at that time. However, mixed-precision
            // training may yield requests of various dtype in a mixed-up
            // sequence causing breakups in fusion. To counter this some look
            // ahead is allowed.

            skipped_size += new_total_byte_size_of_output;
            if (total_byte_size_of_output + skipped_size <=
                TensorFusionThresholdBytes(param_manager)) {
              // Skip response and look ahead for more to fuse.
              skipped_responses.push_back(std::move(responses.front()));
              responses.pop_front();
            } else {
              break;
            }
          }
        }

        // Replace any skipped responses.
        while (!skipped_responses.empty()) {
          responses.push_front(std::move(skipped_responses.back()));
          skipped_responses.pop_back();
        }
      }

      response_list.add_response(response);
      LOG(DEBUG) << "Created response of size " << tensor_size;
    }
  }

  return response_list;
}

// Return the total byte size of the final allgathered output tensor
int64_t Controller::TotalByteSizeOfAllgatherOutput(
    const std::vector<int64_t>& tensor_sizes, const TensorTableEntry& entry) {
  int64_t total_dimension_size = 0;
  for (auto sz : tensor_sizes) {
    total_dimension_size += sz;
  }
  // Every tensor participating in Allgather operation may have
  // different first dimension size, but the rest of dimensions are same
  // for all tensors.  Here we get shape of tensor sliced by first
  // dimension. Allgather output will have shape of: (sum of first
  // dimension of every tensor) x (tensor slice shape).
  int64_t total_count_of_output_entries = total_dimension_size;
  for (int i = 1; i < entry.tensor->shape().dims(); ++i) {
    total_count_of_output_entries *= entry.tensor->shape().dim_size(i);
  }
  int element_size = GetTypeSize(entry.tensor->dtype());
  int64_t total_byte_size_of_output =
      total_count_of_output_entries * element_size;

  return total_byte_size_of_output;
}

} // namespace common
} // namespace horovod
