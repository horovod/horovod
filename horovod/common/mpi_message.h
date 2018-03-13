// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
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

#ifndef HOROVOD_MPI_MESSAGE_H
#define HOROVOD_MPI_MESSAGE_H

#include <string>
#include <vector>

namespace horovod {
namespace common {

enum MPIDataType {
  HOROVOD_UINT8 = 0,
  HOROVOD_INT8 = 1,
  HOROVOD_UINT16 = 2,
  HOROVOD_INT16 = 3,
  HOROVOD_INT32 = 4,
  HOROVOD_INT64 = 5,
  HOROVOD_FLOAT32 = 6,
  HOROVOD_FLOAT64 = 7,
  HOROVOD_BOOL = 8
};

const std::string& MPIDataType_Name(MPIDataType value);

// An MPIRequest is a message sent from a rank greater than zero to the
// coordinator (rank zero), informing the coordinator of an operation that
// the rank wants to do and the tensor that it wants to apply the operation to.
class MPIRequest {
public:
  enum RequestType { ALLREDUCE = 0, ALLGATHER = 1, BROADCAST = 2 };

  static const std::string& RequestType_Name(RequestType value);

  // The request rank is necessary to create a consistent ordering of results,
  // for example in the allgather where the order of outputs should be sorted
  // by rank.
  int32_t request_rank() const;
  void set_request_rank(int32_t value);

  RequestType request_type() const;
  void set_request_type(RequestType value);

  MPIDataType tensor_type() const;
  void set_tensor_type(MPIDataType value);

  const std::string& tensor_name() const;
  void set_tensor_name(const std::string& value);

  int32_t root_rank() const;
  void set_root_rank(int32_t value);

  int32_t device() const;
  void set_device(int32_t value);

  const std::vector<int64_t>& tensor_shape() const;
  void set_tensor_shape(const std::vector<int64_t>& value);
  void add_tensor_shape(int64_t value);

  static void ParseFromString(MPIRequest& request, const std::string& input);
  static void SerializeToString(MPIRequest& request, std::string& output);

private:
  int32_t request_rank_;
  RequestType request_type_;
  MPIDataType tensor_type_;
  int32_t root_rank_;
  int32_t device_;
  std::string tensor_name_;
  std::vector<int64_t> tensor_shape_;
};

class MPIRequestList {
public:
  const std::vector<MPIRequest>& requests() const;
  void set_requests(const std::vector<MPIRequest>& value);
  void add_requests(MPIRequest value);
  bool shutdown() const;
  void set_shutdown(bool value);

  static void ParseFromString(MPIRequestList& request_list,
                              const std::string& input);
  static void SerializeToString(MPIRequestList& request_list,
                                std::string& output);

private:
  std::vector<MPIRequest> requests_;
  bool shutdown_ = false;
};

// An MPIResponse is a message sent from the coordinator (rank zero) to a rank
// greater than zero, informing the rank of an operation should be performed
// now. If the operation requested would result in an error (for example, due
// to a type or shape mismatch), then the MPIResponse can contain an error and
// an error message instead. Finally, an MPIResponse can be a DONE message (if
// there are no more tensors to reduce on this tick of the background loop) or
// SHUTDOWN if all MPI processes should shut down.
class MPIResponse {
public:
  enum ResponseType {
    ALLREDUCE = 0,
    ALLGATHER = 1,
    BROADCAST = 2,
    ERROR = 3,
    DONE = 4,
    SHUTDOWN = 5
  };

  static const std::string& ResponseType_Name(ResponseType value);

  ResponseType response_type() const;
  void set_response_type(ResponseType value);

  // Empty if the type is DONE or SHUTDOWN.
  const std::vector<std::string>& tensor_names() const;
  void set_tensor_names(const std::vector<std::string>& value);
  void add_tensor_names(const std::string& value);

  // Empty unless response_type is ERROR.
  const std::string& error_message() const;
  void set_error_message(const std::string& value);

  const std::vector<int32_t>& devices() const;
  void set_devices(const std::vector<int32_t>& value);
  void add_devices(int32_t value);

  // Empty unless response_type is ALLGATHER.
  // These tensor sizes are the dimension zero sizes of all the input matrices,
  // indexed by the rank.
  const std::vector<int64_t>& tensor_sizes() const;
  void set_tensor_sizes(const std::vector<int64_t>& value);
  void add_tensor_sizes(int64_t value);

  static void ParseFromString(MPIResponse& response, const std::string& input);
  static void SerializeToString(MPIResponse& response, std::string& output);

private:
  ResponseType response_type_;
  std::vector<std::string> tensor_names_;
  std::string error_message_;
  std::vector<int32_t> devices_;
  std::vector<int64_t> tensor_sizes_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MPI_MESSAGE_H
