// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2017 Uber Technologies, Inc.
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

#include "mpi_message.h"
#include "wire/mpi_message_generated.h"
#include <iostream>

namespace horovod {
namespace tensorflow {

const std::string& MPIDataType_Name(MPIDataType value) {
  switch (value) {
  case TF_MPI_UINT8:
    static const std::string uint8("uint8");
    return uint8;
  case TF_MPI_INT8:
    static const std::string int8("int8");
    return int8;
  case TF_MPI_UINT16:
    static const std::string uint16("uint16");
    return uint16;
  case TF_MPI_INT16:
    static const std::string int16("int16");
    return int16;
  case TF_MPI_INT32:
    static const std::string int32("int32");
    return int32;
  case TF_MPI_INT64:
    static const std::string int64("int64");
    return int64;
  case TF_MPI_FLOAT32:
    static const std::string float32("float32");
    return float32;
  case TF_MPI_FLOAT64:
    static const std::string float64("float64");
    return float64;
  case TF_MPI_BOOL:
    static const std::string bool_("bool");
    return bool_;
  default:
    static const std::string unknown("<unknown>");
    return unknown;
  }
}

const std::string& MPIRequest::RequestType_Name(RequestType value) {
  switch (value) {
  case RequestType::ALLREDUCE:
    static const std::string allreduce("ALLREDUCE");
    return allreduce;
  case RequestType::ALLGATHER:
    static const std::string allgather("ALLGATHER");
    return allgather;
  case RequestType::BROADCAST:
    static const std::string broadcast("BROADCAST");
    return broadcast;
  default:
    static const std::string unknown("<unknown>");
    return unknown;
  }
}

int32_t MPIRequest::request_rank() const { return request_rank_; }

void MPIRequest::set_request_rank(int32_t value) { request_rank_ = value; }

MPIRequest::RequestType MPIRequest::request_type() const {
  return request_type_;
}

void MPIRequest::set_request_type(RequestType value) { request_type_ = value; }

MPIDataType MPIRequest::tensor_type() const { return tensor_type_; }

void MPIRequest::set_tensor_type(MPIDataType value) { tensor_type_ = value; }

const std::string& MPIRequest::tensor_name() const { return tensor_name_; }

void MPIRequest::set_tensor_name(const std::string& value) {
  tensor_name_ = value;
}

int32_t MPIRequest::root_rank() const { return root_rank_; }

void MPIRequest::set_root_rank(int32_t value) { root_rank_ = value; }

int32_t MPIRequest::device() const { return device_; }

void MPIRequest::set_device(int32_t value) { device_ = value; }

const std::vector<int64_t>& MPIRequest::tensor_shape() const {
  return tensor_shape_;
}

void MPIRequest::set_tensor_shape(const std::vector<int64_t>& value) {
  tensor_shape_ = value;
}

void MPIRequest::add_tensor_shape(int64_t value) {
  tensor_shape_.push_back(value);
}

void MPIRequest::ParseFromString(MPIRequest& request,
                                 const std::string& input) {
  auto obj = flatbuffers::GetRoot<wire::MPIRequest>((uint8_t*)input.c_str());
  request.set_request_rank(obj->request_rank());
  request.set_request_type((MPIRequest::RequestType)obj->request_type());
  request.set_tensor_type((MPIDataType)obj->tensor_type());
  request.set_tensor_name(obj->tensor_name()->str());
  request.set_root_rank(obj->root_rank());
  request.set_device(obj->device());
  request.set_tensor_shape(std::vector<int64_t>(obj->tensor_shape()->begin(),
                                                obj->tensor_shape()->end()));
}

void MPIRequest::SerializeToString(MPIRequest& request, std::string& output) {
  flatbuffers::FlatBufferBuilder builder(1024);
  wire::MPIRequestBuilder request_builder(builder);
  request_builder.add_request_rank(request.request_rank());
  request_builder.add_request_type(
      (wire::MPIRequestType)request.request_type());
  request_builder.add_tensor_type((wire::MPIDataType)request.tensor_type());
  request_builder.add_tensor_name(builder.CreateString(request.tensor_name()));
  request_builder.add_root_rank(request.root_rank());
  request_builder.add_device(request.device());
  request_builder.add_tensor_shape(
      builder.CreateVector(request.tensor_shape()));
  auto obj = request_builder.Finish();
  builder.Finish(obj);

  uint8_t* buf = builder.GetBufferPointer();
  auto size = builder.GetSize();
  output = std::string((char*)buf, size);
}

const std::string& MPIResponse::ResponseType_Name(ResponseType value) {
  switch (value) {
  case ResponseType::ALLREDUCE:
    static const std::string allreduce("ALLREDUCE");
    return allreduce;
  case ResponseType::ALLGATHER:
    static const std::string allgather("ALLGATHER");
    return allgather;
  case ResponseType::BROADCAST:
    static const std::string broadcast("BROADCAST");
    return broadcast;
  case ResponseType::ERROR:
    static const std::string error("ERROR");
    return error;
  case ResponseType::DONE:
    static const std::string done("DONE");
    return done;
  case ResponseType::SHUTDOWN:
    static const std::string shutdown("SHUTDOWN");
    return shutdown;
  default:
    static const std::string unknown("<unknown>");
    return unknown;
  }
}

MPIResponse::ResponseType MPIResponse::response_type() const {
  return response_type_;
}

void MPIResponse::set_response_type(ResponseType value) {
  response_type_ = value;
}

const std::vector<std::string>& MPIResponse::tensor_names() const {
  return tensor_names_;
}

void MPIResponse::set_tensor_names(const std::vector<std::string>& value) {
  tensor_names_ = value;
}

void MPIResponse::add_tensor_names(const std::string& value) {
  tensor_names_.push_back(value);
}

const std::string& MPIResponse::error_message() const { return error_message_; }

void MPIResponse::set_error_message(const std::string& value) {
  error_message_ = value;
}

const std::vector<int32_t>& MPIResponse::devices() const { return devices_; }

void MPIResponse::set_devices(const std::vector<int32_t>& value) {
  devices_ = value;
}

void MPIResponse::add_devices(int32_t value) { devices_.push_back(value); }

const std::vector<int64_t>& MPIResponse::tensor_sizes() const {
  return tensor_sizes_;
}

void MPIResponse::set_tensor_sizes(const std::vector<int64_t>& value) {
  tensor_sizes_ = value;
}

void MPIResponse::add_tensor_sizes(int64_t value) {
  tensor_sizes_.push_back(value);
}

void MPIResponse::ParseFromString(MPIResponse& response,
                                  const std::string& input) {
  auto obj = flatbuffers::GetRoot<wire::MPIResponse>((uint8_t*)input.c_str());
  response.set_response_type((MPIResponse::ResponseType)obj->response_type());
  for (auto it = obj->tensor_names()->begin(); it != obj->tensor_names()->end();
       it++) {
    response.add_tensor_names(it->str());
  }
  response.set_error_message(obj->error_message()->str());
  response.set_devices(
      std::vector<int32_t>(obj->devices()->begin(), obj->devices()->end()));
  response.set_tensor_sizes(std::vector<int64_t>(obj->tensor_sizes()->begin(),
                                                 obj->tensor_sizes()->end()));
}

void MPIResponse::SerializeToString(MPIResponse& response,
                                    std::string& output) {
  flatbuffers::FlatBufferBuilder builder(1024);
  wire::MPIResponseBuilder response_builder(builder);
  response_builder.add_response_type(
      (wire::MPIResponseType)response.response_type());
  response_builder.add_tensor_names(
      builder.CreateVectorOfStrings(response.tensor_names()));
  response_builder.add_error_message(
      builder.CreateString(response.error_message()));
  response_builder.add_devices(builder.CreateVector(response.devices()));
  response_builder.add_tensor_sizes(
      builder.CreateVector(response.tensor_sizes()));
  auto obj = response_builder.Finish();
  builder.Finish(obj);

  uint8_t* buf = builder.GetBufferPointer();
  auto size = builder.GetSize();
  output = std::string((char*)buf, size);
}

} // namespace tensorflow
} // namespace horovod
