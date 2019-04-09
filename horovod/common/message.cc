// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "message.h"
#include "wire/message_generated.h"
#include <iostream>

namespace horovod {
namespace common {

const std::string& DataType_Name(DataType value) {
  switch (value) {
    case HOROVOD_UINT8:
      static const std::string uint8("uint8");
      return uint8;
    case HOROVOD_INT8:
      static const std::string int8("int8");
      return int8;
    case HOROVOD_UINT16:
      static const std::string uint16("uint16");
      return uint16;
    case HOROVOD_INT16:
      static const std::string int16("int16");
      return int16;
    case HOROVOD_INT32:
      static const std::string int32("int32");
      return int32;
    case HOROVOD_INT64:
      static const std::string int64("int64");
      return int64;
    case HOROVOD_FLOAT16:
      static const std::string float16("float16");
      return float16;
    case HOROVOD_FLOAT32:
      static const std::string float32("float32");
      return float32;
    case HOROVOD_FLOAT64:
      static const std::string float64("float64");
      return float64;
    case HOROVOD_BOOL:
      static const std::string bool_("bool");
      return bool_;
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

const std::string& Request::RequestType_Name(RequestType value) {
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

int32_t Request::request_rank() const { return request_rank_; }

void Request::set_request_rank(int32_t value) { request_rank_ = value; }

Request::RequestType Request::request_type() const {
  return request_type_;
}

void Request::set_request_type(RequestType value) { request_type_ = value; }

DataType Request::tensor_type() const { return tensor_type_; }

void Request::set_tensor_type(DataType value) { tensor_type_ = value; }

const std::string& Request::tensor_name() const { return tensor_name_; }

void Request::set_tensor_name(const std::string& value) {
  tensor_name_ = value;
}

int32_t Request::root_rank() const { return root_rank_; }

void Request::set_root_rank(int32_t value) { root_rank_ = value; }

int32_t Request::device() const { return device_; }

void Request::set_device(int32_t value) { device_ = value; }

const std::vector<int64_t>& Request::tensor_shape() const {
  return tensor_shape_;
}

void Request::set_tensor_shape(const std::vector<int64_t>& value) {
  tensor_shape_ = value;
}

void Request::add_tensor_shape(int64_t value) {
  tensor_shape_.push_back(value);
}

namespace {

void Request_ParseFromWire(Request& request,
                           const wire::Request* obj) {
  request.set_request_rank(obj->request_rank());
  request.set_request_type((Request::RequestType) obj->request_type());
  request.set_tensor_type((DataType) obj->tensor_type());
  request.set_tensor_name(obj->tensor_name()->str());
  request.set_root_rank(obj->root_rank());
  request.set_device(obj->device());
  request.set_tensor_shape(std::vector<int64_t>(obj->tensor_shape()->begin(),
                                                obj->tensor_shape()->end()));
}

void Request_SerializeToWire(const Request& request,
                             flatbuffers::FlatBufferBuilder& builder,
                             flatbuffers::Offset<wire::Request>& obj) {
  // FlatBuffers must be built bottom-up.
  auto tensor_name_wire = builder.CreateString(request.tensor_name());
  auto tensor_shape_wire = builder.CreateVector(request.tensor_shape());

  wire::RequestBuilder request_builder(builder);
  request_builder.add_request_rank(request.request_rank());
  request_builder.add_request_type(
      (wire::RequestType) request.request_type());
  request_builder.add_tensor_type((wire::DataType) request.tensor_type());
  request_builder.add_tensor_name(tensor_name_wire);
  request_builder.add_root_rank(request.root_rank());
  request_builder.add_device(request.device());
  request_builder.add_tensor_shape(tensor_shape_wire);
  obj = request_builder.Finish();
}

} // namespace

void Request::ParseFromBytes(Request& request, const uint8_t* input) {
  auto obj = flatbuffers::GetRoot<wire::Request>(input);
  Request_ParseFromWire(request, obj);
}

void Request::SerializeToString(const Request& request,
                                std::string& output) {
  flatbuffers::FlatBufferBuilder builder(1024);
  flatbuffers::Offset<wire::Request> obj;
  Request_SerializeToWire(request, builder, obj);
  builder.Finish(obj);

  uint8_t* buf = builder.GetBufferPointer();
  auto size = builder.GetSize();
  output = std::string((char*) buf, size);
}

const std::vector<Request>& RequestList::requests() const {
  return requests_;
}

void RequestList::set_requests(const std::vector<Request>& value) {
  requests_ = value;
}

bool RequestList::shutdown() const { return shutdown_; }

void RequestList::set_shutdown(bool value) { shutdown_ = value; }

void RequestList::add_request(const Request& value) {
  requests_.push_back(value);
}

void RequestList::emplace_request(Request&& value) {
  requests_.emplace_back(value);
}

void RequestList::ParseFromBytes(RequestList& request_list,
                                 const uint8_t* input) {
  auto obj = flatbuffers::GetRoot<wire::RequestList>(input);
  for (const auto& req_obj : *obj->requests()) {
    Request request;
    Request_ParseFromWire(request, req_obj);
    request_list.emplace_request(std::move(request));
  }
  request_list.set_shutdown(obj->shutdown());
}

void RequestList::SerializeToString(const RequestList& request_list,
                                    std::string& output) {
  // FlatBuffers must be built bottom-up.
  flatbuffers::FlatBufferBuilder builder(1024);
  std::vector<flatbuffers::Offset<wire::Request>> requests;
  requests.reserve(request_list.requests().size());
  for (const auto& req : request_list.requests()) {
    flatbuffers::Offset<wire::Request> req_obj;
    Request_SerializeToWire(req, builder, req_obj);
    requests.push_back(req_obj);
  }
  auto requests_wire = builder.CreateVector(requests);

  wire::RequestListBuilder request_list_builder(builder);
  request_list_builder.add_requests(requests_wire);
  request_list_builder.add_shutdown(request_list.shutdown());
  auto obj = request_list_builder.Finish();
  builder.Finish(obj);

  uint8_t* buf = builder.GetBufferPointer();
  auto size = builder.GetSize();
  output = std::string((char*) buf, size);
}

const std::string& Response::ResponseType_Name(ResponseType value) {
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
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

Response::ResponseType Response::response_type() const {
  return response_type_;
}

void Response::set_response_type(ResponseType value) {
  response_type_ = value;
}

const std::vector<std::string>& Response::tensor_names() const {
  return tensor_names_;
}

const std::string Response::tensor_names_string() const {
  std::string result;
  bool is_first_name = true;
  for (auto const& s : tensor_names_) {
    if (!is_first_name) {
      result += ", ";
    } else {
      is_first_name = false;
    }
    result += s;
  }
  return result;
}

void Response::set_tensor_names(const std::vector<std::string>& value) {
  tensor_names_ = value;
}

void Response::add_tensor_name(const std::string& value) {
  tensor_names_.push_back(value);
}

const std::string& Response::error_message() const { return error_message_; }

void Response::set_error_message(const std::string& value) {
  error_message_ = value;
}

const std::vector<int32_t>& Response::devices() const { return devices_; }

void Response::set_devices(const std::vector<int32_t>& value) {
  devices_ = value;
}

void Response::add_device(int32_t value) { devices_.push_back(value); }

const std::vector<int64_t>& Response::tensor_sizes() const {
  return tensor_sizes_;
}

void Response::set_tensor_sizes(const std::vector<int64_t>& value) {
  tensor_sizes_ = value;
}

void Response::add_tensor_size(int64_t value) {
  tensor_sizes_.push_back(value);
}

void Response::add_allgather_response(const Response& response) {
  assert(response_type() == Response::ResponseType::ALLGATHER);
  assert(response.tensor_names().size() == 1);
  assert(response.devices() == devices());
  add_tensor_name(response.tensor_names()[0]);
  for (auto size : response.tensor_sizes()) {
    add_tensor_size(size);
  }
}

void Response_ParseFromWire(Response& response,
                            const wire::Response* obj) {
  response.set_response_type((Response::ResponseType) obj->response_type());
  for (const auto& tensor_name_obj : *obj->tensor_names()) {
    response.add_tensor_name(tensor_name_obj->str());
  }
  response.set_error_message(obj->error_message()->str());
  response.set_devices(
      std::vector<int32_t>(obj->devices()->begin(), obj->devices()->end()));
  response.set_tensor_sizes(std::vector<int64_t>(obj->tensor_sizes()->begin(),
                                                 obj->tensor_sizes()->end()));
}

void Response::ParseFromBytes(Response& response, const uint8_t* input) {
  auto obj = flatbuffers::GetRoot<wire::Response>(input);
  Response_ParseFromWire(response, obj);
}

void Response_SerializeToWire(const Response& response,
                              flatbuffers::FlatBufferBuilder& builder,
                              flatbuffers::Offset<wire::Response>& obj) {
  // FlatBuffers must be built bottom-up.
  auto tensor_names_wire =
      builder.CreateVectorOfStrings(response.tensor_names());
  auto error_message_wire = builder.CreateString(response.error_message());
  auto devices_wire = builder.CreateVector(response.devices());
  auto tensor_sizes_wire = builder.CreateVector(response.tensor_sizes());

  wire::ResponseBuilder response_builder(builder);
  response_builder.add_response_type(
      (wire::ResponseType) response.response_type());
  response_builder.add_tensor_names(tensor_names_wire);
  response_builder.add_error_message(error_message_wire);
  response_builder.add_devices(devices_wire);
  response_builder.add_tensor_sizes(tensor_sizes_wire);
  obj = response_builder.Finish();
}

void Response::SerializeToString(const Response& response,
                                 std::string& output) {
  flatbuffers::FlatBufferBuilder builder(1024);
  flatbuffers::Offset<wire::Response> obj;
  Response_SerializeToWire(response, builder, obj);
  builder.Finish(obj);

  uint8_t* buf = builder.GetBufferPointer();
  auto size = builder.GetSize();
  output = std::string((char*) buf, size);
}

const std::vector<Response>& ResponseList::responses() const {
  return responses_;
}

void ResponseList::set_responses(const std::vector<Response>& value) {
  responses_ = value;
}

bool ResponseList::shutdown() const { return shutdown_; }

void ResponseList::set_shutdown(bool value) { shutdown_ = value; }

void ResponseList::add_response(const Response& value) {
  responses_.push_back(value);
}

void ResponseList::emplace_response(Response&& value) {
  responses_.emplace_back(value);
}

void ResponseList::ParseFromBytes(ResponseList& response_list,
                                  const uint8_t* input) {
  auto obj = flatbuffers::GetRoot<wire::ResponseList>(input);
  for (const auto& resp_obj : *obj->responses()) {
    Response response;
    Response_ParseFromWire(response, resp_obj);
    response_list.emplace_response(std::move(response));
  }
  response_list.set_shutdown(obj->shutdown());
}

void ResponseList::SerializeToString(const ResponseList& response_list,
                                     std::string& output) {
  // FlatBuffers must be built bottom-up.
  flatbuffers::FlatBufferBuilder builder(1024);
  std::vector<flatbuffers::Offset<wire::Response>> responses;
  responses.reserve(response_list.responses().size());
  for (const auto& resp : response_list.responses()) {
    flatbuffers::Offset<wire::Response> resp_obj;
    Response_SerializeToWire(resp, builder, resp_obj);
    responses.push_back(resp_obj);
  }
  auto responses_wire = builder.CreateVector(responses);

  wire::ResponseListBuilder response_list_builder(builder);
  response_list_builder.add_responses(responses_wire);
  response_list_builder.add_shutdown(response_list.shutdown());
  auto obj = response_list_builder.Finish();
  builder.Finish(obj);

  uint8_t* buf = builder.GetBufferPointer();
  auto size = builder.GetSize();
  output = std::string((char*) buf, size);
}

void ResponseCache::set_capacity(uint32_t capacity) {
  capacity_ = capacity;
  iters_.reserve(capacity + RESERVED_CACHE_BITS);
}

uint32_t ResponseCache::capacity() const {return capacity_;}

size_t ResponseCache::current_size() const {return cache_.size();}

bool ResponseCache::cached(const Request& message) const {
  return table_.find(message.tensor_name()) != table_.end();
}

bool ResponseCache::cached(const Response& response) const {
  assert(response.tensor_names().size() == 1);
  return table_.find(response.tensor_names()[0]) != table_.end();
}

void ResponseCache::put_(const Response& response) {
  uint32_t cache_bit;
  if (this->cached(response)) {
    cache_bit = table_[response.tensor_names()[0]];
    auto it = iters_[cache_bit];
    // Do not allow caching of response with same name as existing
    // response but with different params.
    assert(it->response_type() == response.response_type() &&
           it->devices() == response.devices() &&
           it->tensor_sizes() == response.tensor_sizes());
    cache_.push_front(std::move(*it));
    cache_.erase(it);
  } else if (cache_.size() == capacity_) {
    auto& entry = cache_.back();
    cache_bit = table_[entry.tensor_names()[0]];
    table_.erase(entry.tensor_names()[0]);
    cache_.pop_back();
    cache_.push_front(response);
  } else {
    cache_bit = counter_++;
    iters_.resize(counter_);
    cache_.push_front(response);
  }

  iters_[cache_bit] = cache_.begin();
  table_[response.tensor_names()[0]] = cache_bit;

  bits_outdated_ = true;
}

void ResponseCache::put(const Response& response) {
  if (capacity_ == 0) return;

  // If response is fused, split back into individual responses
  if (response.tensor_names().size() > 0) {
    for (auto& name : response.tensor_names()) {
      Response new_response;
      new_response.add_tensor_name(name);
      new_response.set_response_type(response.response_type());
      new_response.set_devices(response.devices());
      new_response.set_tensor_sizes(response.tensor_sizes());
      this->put_(new_response);
    }
  } else {
    this->put_(response);
  }
}

const Response& ResponseCache::get_response(const Request& message) {
  assert(this->cached(message));
  uint32_t cache_bit = table_[message.tensor_name()];
  auto it = iters_[cache_bit];
  cache_.push_front(std::move(*it));
  cache_.erase(it);
  iters_[cache_bit] = cache_.begin();
  bits_outdated_ = true;
  return cache_.front();
}

const Response& ResponseCache::get_response(uint32_t cache_bit) {
  assert(cache_bit < iters_.size());
  auto it = iters_[cache_bit];
  cache_.push_front(std::move(*it));
  cache_.erase(it);
  iters_[cache_bit] = cache_.begin();
  bits_outdated_ = true;
  return cache_.front();
}

const Response& ResponseCache::peek_response(const Request& message) const {
  assert(this->cached(message));
  uint32_t cache_bit = table_.at(message.tensor_name());
  return *iters_[cache_bit];
}

const Response& ResponseCache::peek_response(uint32_t cache_bit) const {
  assert(cache_bit < iters_.size());
  return *iters_[cache_bit];
}

uint32_t ResponseCache::peek_cache_bit(const Request& message) const {
  assert(this->cached(message));
  return table_.at(message.tensor_name());
}

void ResponseCache::update_cache_bits() {
  if (!bits_outdated_) return;

  // Iterate over current cache state and reassign cache bits. Least recently
  // used get lower cache positions.
  auto it = --cache_.end();
  for (int i = 0; i < (int)cache_.size(); ++i) {
    iters_[i + RESERVED_CACHE_BITS] = it;
    table_[it->tensor_names()[0]] =  i + RESERVED_CACHE_BITS;
    --it;
  }

  bits_outdated_ = false;
}

} // namespace common
} // namespace horovod
