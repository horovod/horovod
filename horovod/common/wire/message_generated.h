// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright Microsoft
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

// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_MESSAGE_HOROVOD_COMMON_WIRE_H_
#define FLATBUFFERS_GENERATED_MESSAGE_HOROVOD_COMMON_WIRE_H_

#include "flatbuffers/flatbuffers.h"

namespace horovod {
namespace common {
namespace wire {

struct Request;
struct RequestBuilder;

struct RequestList;
struct RequestListBuilder;

struct Response;
struct ResponseBuilder;

struct ResponseList;
struct ResponseListBuilder;

enum DataType : int8_t {
  DataType_HOROVOD_UINT8 = 0,
  DataType_HOROVOD_INT8 = 1,
  DataType_HOROVOD_UINT16 = 2,
  DataType_HOROVOD_INT16 = 3,
  DataType_HOROVOD_INT32 = 4,
  DataType_HOROVOD_INT64 = 5,
  DataType_HOROVOD_FLOAT16 = 6,
  DataType_HOROVOD_FLOAT32 = 7,
  DataType_HOROVOD_FLOAT64 = 8,
  DataType_HOROVOD_BOOL = 9,
  DataType_MIN = DataType_HOROVOD_UINT8,
  DataType_MAX = DataType_HOROVOD_BOOL
};

inline const DataType (&EnumValuesDataType())[10] {
  static const DataType values[] = {
    DataType_HOROVOD_UINT8,
    DataType_HOROVOD_INT8,
    DataType_HOROVOD_UINT16,
    DataType_HOROVOD_INT16,
    DataType_HOROVOD_INT32,
    DataType_HOROVOD_INT64,
    DataType_HOROVOD_FLOAT16,
    DataType_HOROVOD_FLOAT32,
    DataType_HOROVOD_FLOAT64,
    DataType_HOROVOD_BOOL
  };
  return values;
}

inline const char * const *EnumNamesDataType() {
  static const char * const names[11] = {
    "HOROVOD_UINT8",
    "HOROVOD_INT8",
    "HOROVOD_UINT16",
    "HOROVOD_INT16",
    "HOROVOD_INT32",
    "HOROVOD_INT64",
    "HOROVOD_FLOAT16",
    "HOROVOD_FLOAT32",
    "HOROVOD_FLOAT64",
    "HOROVOD_BOOL",
    nullptr
  };
  return names;
}

inline const char *EnumNameDataType(DataType e) {
  if (flatbuffers::IsOutRange(e, DataType_HOROVOD_UINT8, DataType_HOROVOD_BOOL)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesDataType()[index];
}

enum RequestType : int8_t {
  RequestType_ALLREDUCE = 0,
  RequestType_ALLGATHER = 1,
  RequestType_BROADCAST = 2,
  RequestType_JOIN = 3,
  RequestType_ADASUM = 4,
  RequestType_ALLTOALL = 5,
  RequestType_BARRIER = 6,
  RequestType_REDUCESCATTER = 7,
  RequestType_MIN = RequestType_ALLREDUCE,
  RequestType_MAX = RequestType_REDUCESCATTER
};

inline const RequestType (&EnumValuesRequestType())[8] {
  static const RequestType values[] = {
    RequestType_ALLREDUCE,
    RequestType_ALLGATHER,
    RequestType_BROADCAST,
    RequestType_JOIN,
    RequestType_ADASUM,
    RequestType_ALLTOALL,
    RequestType_BARRIER,
    RequestType_REDUCESCATTER
  };
  return values;
}

inline const char * const *EnumNamesRequestType() {
  static const char * const names[9] = {
    "ALLREDUCE",
    "ALLGATHER",
    "BROADCAST",
    "JOIN",
    "ADASUM",
    "ALLTOALL",
    "BARRIER",
    "REDUCESCATTER",
    nullptr
  };
  return names;
}

inline const char *EnumNameRequestType(RequestType e) {
  if (flatbuffers::IsOutRange(e, RequestType_ALLREDUCE, RequestType_REDUCESCATTER)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesRequestType()[index];
}

enum ResponseType : int8_t {
  ResponseType_ALLREDUCE = 0,
  ResponseType_ALLGATHER = 1,
  ResponseType_BROADCAST = 2,
  ResponseType_JOIN = 3,
  ResponseType_ADASUM = 4,
  ResponseType_ALLTOALL = 5,
  ResponseType_BARRIER = 6,
  ResponseType_REDUCESCATTER = 7,
  ResponseType_ERROR = 8,
  ResponseType_MIN = ResponseType_ALLREDUCE,
  ResponseType_MAX = ResponseType_ERROR
};

inline const ResponseType (&EnumValuesResponseType())[9] {
  static const ResponseType values[] = {
    ResponseType_ALLREDUCE,
    ResponseType_ALLGATHER,
    ResponseType_BROADCAST,
    ResponseType_JOIN,
    ResponseType_ADASUM,
    ResponseType_ALLTOALL,
    ResponseType_BARRIER,
    ResponseType_REDUCESCATTER,
    ResponseType_ERROR
  };
  return values;
}

inline const char * const *EnumNamesResponseType() {
  static const char * const names[10] = {
    "ALLREDUCE",
    "ALLGATHER",
    "BROADCAST",
    "JOIN",
    "ADASUM",
    "ALLTOALL",
    "BARRIER",
    "REDUCESCATTER",
    "ERROR",
    nullptr
  };
  return names;
}

inline const char *EnumNameResponseType(ResponseType e) {
  if (flatbuffers::IsOutRange(e, ResponseType_ALLREDUCE, ResponseType_ERROR)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesResponseType()[index];
}

struct Request FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef RequestBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_REQUEST_RANK = 4,
    VT_REQUEST_TYPE = 6,
    VT_TENSOR_TYPE = 8,
    VT_TENSOR_NAME = 10,
    VT_ROOT_RANK = 12,
    VT_DEVICE = 14,
    VT_TENSOR_SHAPE = 16,
    VT_PRESCALE_FACTOR = 18,
    VT_POSTSCALE_FACTOR = 20
  };
  int32_t request_rank() const {
    return GetField<int32_t>(VT_REQUEST_RANK, 0);
  }
  horovod::common::wire::RequestType request_type() const {
    return static_cast<horovod::common::wire::RequestType>(GetField<int8_t>(VT_REQUEST_TYPE, 0));
  }
  horovod::common::wire::DataType tensor_type() const {
    return static_cast<horovod::common::wire::DataType>(GetField<int8_t>(VT_TENSOR_TYPE, 0));
  }
  const flatbuffers::String *tensor_name() const {
    return GetPointer<const flatbuffers::String *>(VT_TENSOR_NAME);
  }
  int32_t root_rank() const {
    return GetField<int32_t>(VT_ROOT_RANK, 0);
  }
  int32_t device() const {
    return GetField<int32_t>(VT_DEVICE, 0);
  }
  const flatbuffers::Vector<int64_t> *tensor_shape() const {
    return GetPointer<const flatbuffers::Vector<int64_t> *>(VT_TENSOR_SHAPE);
  }
  double prescale_factor() const {
    return GetField<double>(VT_PRESCALE_FACTOR, 0.0);
  }
  double postscale_factor() const {
    return GetField<double>(VT_POSTSCALE_FACTOR, 0.0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_REQUEST_RANK) &&
           VerifyField<int8_t>(verifier, VT_REQUEST_TYPE) &&
           VerifyField<int8_t>(verifier, VT_TENSOR_TYPE) &&
           VerifyOffset(verifier, VT_TENSOR_NAME) &&
           verifier.VerifyString(tensor_name()) &&
           VerifyField<int32_t>(verifier, VT_ROOT_RANK) &&
           VerifyField<int32_t>(verifier, VT_DEVICE) &&
           VerifyOffset(verifier, VT_TENSOR_SHAPE) &&
           verifier.VerifyVector(tensor_shape()) &&
           VerifyField<double>(verifier, VT_PRESCALE_FACTOR) &&
           VerifyField<double>(verifier, VT_POSTSCALE_FACTOR) &&
           verifier.EndTable();
  }
};

struct RequestBuilder {
  typedef Request Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_request_rank(int32_t request_rank) {
    fbb_.AddElement<int32_t>(Request::VT_REQUEST_RANK, request_rank, 0);
  }
  void add_request_type(horovod::common::wire::RequestType request_type) {
    fbb_.AddElement<int8_t>(Request::VT_REQUEST_TYPE, static_cast<int8_t>(request_type), 0);
  }
  void add_tensor_type(horovod::common::wire::DataType tensor_type) {
    fbb_.AddElement<int8_t>(Request::VT_TENSOR_TYPE, static_cast<int8_t>(tensor_type), 0);
  }
  void add_tensor_name(flatbuffers::Offset<flatbuffers::String> tensor_name) {
    fbb_.AddOffset(Request::VT_TENSOR_NAME, tensor_name);
  }
  void add_root_rank(int32_t root_rank) {
    fbb_.AddElement<int32_t>(Request::VT_ROOT_RANK, root_rank, 0);
  }
  void add_device(int32_t device) {
    fbb_.AddElement<int32_t>(Request::VT_DEVICE, device, 0);
  }
  void add_tensor_shape(flatbuffers::Offset<flatbuffers::Vector<int64_t>> tensor_shape) {
    fbb_.AddOffset(Request::VT_TENSOR_SHAPE, tensor_shape);
  }
  void add_prescale_factor(double prescale_factor) {
    fbb_.AddElement<double>(Request::VT_PRESCALE_FACTOR, prescale_factor, 0.0);
  }
  void add_postscale_factor(double postscale_factor) {
    fbb_.AddElement<double>(Request::VT_POSTSCALE_FACTOR, postscale_factor, 0.0);
  }
  explicit RequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<Request> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Request>(end);
    return o;
  }
};

inline flatbuffers::Offset<Request> CreateRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t request_rank = 0,
    horovod::common::wire::RequestType request_type = horovod::common::wire::RequestType_ALLREDUCE,
    horovod::common::wire::DataType tensor_type = horovod::common::wire::DataType_HOROVOD_UINT8,
    flatbuffers::Offset<flatbuffers::String> tensor_name = 0,
    int32_t root_rank = 0,
    int32_t device = 0,
    flatbuffers::Offset<flatbuffers::Vector<int64_t>> tensor_shape = 0,
    double prescale_factor = 0.0,
    double postscale_factor = 0.0) {
  RequestBuilder builder_(_fbb);
  builder_.add_postscale_factor(postscale_factor);
  builder_.add_prescale_factor(prescale_factor);
  builder_.add_tensor_shape(tensor_shape);
  builder_.add_device(device);
  builder_.add_root_rank(root_rank);
  builder_.add_tensor_name(tensor_name);
  builder_.add_request_rank(request_rank);
  builder_.add_tensor_type(tensor_type);
  builder_.add_request_type(request_type);
  return builder_.Finish();
}

inline flatbuffers::Offset<Request> CreateRequestDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t request_rank = 0,
    horovod::common::wire::RequestType request_type = horovod::common::wire::RequestType_ALLREDUCE,
    horovod::common::wire::DataType tensor_type = horovod::common::wire::DataType_HOROVOD_UINT8,
    const char *tensor_name = nullptr,
    int32_t root_rank = 0,
    int32_t device = 0,
    const std::vector<int64_t> *tensor_shape = nullptr,
    double prescale_factor = 0.0,
    double postscale_factor = 0.0) {
  auto tensor_name__ = tensor_name ? _fbb.CreateString(tensor_name) : 0;
  auto tensor_shape__ = tensor_shape ? _fbb.CreateVector<int64_t>(*tensor_shape) : 0;
  return horovod::common::wire::CreateRequest(
      _fbb,
      request_rank,
      request_type,
      tensor_type,
      tensor_name__,
      root_rank,
      device,
      tensor_shape__,
      prescale_factor,
      postscale_factor);
}

struct RequestList FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef RequestListBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_REQUESTS = 4,
    VT_SHUTDOWN = 6
  };
  const flatbuffers::Vector<flatbuffers::Offset<horovod::common::wire::Request>> *requests() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<horovod::common::wire::Request>> *>(VT_REQUESTS);
  }
  bool shutdown() const {
    return GetField<uint8_t>(VT_SHUTDOWN, 0) != 0;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_REQUESTS) &&
           verifier.VerifyVector(requests()) &&
           verifier.VerifyVectorOfTables(requests()) &&
           VerifyField<uint8_t>(verifier, VT_SHUTDOWN) &&
           verifier.EndTable();
  }
};

struct RequestListBuilder {
  typedef RequestList Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_requests(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<horovod::common::wire::Request>>> requests) {
    fbb_.AddOffset(RequestList::VT_REQUESTS, requests);
  }
  void add_shutdown(bool shutdown) {
    fbb_.AddElement<uint8_t>(RequestList::VT_SHUTDOWN, static_cast<uint8_t>(shutdown), 0);
  }
  explicit RequestListBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<RequestList> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<RequestList>(end);
    return o;
  }
};

inline flatbuffers::Offset<RequestList> CreateRequestList(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<horovod::common::wire::Request>>> requests = 0,
    bool shutdown = false) {
  RequestListBuilder builder_(_fbb);
  builder_.add_requests(requests);
  builder_.add_shutdown(shutdown);
  return builder_.Finish();
}

inline flatbuffers::Offset<RequestList> CreateRequestListDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<horovod::common::wire::Request>> *requests = nullptr,
    bool shutdown = false) {
  auto requests__ = requests ? _fbb.CreateVector<flatbuffers::Offset<horovod::common::wire::Request>>(*requests) : 0;
  return horovod::common::wire::CreateRequestList(
      _fbb,
      requests__,
      shutdown);
}

struct Response FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ResponseBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_RESPONSE_TYPE = 4,
    VT_TENSOR_NAMES = 6,
    VT_ERROR_MESSAGE = 8,
    VT_DEVICES = 10,
    VT_TENSOR_SIZES = 12,
    VT_TENSOR_TYPE = 14,
    VT_PRESCALE_FACTOR = 16,
    VT_POSTSCALE_FACTOR = 18,
    VT_LAST_JOINED_RANK = 20
  };
  horovod::common::wire::ResponseType response_type() const {
    return static_cast<horovod::common::wire::ResponseType>(GetField<int8_t>(VT_RESPONSE_TYPE, 0));
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *tensor_names() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_TENSOR_NAMES);
  }
  const flatbuffers::String *error_message() const {
    return GetPointer<const flatbuffers::String *>(VT_ERROR_MESSAGE);
  }
  const flatbuffers::Vector<int32_t> *devices() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_DEVICES);
  }
  const flatbuffers::Vector<int64_t> *tensor_sizes() const {
    return GetPointer<const flatbuffers::Vector<int64_t> *>(VT_TENSOR_SIZES);
  }
  horovod::common::wire::DataType tensor_type() const {
    return static_cast<horovod::common::wire::DataType>(GetField<int8_t>(VT_TENSOR_TYPE, 0));
  }
  double prescale_factor() const {
    return GetField<double>(VT_PRESCALE_FACTOR, 0.0);
  }
  double postscale_factor() const {
    return GetField<double>(VT_POSTSCALE_FACTOR, 0.0);
  }
  int32_t last_joined_rank() const {
    return GetField<int32_t>(VT_LAST_JOINED_RANK, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int8_t>(verifier, VT_RESPONSE_TYPE) &&
           VerifyOffset(verifier, VT_TENSOR_NAMES) &&
           verifier.VerifyVector(tensor_names()) &&
           verifier.VerifyVectorOfStrings(tensor_names()) &&
           VerifyOffset(verifier, VT_ERROR_MESSAGE) &&
           verifier.VerifyString(error_message()) &&
           VerifyOffset(verifier, VT_DEVICES) &&
           verifier.VerifyVector(devices()) &&
           VerifyOffset(verifier, VT_TENSOR_SIZES) &&
           verifier.VerifyVector(tensor_sizes()) &&
           VerifyField<int8_t>(verifier, VT_TENSOR_TYPE) &&
           VerifyField<double>(verifier, VT_PRESCALE_FACTOR) &&
           VerifyField<double>(verifier, VT_POSTSCALE_FACTOR) &&
           VerifyField<int32_t>(verifier, VT_LAST_JOINED_RANK) &&
           verifier.EndTable();
  }
};

struct ResponseBuilder {
  typedef Response Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_response_type(horovod::common::wire::ResponseType response_type) {
    fbb_.AddElement<int8_t>(Response::VT_RESPONSE_TYPE, static_cast<int8_t>(response_type), 0);
  }
  void add_tensor_names(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> tensor_names) {
    fbb_.AddOffset(Response::VT_TENSOR_NAMES, tensor_names);
  }
  void add_error_message(flatbuffers::Offset<flatbuffers::String> error_message) {
    fbb_.AddOffset(Response::VT_ERROR_MESSAGE, error_message);
  }
  void add_devices(flatbuffers::Offset<flatbuffers::Vector<int32_t>> devices) {
    fbb_.AddOffset(Response::VT_DEVICES, devices);
  }
  void add_tensor_sizes(flatbuffers::Offset<flatbuffers::Vector<int64_t>> tensor_sizes) {
    fbb_.AddOffset(Response::VT_TENSOR_SIZES, tensor_sizes);
  }
  void add_tensor_type(horovod::common::wire::DataType tensor_type) {
    fbb_.AddElement<int8_t>(Response::VT_TENSOR_TYPE, static_cast<int8_t>(tensor_type), 0);
  }
  void add_prescale_factor(double prescale_factor) {
    fbb_.AddElement<double>(Response::VT_PRESCALE_FACTOR, prescale_factor, 0.0);
  }
  void add_postscale_factor(double postscale_factor) {
    fbb_.AddElement<double>(Response::VT_POSTSCALE_FACTOR, postscale_factor, 0.0);
  }
  void add_last_joined_rank(int32_t last_joined_rank) {
    fbb_.AddElement<int32_t>(Response::VT_LAST_JOINED_RANK, last_joined_rank, 0);
  }
  explicit ResponseBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<Response> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<Response>(end);
    return o;
  }
};

inline flatbuffers::Offset<Response> CreateResponse(
    flatbuffers::FlatBufferBuilder &_fbb,
    horovod::common::wire::ResponseType response_type = horovod::common::wire::ResponseType_ALLREDUCE,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> tensor_names = 0,
    flatbuffers::Offset<flatbuffers::String> error_message = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> devices = 0,
    flatbuffers::Offset<flatbuffers::Vector<int64_t>> tensor_sizes = 0,
    horovod::common::wire::DataType tensor_type = horovod::common::wire::DataType_HOROVOD_UINT8,
    double prescale_factor = 0.0,
    double postscale_factor = 0.0,
    int32_t last_joined_rank = 0) {
  ResponseBuilder builder_(_fbb);
  builder_.add_postscale_factor(postscale_factor);
  builder_.add_prescale_factor(prescale_factor);
  builder_.add_last_joined_rank(last_joined_rank);
  builder_.add_tensor_sizes(tensor_sizes);
  builder_.add_devices(devices);
  builder_.add_error_message(error_message);
  builder_.add_tensor_names(tensor_names);
  builder_.add_tensor_type(tensor_type);
  builder_.add_response_type(response_type);
  return builder_.Finish();
}

inline flatbuffers::Offset<Response> CreateResponseDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    horovod::common::wire::ResponseType response_type = horovod::common::wire::ResponseType_ALLREDUCE,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *tensor_names = nullptr,
    const char *error_message = nullptr,
    const std::vector<int32_t> *devices = nullptr,
    const std::vector<int64_t> *tensor_sizes = nullptr,
    horovod::common::wire::DataType tensor_type = horovod::common::wire::DataType_HOROVOD_UINT8,
    double prescale_factor = 0.0,
    double postscale_factor = 0.0,
    int32_t last_joined_rank = 0) {
  auto tensor_names__ = tensor_names ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*tensor_names) : 0;
  auto error_message__ = error_message ? _fbb.CreateString(error_message) : 0;
  auto devices__ = devices ? _fbb.CreateVector<int32_t>(*devices) : 0;
  auto tensor_sizes__ = tensor_sizes ? _fbb.CreateVector<int64_t>(*tensor_sizes) : 0;
  return horovod::common::wire::CreateResponse(
      _fbb,
      response_type,
      tensor_names__,
      error_message__,
      devices__,
      tensor_sizes__,
      tensor_type,
      prescale_factor,
      postscale_factor,
      last_joined_rank);
}

struct ResponseList FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ResponseListBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_RESPONSES = 4,
    VT_SHUTDOWN = 6
  };
  const flatbuffers::Vector<flatbuffers::Offset<horovod::common::wire::Response>> *responses() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<horovod::common::wire::Response>> *>(VT_RESPONSES);
  }
  bool shutdown() const {
    return GetField<uint8_t>(VT_SHUTDOWN, 0) != 0;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_RESPONSES) &&
           verifier.VerifyVector(responses()) &&
           verifier.VerifyVectorOfTables(responses()) &&
           VerifyField<uint8_t>(verifier, VT_SHUTDOWN) &&
           verifier.EndTable();
  }
};

struct ResponseListBuilder {
  typedef ResponseList Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_responses(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<horovod::common::wire::Response>>> responses) {
    fbb_.AddOffset(ResponseList::VT_RESPONSES, responses);
  }
  void add_shutdown(bool shutdown) {
    fbb_.AddElement<uint8_t>(ResponseList::VT_SHUTDOWN, static_cast<uint8_t>(shutdown), 0);
  }
  explicit ResponseListBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<ResponseList> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ResponseList>(end);
    return o;
  }
};

inline flatbuffers::Offset<ResponseList> CreateResponseList(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<horovod::common::wire::Response>>> responses = 0,
    bool shutdown = false) {
  ResponseListBuilder builder_(_fbb);
  builder_.add_responses(responses);
  builder_.add_shutdown(shutdown);
  return builder_.Finish();
}

inline flatbuffers::Offset<ResponseList> CreateResponseListDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<horovod::common::wire::Response>> *responses = nullptr,
    bool shutdown = false) {
  auto responses__ = responses ? _fbb.CreateVector<flatbuffers::Offset<horovod::common::wire::Response>>(*responses) : 0;
  return horovod::common::wire::CreateResponseList(
      _fbb,
      responses__,
      shutdown);
}

}  // namespace wire
}  // namespace common
}  // namespace horovod

#endif  // FLATBUFFERS_GENERATED_MESSAGE_HOROVOD_COMMON_WIRE_H_
