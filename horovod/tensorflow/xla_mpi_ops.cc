// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/human_readable_json.h"

#if HAVE_GPU

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                   \
        << "CUDA: " << cudaGetErrorString(e);                                  \
  }

#endif // HAVE_GPU

#define OMPI_SKIP_MPICXX
#include "../common/operations.h"
#include "../common/utils/env_parser.h"
#include "./custom_call_config_generated.h"

using namespace tensorflow;

namespace horovod {
namespace xla {
namespace {

common::DataType GetHVDType(::xla::PrimitiveType type) {
  switch (type) {
  case ::xla::U8:
    return common::HOROVOD_UINT8;
  case ::xla::S8:
    return common::HOROVOD_INT8;
  case ::xla::U16:
    return common::HOROVOD_UINT16;
  case ::xla::S16:
    return common::HOROVOD_INT16;
  case ::xla::S32:
    return common::HOROVOD_INT32;
  case ::xla::S64:
    return common::HOROVOD_INT64;
  case ::xla::F16:
    return common::HOROVOD_FLOAT16;
  case ::xla::F32:
    return common::HOROVOD_FLOAT32;
  case ::xla::F64:
    return common::HOROVOD_FLOAT64;
  case ::xla::PRED:
    return common::HOROVOD_BOOL;
  default:
    throw std::logic_error("Invalid XLA tensor type.");
  }
}

#if HAVE_GPU
class XLAReadyEvent : public common::ReadyEvent {
public:
  XLAReadyEvent(cudaStream_t stream) : stream_(stream) {
    CUDA_CALL(cudaEventCreate(&event_));
    CUDA_CALL(cudaEventRecord(event_, stream));
  }
  ~XLAReadyEvent() { CUDA_CALL(cudaEventDestroy(event_)); }

  bool Ready() const override {
    cudaError_t result = cudaEventQuery(event_);
    return cudaErrorNotReady != result;
  }
  gpuEvent_t event() const override {
    return event_;
  }

private:
  cudaStream_t stream_; // Not Owned.
  cudaEvent_t event_;   // Owned.
};
#endif

class XLATensor : public common::Tensor {
public:
  XLATensor(common::DataType type, common::TensorShape shape, void* buffer)
      : type_(type), shape_(std::move(shape)), buffer_(buffer) {}

  virtual const common::DataType dtype() const override { return type_; }
  virtual const common::TensorShape shape() const override { return shape_; }
  virtual const void* data() const override { return buffer_; }
  virtual int64_t size() const override {
    return shape_.num_elements() * common::DataType_Size(type_);
  }

protected:
  common::DataType type_;
  common::TensorShape shape_;
  void* buffer_; // Not owned.
};

class XLAOpContext : public common::OpContext {
public:
  XLAOpContext(int device) : device_(device) {}

  virtual common::Status AllocatePersistent(
      int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) override;
  virtual common::Status
  AllocateOutput(common::TensorShape shape,
                 std::shared_ptr<common::Tensor>* tensor) override;
  //virtual common::Status
  //AllocateOutput(int output_index, common::TensorShape shape,
  //               std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Status
  AllocateZeros(int64_t num_elements, common::DataType dtype,
                std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Framework framework() const override;

private:
  int device_;
};

class XLAPersistentBuffer : public common::PersistentBuffer {
public:
  XLAPersistentBuffer(int device, int64_t size);
  virtual const void*
  AccessData(std::shared_ptr<common::OpContext> context) const override;

private:
  int device_;
  void* buffer_;
};

XLAPersistentBuffer::XLAPersistentBuffer(int device, int64_t size)
    : device_(device) {
#if HAVE_CUDA
  int restore_device;
  CUDA_CALL(cudaGetDevice(&restore_device));
  CUDA_CALL(cudaSetDevice(device));
  CUDA_CALL(cudaMalloc((void**)&buffer_, size));
  CUDA_CALL(cudaSetDevice(restore_device));
#else
  throw std::logic_error("Internal error. Requested XLAPersistentBuffer "
                         "with GPU device but not compiled with CUDA.");
#endif
}

const void* XLAPersistentBuffer::AccessData(
    std::shared_ptr<common::OpContext> /*context*/) const {
  return buffer_;
}

common::Status XLAOpContext::AllocatePersistent(
    int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) {
  *tensor = std::make_shared<XLAPersistentBuffer>(device_, size);
  return common::Status::OK();
}

common::Status
XLAOpContext::AllocateOutput(common::TensorShape shape,
                             std::shared_ptr<common::Tensor>* tensor) {
  // Let XLA allocate I/O buffers.
  return common::Status::PreconditionError(
      "AllocateOutput is not supported for XLA.");
}

//common::Status
//XLAOpContext::AllocateOutput(int output_index, common::TensorShape shape,
//                             std::shared_ptr<common::Tensor>* tensor) {
//  // Let XLA allocate I/O buffers.
//  return common::Status::PreconditionError(
//      "AllocateOutput is not supported for XLA.");
//}

common::Status
XLAOpContext::AllocateZeros(int64_t num_elements, common::DataType dtype,
                            std::shared_ptr<common::Tensor>* tensor) {
  // Let XLA allocate I/O buffers.
  return common::Status::PreconditionError(
      "AllocateZeros is not supported for XLA.");
}

common::Framework XLAOpContext::framework() const {
  return common::Framework::XLA;
}

// On GPU this event will signal that data is ready, and tensors are
// allocated.
#if HAVE_GPU
common::ReadyEvent* RecordReadyEvent(cudaStream_t stream) {
  return new XLAReadyEvent(stream);
}
#endif

class CustomCallConfig {
public:
  std::string SerializeToString();
  void ParseFromString(std::string);

public:
  std::string tensor_name_;
  common::DataType tensor_type_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  float prescale_factor_;
  float postscale_factor_;
  int root_rank_;
  int reduce_op_;
};

std::string CustomCallConfig::SerializeToString() {
  flatbuffers::FlatBufferBuilder fbb(1024);

  std::vector<flatbuffers::Offset<wire::TensorShape>> input_shapes_obj;
  absl::c_for_each(input_shapes_, [&](const std::vector<int64_t>& dims) {
    input_shapes_obj.push_back(wire::CreateTensorShapeDirect(fbb, &dims));
  });
  std::vector<flatbuffers::Offset<wire::TensorShape>> output_shapes_obj;
  absl::c_for_each(output_shapes_, [&](const std::vector<int64_t>& dims) {
    output_shapes_obj.push_back(wire::CreateTensorShapeDirect(fbb, &dims));
  });
  auto wire = wire::CreateCustomCallConfigDirect(
      fbb, tensor_name_.c_str(), (common::wire::DataType)tensor_type_,
      &input_shapes_obj, &output_shapes_obj, prescale_factor_,
      postscale_factor_, root_rank_, reduce_op_);
  fbb.Finish(wire);

  uint8_t* buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();
  return std::string((char*)buf, size);
}

void CustomCallConfig::ParseFromString(std::string input) {
  const wire::CustomCallConfig* obj =
      flatbuffers::GetRoot<wire::CustomCallConfig>(
          (const uint8_t*)input.data());

  tensor_name_ = obj->tensor_name()->str();
  tensor_type_ = (common::DataType)obj->tensor_type();
  for (auto it = obj->input_shapes()->begin(); it != obj->input_shapes()->end();
       it++) {
    auto shape_obj = *it;
    input_shapes_.push_back(std::vector<int64_t>(shape_obj->dims()->begin(),
                                                 shape_obj->dims()->end()));
  }
  for (auto it = obj->output_shapes()->begin();
       it != obj->output_shapes()->end(); it++) {
    auto shape_obj = *it;
    output_shapes_.push_back(std::vector<int64_t>(shape_obj->dims()->begin(),
                                                  shape_obj->dims()->end()));
  }
  prescale_factor_ = obj->prescale_factor();
  postscale_factor_ = obj->postscale_factor();
  root_rank_ = obj->root_rank();
  reduce_op_ = obj->reduce_op();

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "tensor_name " << tensor_name_;
    VLOG(2) << "tensor_type " << tensor_type_;
    VLOG(2) << "prescale_factor = " << prescale_factor_;
    VLOG(2) << "postscale_factor = " << postscale_factor_;
    VLOG(2) << "root_rank = " << root_rank_;
    VLOG(2) << "reduce_op = " << reduce_op_;
  }
}

class HVDAllreduceOp : public XlaOpKernel {
public:
  explicit HVDAllreduceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_op", &reduce_op_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prescale_factor", &prescale_factor_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("postscale_factor", &postscale_factor_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ignore_name_scope", &ignore_name_scope_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    node_name_ = name();
    if (ignore_name_scope_) {
      auto pos = node_name_.find_last_of('/');
      if (pos != std::string::npos) {
        node_name_ = node_name_.substr(pos + 1);
      }
    }

    // for DEBUG, use this to confirm the XLA lowering occurs in CI.
    // TODO: remove it later.
    std::cerr << "XLA: Compile " << node_name_;

    // Generate below HLOs:
    //     out = custom-call(in), custom_call_target="CallbackHVDAllreduce"
    //     out' = custom-call(out),
    //         custom_call_target="CallbackHVDAllreduceDone"
    // Note that tensors in, out, and out' are all aliased. It means that the
    // all-reduce operation is in-place.
    ::xla::XlaBuilder* const b = ctx->builder();
    // First, generate HVDAllreduce.
    std::vector<
        std::pair<::xla::ShapeIndex, std::pair<int64, ::xla::ShapeIndex>>>
        output_operand_aliasing = {
            {::xla::ShapeIndex{}, {0, ::xla::ShapeIndex{}}}};
    auto input = ctx->Input(0);
    ::xla::XlaOp allreduce_start =
        b->ReportErrorOrReturn(BuildAllreduceCustomCall(
            b, {input}, /*is_start=*/true, output_operand_aliasing));
    // Then, generate HVDAllreduceDone.
    ::xla::XlaOp allreduce_end = b->ReportErrorOrReturn(
        BuildAllreduceCustomCall(b, {allreduce_start},
                                 /*is_start=*/false, output_operand_aliasing));
    ctx->SetOutput(0, allreduce_end);
    return;
  }

private:
  ::xla::StatusOr<::xla::XlaOp> BuildAllreduceCustomCall(
      ::xla::XlaBuilder* b, absl::Span<const ::xla::XlaOp> operands,
      bool is_start,
      absl::Span<const std::pair<::xla::ShapeIndex,
                                 std::pair<int64, ::xla::ShapeIndex>>>
          output_operand_aliasing = {});

private:
  std::string node_name_;
  int reduce_op_;
  // Using float since TF does not support double OP attributes
  float prescale_factor_;
  float postscale_factor_;
  bool ignore_name_scope_;
};

// Implement this customized registrar so that we can make XLA Ops an opt-in,
// controlled by HOROVOD_ENABLE_XLA_OPS.
#define HVD_REGISTER_XLA_OP(NAME, OP) \
  HVD_REGISTER_XLA_OP_UNIQ_HELPER(__COUNTER__, NAME, OP)

#define HVD_REGISTER_XLA_OP_UNIQ_HELPER(COUNTER, OP_NAME, OP)       \
  HVD_REGISTER_XLA_OP_UNIQ(COUNTER, OP_NAME, OP)

#define HVD_REGISTER_XLA_OP_UNIQ(CTR, OP_NAME, OP)                  \
  static HVDXlaOpRegistrar xla_op_registrar__body__##CTR##__object( \
      OP_NAME,                                                      \
      [](::tensorflow::OpKernelConstruction* context)               \
          -> ::tensorflow::OpKernel* { return new OP(context); });

class HVDXlaOpRegistrar {
 public:
  HVDXlaOpRegistrar(string op_name,
                    ::tensorflow::XlaOpRegistry::Factory factory) {
    bool enable_xla_ops = false;
    common::SetBoolFromEnv(HOROVOD_ENABLE_XLA_OPS, enable_xla_ops, true);
    if (enable_xla_ops) {
      xla_op_registrar_ = new XlaOpRegistrar(
          ::tensorflow::XlaOpRegistrationBuilder::Name(op_name).Build(factory));
    }
  }

 private:
  XlaOpRegistrar* xla_op_registrar_;
};

HVD_REGISTER_XLA_OP("HorovodAllreduce", HVDAllreduceOp);


// Builds a custom call to a method named 'allreduce'.
::xla::StatusOr<::xla::XlaOp> HVDAllreduceOp::BuildAllreduceCustomCall(
    ::xla::XlaBuilder* b, absl::Span<const ::xla::XlaOp> operands,
    bool is_start,
    absl::Span<
        const std::pair<::xla::ShapeIndex, std::pair<int64, ::xla::ShapeIndex>>>
        output_operand_aliasing) {
  string call_target_name =
      is_start ? "CallbackHVDAllreduce" : "CallbackHVDAllreduceDone";
  CustomCallConfig config;
  config.tensor_name_ = node_name_;
  for (const ::xla::XlaOp& opnd : operands) {
    TF_ASSIGN_OR_RETURN(::xla::Shape shape, b->GetShape(opnd));
    config.input_shapes_.push_back(std::vector<int64_t>(
        shape.dimensions().begin(), shape.dimensions().end()));
  }
  TF_ASSIGN_OR_RETURN(::xla::Shape output_shape, b->GetShape(operands.at(0)));
  config.output_shapes_.push_back(std::vector<int64_t>(
      output_shape.dimensions().begin(), output_shape.dimensions().end()));
  config.tensor_type_ = GetHVDType(output_shape.element_type());
  config.prescale_factor_ = prescale_factor_;
  config.postscale_factor_ = postscale_factor_;
  config.reduce_op_ = reduce_op_;

  return ::xla::CustomCall(
      b, call_target_name, operands, output_shape, config.SerializeToString(),
      /*has_side_effect=*/false, output_operand_aliasing, /*literal=*/nullptr,
      is_start ? ::xla::CustomCallSchedule::EARLIEST
               : ::xla::CustomCallSchedule::LATEST);
}

// Returns a hash for rendezvous.
uint64 GetRendezvousKeyHash(const string& key) {
  string k = strings::StrCat(key);
  return Hash64(k.data(), k.size());
}

class HVDCustomCallRendezvous {
public:
  struct Payload {
    std::shared_ptr<gpuEvent_t> event;
  };

  // This `Signal` method either a) flips the `Ready` value in the front of the
  // queue if the front value is false, or otherwise b) enqueues a `Ready=true`
  // into the queue.
  // Requirement: tensor_name shall be unique in a graph.
  void Signal(string tensor_name, common::Event hvd_event) {
    uint64 key_hash = GetRendezvousKeyHash(tensor_name);
    mutex_lock l(mu_);
    InitQueue(key_hash);

    Queue& queue = *table_[key_hash];
    if (queue.empty() || queue.front()!=nullptr) {
      // No earlier waiters exist, so just leave a signal.
      queue.push_back(new Payload{hvd_event.event});
      return;
    }

    // There is an earlier waiter to consume this signal. Place payload
    // there.
    queue.front() = new Payload{hvd_event.event};
  }

  // The `Wait` method waits for a `Ready==true` event. We assume there is at
  // most one outstanding `Wait` call to simplfy the implementation. As such,
  // this method always operates on the front value of the queue.
  void Wait(string tensor_name, CUstream stream) {
    uint64 key_hash = GetRendezvousKeyHash(tensor_name);

    {
      mutex_lock l(mu_);
      InitQueue(key_hash);
      Queue& queue = *table_[key_hash];
      if (queue.empty()) {
        // So long as the queue is empty, place a signal but set it to false.
        // We'll then wait for the `Signal` method to flip the Boolean value.
        queue.push_back(nullptr);
      }
    }

    auto test_signal_value = [&]() {
      mutex_lock l(mu_);
      Queue& queue = *table_[key_hash];
      return nullptr!=queue.front();
    };
    while (!test_signal_value()) {
      // Busy waiting. As we don't anticipate the blocking occurs frequently,
      // this busy waiting should be fine. If this creates any performance
      // overhead, implement conditional var wait.
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }

    mutex_lock l(mu_);
    Queue* queue = table_[key_hash];
    Payload* payload = queue->front();
    auto event = payload->event;
    queue->pop_front();
    if (queue->empty()) {
      table_.erase(key_hash);
      delete queue;
    }
    if (event) {
#if HAVE_GPU
      CUDA_CALL(cudaStreamWaitEvent(stream, *event, /*flags=*/0));
#endif
    }
    delete payload;
  }

private:
  // This method is not thread-safe.
  void InitQueue(uint64 key_hash) {
    auto it = table_.find(key_hash);
    if (it == table_.end()) {
      table_[key_hash] = new Queue();
    }
  }

private:
  // `nullptr` denotes non-readiness of the payload.
  typedef std::deque<Payload*> Queue;
  // hash to queue.
  typedef absl::flat_hash_map<uint64, Queue*> Table;

  mutex mu_;
  Table table_ GUARDED_BY(mu_);
};

/*static*/ HVDCustomCallRendezvous* GetHVDCustomCallRendezvous() {
  static HVDCustomCallRendezvous* self = new HVDCustomCallRendezvous();
  return self;
}

int GetDeviceOrdinal(void* ptr) {
#if HAVE_GPU
  cudaPointerAttributes attrs;
  CUDA_CALL(cudaPointerGetAttributes(&attrs, ptr));
  return attrs.device;
#else
  return CPU_DEVICE_ID;
#endif
}

// Custom calls
void CallbackHVDAllreduce(CUstream stream, void** buffers,
                          const char* opaque, size_t opaque_len) {
  CustomCallConfig config;
  config.ParseFromString(std::string(opaque, opaque_len));

  // Enqueue requests.
  common::ReadyEventList ready_event_list;
#if HAVE_GPU
  ready_event_list.AddReadyEvent(
      std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(stream)));
#endif
  int dev_ordinal = GetDeviceOrdinal(buffers[0]);
  auto hvd_context = std::make_shared<XLAOpContext>(dev_ordinal);
  auto hvd_input = std::make_shared<XLATensor>(
      config.tensor_type_, common::TensorShape(config.input_shapes_[0]),
      buffers[0]);
  auto hvd_output = std::make_shared<XLATensor>(
      config.tensor_type_, common::TensorShape(config.input_shapes_[0]),
      buffers[1]);
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_input, hvd_output, ready_event_list, config.tensor_name_,
      dev_ordinal,
      [=](const common::Status& status) {
        // When request is done processing, signal `HVDAllreduceDone`.
        GetHVDCustomCallRendezvous()->Signal(
            config.tensor_name_, status.event);
      },
      (horovod::common::ReduceOp)config.reduce_op_,
      (double)config.prescale_factor_, (double)config.postscale_factor_);
  CHECK(enqueue_result.ok()) << enqueue_result.reason();
}

void CallbackHVDAllreduceDone(CUstream stream, void** /*buffers*/,
                              const char* opaque, size_t opaque_len) {
  // Blocking until the request is done processing.
  VLOG(2) << "hvd-allreduce-done - Start";
  CustomCallConfig config;
  config.ParseFromString(std::string(opaque, opaque_len));
  GetHVDCustomCallRendezvous()->Wait(config.tensor_name_, stream);
  VLOG(2) << "hvd-allreduce-done - End";
}

XLA_REGISTER_CUSTOM_CALL_TARGET(CallbackHVDAllreduce, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET(CallbackHVDAllreduceDone, "CUDA");

} // namespace
} // namespace tensorflow
} // namespace horovod
