#ifndef HOROVOD_NVTX_OP_RANGE_H
#define HOROVOD_NVTX_OP_RANGE_H

#if HAVE_NVTX
#include <memory>
#include <nvtx3/nvToolsExt.h>
#else
#include <cstdint>
#endif // HAVE_NVTX


namespace horovod {
namespace common {

enum class RegisteredNvtxOp {
  HorovodAllreduce = 0,
  HorovodGroupedAllreduce,
  HorovodAllgather,
  HorovodGroupedAllgather,
  HorovodBroadcast,
  HorovodAlltoall,
  HorovodReducescatter,
  HorovodGroupedReducescatter,
  // Insert values for new ops above this line. Also add corresponding
  // REGISTER_STRING lines in the constructor NvtxOpsHandle::NvtxOpsHandle().
  END,
};

#if HAVE_NVTX
class NvtxOpsHandle {
public:
  NvtxOpsHandle() noexcept;
  ~NvtxOpsHandle();

  inline nvtxRangeId_t StartRange(RegisteredNvtxOp msg, int64_t payload) {
    if (domain_ == nullptr) {
      return invalid_range_id;
    }

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    eventAttrib.message.registered = op_names_[static_cast<int>(msg)];
    eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    eventAttrib.payload.llValue = payload;

    nvtxRangeId_t range_id = nvtxDomainRangeStartEx(domain_, &eventAttrib);
    return range_id;
  }

  inline void EndRange(nvtxRangeId_t range_id) {
    if (domain_ == nullptr || range_id == invalid_range_id) {
      return;
    }
    nvtxDomainRangeEnd(domain_, range_id);
  }

  void Disable();

  static constexpr nvtxRangeId_t invalid_range_id = 0xfffffffffffffffful;

private:
  nvtxDomainHandle_t domain_;   // nullptr if disabled
  nvtxStringHandle_t op_names_[static_cast<int>(RegisteredNvtxOp::END)];
};

class NvtxOpRange {
public:
  NvtxOpRange(RegisteredNvtxOp msg, int64_t payload)
      : range_id_(nvtx_ops_handle.StartRange(msg, payload)) {
  }

  ~NvtxOpRange() { nvtx_ops_handle.EndRange(range_id_); }

  static NvtxOpsHandle nvtx_ops_handle;

private:
  nvtxRangeId_t range_id_;
};

class SharedNvtxOpRange {
public:
  void Start(RegisteredNvtxOp msg, int64_t payload) {
    p_ = std::make_shared<NvtxOpRange>(msg, payload);
  }

  void End() {
    p_.reset();
  }

private:
  std::shared_ptr<NvtxOpRange> p_;
};

#else // HAVE_NVTX
class SharedNvtxOpRange {
public:
  void Start(RegisteredNvtxOp msg, int64_t payload) { }
  void End() { }
};
#endif // HAVE_NVTX

} // namespace common
} // namespace horovod

#endif // HOROVOD_NVTX_OP_RANGE_H
