#ifndef HOROVOD_NVTX_OP_RANGE_H
#define HOROVOD_NVTX_OP_RANGE_H

#if HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif // HAVE_NVTX

namespace horovod {
namespace common {

enum class RegisteredNvtxOp {
  HorovodAllreduce = 0,
  HorovodGroupedAllreduce,
  HorovodAllgather,
  HorovodBroadcast,
  HorovodAlltoall,
  // Insert new enum values above this line
  END,
};

#if HAVE_NVTX
class NvtxOpsHandle {
public:
  NvtxOpsHandle() noexcept;
  ~NvtxOpsHandle();

  inline nvtxRangeId_t Start(RegisteredNvtxOp msg, int64_t payload) {
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

  inline void End(nvtxRangeId_t range_id) {
    nvtxDomainRangeEnd(domain_, range_id);
  }

private:
  nvtxDomainHandle_t domain_;
  nvtxStringHandle_t op_names_[static_cast<int>(RegisteredNvtxOp::END)];
};

class NvtxOpRange {
public:
  NvtxOpRange(RegisteredNvtxOp msg, int64_t payload)
      : range_id_(nvtx_ops_handle_.Start(msg, payload)) {
  }

  ~NvtxOpRange() {
    nvtx_ops_handle_.End(range_id_);
  }

private:
  static NvtxOpsHandle nvtx_ops_handle_;
  nvtxRangeId_t range_id_;
};

#else // HAVE_NVTX
class NvtxOpRange {
public:
  NvtxOpRange(RegisteredNvtxOp msg, int64_t payload) { }
  ~NvtxOpRange() = default;
};
#endif // HAVE_NVTX

} // namespace common
} // namespace horovod

#endif // HOROVOD_NVTX_OP_RANGE_H
