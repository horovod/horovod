#include "nvtx_op_range.h"

#if HAVE_NVTX

namespace horovod {
namespace common {

NvtxOpsHandle::NvtxOpsHandle() noexcept
    : domain_(nvtxDomainCreateA("HorovodOps")), op_names_{}
{
#define REGISTER_STRING(op) op_names_[static_cast<int>(RegisteredNvtxOp::op)] = nvtxDomainRegisterStringA(domain_, #op)
  REGISTER_STRING(HorovodAllreduce);
  REGISTER_STRING(HorovodGroupedAllreduce);
  REGISTER_STRING(HorovodAllgather);
  REGISTER_STRING(HorovodGroupedAllgather);
  REGISTER_STRING(HorovodBroadcast);
  REGISTER_STRING(HorovodAlltoall);
  REGISTER_STRING(HorovodReducescatter);
  REGISTER_STRING(HorovodGroupedReducescatter);
#undef REGISTER_STRING
}

NvtxOpsHandle::~NvtxOpsHandle() {
  Disable();
}

void NvtxOpsHandle::Disable() {
  if (domain_ != nullptr) {
    nvtxDomainDestroy(domain_);
    domain_ = nullptr;
  }
}

NvtxOpsHandle NvtxOpRange::nvtx_ops_handle;

} // namespace common
} // namespace horovod

#endif // HAVE_NVTX
