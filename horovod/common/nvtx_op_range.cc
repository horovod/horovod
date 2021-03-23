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
  REGISTER_STRING(HorovodBroadcast);
  REGISTER_STRING(HorovodAlltoall);
#undef REGISTER_STRING
}

NvtxOpsHandle::~NvtxOpsHandle() {
  nvtxDomainDestroy(domain_);
}

NvtxOpsHandle NvtxOpRange::nvtx_ops_handle_;

} // namespace common
} // namespace horovod

#endif // HAVE_NVTX
