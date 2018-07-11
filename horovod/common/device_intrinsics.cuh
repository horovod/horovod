// Basic GPU intrinsics
// by Yuechao Pan
// for NVIDIA

#pragma once

__device__ int64_t atomicAdd(int64_t* addr, int64_t val)
{
  return (int64_t)atomicAdd(
    (unsigned long long*)addr,
    (unsigned long long )val);
}

__device__ uint64_t atomicAdd(uint64_t* addr, uint64_t val)
{
  return (uint64_t)atomicAdd(
    (unsigned long long*)addr,
    (unsigned long long )val);
}
