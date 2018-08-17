// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
// Basic GPU intrinsics

#pragma once

#include "dgc.h"

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

__device__ int64_t atomicMax(int64_t* addr, int64_t val)
{
  return (int64_t)atomicMax(
    (signed long long*)addr,
    (signed long long )val);
}

__device__ float atomicMin(float* addr, float val)
{
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        old = atomicCAS(addr_as_int, expected,
          __float_as_int(fminf(val, __int_as_float(expected))));
    } while (expected != old);
    return __int_as_float(old);
}

__device__ long long atomicCAS(long long* addr, long long compare, long long val)
{
  return (long long)atomicCAS(
    (unsigned long long*)addr,
    (unsigned long long )compare,
    (unsigned long long )val);
}

__device__ static float atomicMax(float* addr, float val)
{
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        old = atomicCAS(addr_as_int, expected,
          __float_as_int(fmaxf(val, __int_as_float(expected))));
    } while (expected != old);
    return __int_as_float(old);
}

__device__ static double atomicMax(double* addr, double val)
{
    long long* addr_as_longlong = (long long*)addr;
    long long old = *addr_as_longlong;
    long long expected;
    do {
        expected = old;
        old = atomicCAS(addr_as_longlong, expected,
          __double_as_longlong(fmax(val, __longlong_as_double(expected))));
    } while (expected != old);
    return __longlong_as_double(old);
}
template <typename T, typename SizeT, typename Compare>
__device__ SizeT binarySearch(T* elements, SizeT lower_bound, SizeT upper_bound,
  T element_to_find, Compare lessThan)
{
  while (lower_bound < upper_bound)
  {
    SizeT mid_point = (lower_bound + upper_bound) >> 1;
    auto element = elements[mid_point];
    if (lessThan(element, element_to_find))
      lower_bound = mid_point + 1;
    else upper_bound = mid_point;
  }

  SizeT retval = horovod::dgc::PreDefinedValues<SizeT>::InvalidValue;
  if (upper_bound == lower_bound)
  {
    if (lessThan(element_to_find, elements[upper_bound]))
      retval = upper_bound - 1;
    else
      retval = upper_bound;
  }
  return retval;
}

template <typename T, typename SizeT>
__device__ SizeT binarySearch(T* elements, SizeT lower_bound, SizeT upper_bound,
  T element_to_find)
{
  return binarySearch(elements, lower_bound, upper_bound, element_to_find,
    []__device__ (const T &a, const T &b)
    {
      return (a < b);
    });
}
