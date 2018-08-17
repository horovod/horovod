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
// DGC data type definations

#pragma once

#include <mpi.h>
#include <nccl.h>

namespace horovod {
namespace dgc {

template <typename T>
struct PreDefinedValues{};

template <>
struct PreDefinedValues<float>
{
  static const ncclDataType_t NCCLDataType = ncclFloat32;
  MPI_Datatype getMpiDataType() {return MPI_FLOAT;}
  constexpr static const float InvalidValue = NAN;
};

template <>
struct PreDefinedValues<double>
{
  static const ncclDataType_t NCCLDataType = ncclFloat64;
  MPI_Datatype getMpiDataType() {return MPI_DOUBLE;}
  constexpr static const double InvalidValue = NAN;
};

template <>
struct PreDefinedValues<int32_t>
{
  static const ncclDataType_t NCCLDataType = ncclInt32;
  static MPI_Datatype getMpiDataType() {return MPI_INT;}
  static const int32_t AllZeros = (int32_t)0;
  static const int32_t AllOnes  = ~AllZeros;
  static const int32_t InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<uint32_t>
{
  static const ncclDataType_t NCCLDataType = ncclUint32;
  static MPI_Datatype getMpiDataType() {return MPI_UNSIGNED;}
  static const uint32_t AllZeros = (uint32_t)0;
  static const uint32_t AllOnes  = ~AllZeros;
  static const uint32_t InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<int64_t>
{
  static const ncclDataType_t NCCLDataType = ncclInt64;
  static MPI_Datatype getMpiDataType() {return MPI_LONG_LONG;}
  static const int64_t AllZeros = (int64_t)0;
  static const int64_t AllOnes  = ~AllZeros;
  static const int64_t InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<uint64_t>
{
  static const ncclDataType_t NCCLDataType = ncclUint64;
  static MPI_Datatype getMpiDataType() {return MPI_UNSIGNED_LONG_LONG;}
  static const uint64_t AllZeros = (uint64_t)0;
  static const uint64_t AllOnes  = ~AllZeros;
  static const uint64_t InvalidValue = AllOnes;
};

template <typename T>
__device__ __host__ __forceinline__
bool isValid(const T &val)
{
    return (val != PreDefinedValues<T>::InvalidValue);
}

template <>
__device__ __host__ __forceinline__
bool isValid(const float &val)
{
    return (!isnan(val));
}

template <>
__device__ __host__ __forceinline__
bool isValid(const double &val)
{
    return (!isnan(val));
}

template <>
__device__ __host__ __forceinline__
bool isValid(const long double &val)
{
    return (!isnan(val));
}

} // end of namespace dgc
} // end of namespace horovod
