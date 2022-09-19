// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#include "half.h"

#include <algorithm>

#if __AVX__ && __F16C__
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace horovod {
namespace common {

#if __AVX__ && __F16C__
// Query CPUID to determine AVX and F16C runtime support.
bool is_avx_and_f16c() {
  static bool initialized = false;
  static bool result = false;
  if (!initialized) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
      result = (ecx & bit_AVX) && (ecx & bit_F16C);
    }
    initialized = true;
  }
  return result;
}
#endif

#if HAVE_MPI
// float16 custom data type summation operation.
void float16_sum(void* invec, void* inoutvec, int* len,
                 MPI_Datatype* datatype) {
  // cast invec and inoutvec to your float16 type
  auto* in = (unsigned short*)invec;
  auto* inout = (unsigned short*)inoutvec;

  int i = 0;
#if __AVX__ && __F16C__
  if (is_avx_and_f16c()) {
    for (; i < (*len / 8) * 8; i += 8) {
      // convert in & inout to m256
      __m256 in_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in + i)));
      __m256 inout_m256 =
          _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(inout + i)));

      // add them together to new_inout_m256
      __m256 new_inout_m256 = _mm256_add_ps(in_m256, inout_m256);

      // convert back and store in inout
      __m128i new_inout_m128i = _mm256_cvtps_ph(new_inout_m256, 0);
      _mm_storeu_si128((__m128i*)(inout + i), new_inout_m128i);
    }
  }
#endif
  for (; i < *len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float += in_float;
    Float2HalfBits(&inout_float, inout + i);
  }
}

// float16 custom data type min operation.
void float16_min(void* invec, void* inoutvec, int* len,
                 MPI_Datatype* datatype) {
  // cast invec and inoutvec to your float16 type
  auto* in = (unsigned short*)invec;
  auto* inout = (unsigned short*)inoutvec;

  for (int i = 0; i < *len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float = std::min(inout_float, in_float);
    Float2HalfBits(&inout_float, inout + i);
  }
}

// float16 custom data type max operation.
void float16_max(void* invec, void* inoutvec, int* len,
                 MPI_Datatype* datatype) {
  // cast invec and inoutvec to your float16 type
  auto* in = (unsigned short*)invec;
  auto* inout = (unsigned short*)inoutvec;

  for (int i = 0; i < *len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float = std::max(inout_float, in_float);
    Float2HalfBits(&inout_float, inout + i);
  }
}

// float16 custom data type product operation.
void float16_prod(void* invec, void* inoutvec, int* len,
                  MPI_Datatype* datatype) {
  // cast invec and inoutvec to your float16 type
  auto* in = (unsigned short*)invec;
  auto* inout = (unsigned short*)inoutvec;

  int i = 0;
#if __AVX__ && __F16C__
  if (is_avx_and_f16c()) {
    for (; i < (*len / 8) * 8; i += 8) {
      // convert in & inout to m256
      __m256 in_m256 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(in + i)));
      __m256 inout_m256 =
          _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(inout + i)));

      // add them together to new_inout_m256
      __m256 new_inout_m256 = _mm256_mul_ps(in_m256, inout_m256);

      // convert back and store in inout
      __m128i new_inout_m128i = _mm256_cvtps_ph(new_inout_m256, 0);
      _mm_storeu_si128((__m128i*)(inout + i), new_inout_m128i);
    }
  }
#endif
  for (; i < *len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float *= in_float;
    Float2HalfBits(&inout_float, inout + i);
  }
}
#endif

} // namespace common
} // namespace horovod
