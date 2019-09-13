// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Microsoft Corp.
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

#ifndef HOROVOD_MSALLREDUCE_OPERATIONS_H
#define HOROVOD_MSALLREDUCE_OPERATIONS_H

#include <iostream>
#include <cstring>
#include <immintrin.h>
#include <emmintrin.h>

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "../mpi_context.h"
#include "p2p_operations.h"


namespace horovod {
namespace common {

class MsAllreduceOp : public PointToPointOp {
public:
  MsAllreduceOp(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const override;

protected:
  // TODO fix this API
  template<typename T, typename F, typename S>
  void MsAllreduce_Internal(T* gradient_buffer, T* result_buffer, int buffer_length, MPI_Comm* node_comm, int layerid, TensorTableEntry entry, F dotProdFunc, S scaleAddFunc);
  
  // TODO new parasail begin  
  template<typename T, typename F, typename S>
  void SyncLocalReduce(T *grad_buffer, T *recv_buffer, int count, MPI_Datatype mpi_type, MPI_Comm communicator, int layerid, TensorTableEntry entry, F dotProdFunc, S scaleAddFunc);
  
  template <typename T>
  void SyncLocalBroadcast(T *grad_buffer, int count, MPI_Datatype mpi_type, MPI_Comm communicator, int layerid);

  template<typename T, typename F, typename S>
  void SyncAllreduce(T* grad_buffer, T* recv_buffer, int count, MPI_Comm communicator, MPI_Comm* reduction_comms, int layerid, TensorTableEntry entry, F dotProdFunc, S scaleAddFunc);

  template<typename T>
  void static ScaledAdd(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b, HorovodGlobalState *global_state, int layerid);
  
  template<typename T, typename F, typename S>
  void PairwiseReduceWithComm(T* a, T* b, int count, int layerid, MPI_Comm& comm, bool isLeftNeighbor, F dotProdFunc, S scaleAddFunc);

  template<typename T>
  void static ComputeDotAndNormSqrds(const T* __restrict__  a, const T* __restrict__ b, int n, double& dotProduct, double& anormsq, double& bnormsq, HorovodGlobalState *global_state, int layerid);  
  
  // TODO over-write ComputeDotAndNormSqrds for float16
  inline void static ComputeDotAndNormSqrdsfp16(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, int len, double& dotProduct, double& anormsq, double& bnormsq, HorovodGlobalState *global_state, int layerid) {
      int i;
      __m256d dotProductVec = _mm256_setzero_pd();
      __m256d anormVec = _mm256_setzero_pd();
      __m256d bnormVec = _mm256_setzero_pd();
      for (i = 0; i < len - 7; i += 8) {
          __m256 aVec = _mm_loadu_ph(&a[i]);
          __m256 bVec = _mm_loadu_ph(&b[i]);
          __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
          __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
          __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
          __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
          dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
          dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
          anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
          anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
          bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
          bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
      }
      if (i < len) {
          __m256 aVec = _mm_loadu_ph_partial(&a[i], len - i);
          __m256 bVec = _mm_loadu_ph_partial(&b[i], len - i);
        __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
        __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
        __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
        __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
          dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
          dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
          anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
          anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
          bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
          bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
      }
  
      dotProduct = _mm256Reduction_pd(dotProductVec);
      anormsq = _mm256Reduction_pd(anormVec);
      bnormsq = _mm256Reduction_pd(bnormVec);
  }

  inline void static ScaledAddfp16(int len, double acoeff, uint16_t* __restrict__ a, double bcoeff, uint16_t* __restrict__ b, HorovodGlobalState *global_state, int layerid) {
      int i;
      __m256 acoeffVec = _mm256_set1_ps((float)(acoeff));
      __m256 bcoeffVec = _mm256_set1_ps((float)bcoeff);
      for (i = 0; i < len - 7; i += 8) {
          __m256 aVec = _mm_loadu_ph(&a[i]);
          __m256 bVec = _mm_loadu_ph(&b[i]);
          aVec = _mm256_mul_ps(acoeffVec, aVec);
          _mm_store_ph(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec));
      }
      if (i < len) {
          __m256 aVec = _mm_loadu_ph_partial(&a[i], len - i);
          __m256 bVec = _mm_loadu_ph_partial(&b[i], len - i);
          aVec = _mm256_mul_ps(acoeffVec, aVec);
          _mm_store_ph_partial(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec), len - i);
      }
  }

  void virtual memcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid);

private:
  // reduces 8xfloat32 into one scalar
  inline float static  _mm256Reduction(__m256 x) {
      const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
      const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
      const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
      return _mm_cvtss_f32(x32);
  }

  // reduce 4xfloat64 into one double
  inline double static _mm256Reduction_pd(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128
    
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar      
  }

  // load 8 float16s from a and return the __m256 register
  inline __m256 static _mm_loadu_ph(const uint16_t* a) {
      __m128i r = _mm_loadu_si128((__m128i*)(a));
      return _mm256_cvtph_ps(r);
  }

  // store 8 float16 from val into a 
  inline void static _mm_store_ph(uint16_t* a, __m256 val) {
      __m128i r = _mm256_cvtps_ph(val, 0);
      _mm_storeu_si128((__m128i*)a, r);
  }

  // load len (< 8) float16s from a, fill the rest with 0s, and return the __m256 register
  inline __m256 static _mm_loadu_ph_partial(const uint16_t* a, int len) {
      short e[8];
      std::memset(e, 0, sizeof(e));
      std::memcpy(e, a, std::min(len, 8) * sizeof(short));
      __m128i es = _mm_set_epi16(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
      return _mm256_cvtph_ps(es);
  }

  // store the first len (< 8) float16s from val and store into a
  inline void static _mm_store_ph_partial(uint16_t* a, __m256 val, int len) {
      __m128i r = _mm256_cvtps_ph(val, 0);
      //for (int i = 0; i < std::min(len, 8); i++) 
      //    a[i].value = _mm_extract_epi16(r, i);
      // but we cannot do this because the second argument to _mm_extract_epi16 has to be a compile time constant 
      if (0 < len) a[0] = (short)_mm_extract_epi16(r, 0);
      if (1 < len) a[1] = (short)_mm_extract_epi16(r, 1);
      if (2 < len) a[2] = (short)_mm_extract_epi16(r, 2);
      if (3 < len) a[3] = (short)_mm_extract_epi16(r, 3);
      if (4 < len) a[4] = (short)_mm_extract_epi16(r, 4);
      if (5 < len) a[5] = (short)_mm_extract_epi16(r, 5);
      if (6 < len) a[6] = (short)_mm_extract_epi16(r, 6);
      if (7 < len) a[7] = (short)_mm_extract_epi16(r, 7);
  }
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MSALLREDUCE_OPERATIONS_H
