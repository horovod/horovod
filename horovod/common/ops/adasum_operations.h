//TODO license
#ifndef HOROVOD_ADASUM_OPERATIONS_H
#define HOROVOD_ADASUM_OPERATIONS_H

#include <iostream>
#include <cstring>
#include <immintrin.h>
#include <emmintrin.h>

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "p2p_operations.h"


namespace horovod {
namespace common {

static bool IsPowerOfTwo(ulong x)
{
  return (x != 0) && ((x & (x - 1)) == 0);
}

template <typename Communicator_type>
class AdasumOp : public PointToPointOp<Communicator_type> {
public:
  AdasumOp(HorovodGlobalState* global_state) : PointToPointOp<Communicator_type>(global_state) {
    if (this->global_state_->adasum_num_threads > 0) {
      for (int i = 0; i < this->global_state_->adasum_num_threads; i++) {
          temp_buffers_.emplace_back();
      }
    }
  };

protected:
  std::atomic_int finished_parallel_reductions_;

  std::mutex buffer_lock_;

  std::deque<FusionBufferManager> temp_buffers_;

  virtual int GetLocalRankWithComm(Communicator_type communicator) = 0;

  virtual int GetSizeWithComm(Communicator_type communicator) = 0;

  int GetPerElementSize(TensorTableEntry entry) {
   return GetPerElementSize(entry.tensor->dtype());
  }

  int GetPerElementSize(DataType horovod_datatype) {
    switch(horovod_datatype) {
        case DataType::HOROVOD_FLOAT16:
          return 2;
        case DataType::HOROVOD_FLOAT32:
          return 4;
        case DataType::HOROVOD_FLOAT64:
          return 8;
        default:
          throw std::logic_error("Unsupported data type.");
    }
  }

  virtual void DispatchComputeDotAndNormSqrds(const void* __restrict__  a,
                                              const void* __restrict__ b,
                                              DataType horovod_datatype,
                                              int count,
                                              double& dotProduct,
                                              double& anormsq,
                                              double& bnormsq,
                                              HorovodGlobalState *global_state,
                                              int layerid) = 0;
  
  template<typename T>
  void ComputeDotAndNormSqrds(const T* __restrict__  a, const T* __restrict__ b, int count, double& dotProduct, double& anormsq, double& bnormsq, HorovodGlobalState *global_state, int layerid) {
      dotProduct = 0.;
      anormsq = 0.;
      bnormsq = 0.;

      for (int i = 0; i < count; i++) {
          dotProduct += a[i] * b[i];
          anormsq += a[i] * a[i];
          bnormsq += b[i] * b[i];
      }
  }
  
  virtual void DispatchScaledAdd(DataType horovod_datatype,
                                 int count,
                                 double acoeff,
                                 void* __restrict__ a,
                                 double bcoeff,
                                 void* __restrict__ b,
                                 HorovodGlobalState *global_state,
                                 int layerid) = 0;
  
  template<typename T>
  void ScaledAdd(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b, HorovodGlobalState *global_state, int layerid) {
    for (int i = 0; i < n; i++) {
        a[i] = acoeff * a[i] + bcoeff * b[i];
    }
  }
  
  virtual void SyncLocalReduce(void *grad_buffer, void *recv_buffer, Communicator_type communicator, int layerid, TensorTableEntry entry) {
    int redn_rank;
    int true_rank = GetLocalRankWithComm(communicator);
    int size = GetSizeWithComm(communicator);
    int buffer_len = entry.tensor->size();
    DataType data_type = entry.tensor->dtype();
    int count = buffer_len / GetPerElementSize(entry);
    int root_node_rotation = false ? (layerid % size) : 0;
    redn_rank = (true_rank ^ root_node_rotation);
  
    // Do a tree reduction
    // The reduction ranks used are a permutation of true ranks (permuted based on layerid)
    // This spreads the load of tree reduction across different true ranks
  
    // at each level l, node X0[0..0] receives from X1[0...],
    // where [0..0] is l zeros in the bit representation of the rank of a node
    int level;
    for (level = 1; level < size; level *= 2) {
      int neighbor_redn_rank = redn_rank ^ level;
      int neighbor_true_rank = (neighbor_redn_rank ^ root_node_rotation);
      if (redn_rank % level != 0)
        continue; // stay idle at this level
  
      if (neighbor_redn_rank >= size)
        continue; // no neighbor and so stay idle at this level
      
      if ((redn_rank & level) == 0) {
        // recv buffer from neighbor
        this->PointToPointRecv(recv_buffer, (int64_t)buffer_len, data_type, neighbor_true_rank, layerid, communicator);
        
        double anormsq = 0, bnormsq = 0, dotProduct = 0;
        DispatchComputeDotAndNormSqrds(grad_buffer, recv_buffer, data_type, count, dotProduct, anormsq, bnormsq, AdasumOp<Communicator_type>::global_state_, layerid);
        double acoeff = 1;
        double bcoeff = 1;
        if (anormsq >= 1e-8)
  	    acoeff = 1.0 - dotProduct / anormsq * 0.5;
        if (bnormsq >= 1e-8)
  	    bcoeff = 1.0 - dotProduct / bnormsq * 0.5;
  
        DispatchScaledAdd(data_type, count, acoeff, grad_buffer, bcoeff, recv_buffer, AdasumOp<Communicator_type>::global_state_, layerid);
      }
      else {
        // send grad_buffer to neighbor
        this->PointToPointSend(grad_buffer, (int64_t)buffer_len, data_type, neighbor_true_rank, layerid, communicator);
      }
    }
  }
  
  virtual void SyncLocalBroadcast(void *grad_buffer, Communicator_type communicator, TensorTableEntry entry, int layerid) {
    // assumes broadcast from 0
    int redn_rank;
    int true_rank = GetLocalRankWithComm(communicator);
    int size = GetSizeWithComm(communicator);
    
    int buffer_len = entry.tensor->size();
    int root_node_rotation = false ? (layerid % size) : 0;
    redn_rank = (true_rank ^ root_node_rotation);
    int level;
    for (level = 1; level < size; level *= 2);
    level /= 2; // this make sure that level < size
  
    for(; level > 0; level /= 2) {
      int neighbor_redn_rank = redn_rank ^ level;
      int neighbor_true_rank = (neighbor_redn_rank ^ root_node_rotation);
      if (redn_rank % level != 0)
        continue;
      if (neighbor_redn_rank >= size)
        continue;
      if ((redn_rank & level) == 0) {
        // send grad_buffer to neighbor
        // and dont wait for the send to finish
        this->PointToPointSend(grad_buffer, buffer_len, entry.tensor->dtype(), neighbor_true_rank, layerid, communicator);
      }
      else {
        // recv grad_buffer from neighbor
        this->PointToPointRecv(grad_buffer, buffer_len, entry.tensor->dtype(), neighbor_true_rank, layerid, communicator);
      }
    }
  }

  template<typename T>
  void SyncAllreduce(T* grad_buffer,
                     T* recv_buffer, 
                     Communicator_type communicator,
                     Communicator_type* reduction_comms, 
                     int layerid, 
                     TensorTableEntry entry) {
    int rank = GetLocalRankWithComm(communicator);
    int size = GetSizeWithComm(communicator);
    int per_element_size = GetPerElementSize(entry);
    int count = entry.tensor->size() / per_element_size;
    DataType horovod_datatype = entry.tensor->dtype();
    //MPI_Allreduce((float*) grad_buffer, (float*) recv_buffer, count/2, MPI_FLOAT, MPI_SUM, communicator);

    //return;
    if (IsPowerOfTwo(size) == false) {
      throw std::logic_error("BUGBUG: need to implement logic for non power of two ranks");
    }
    
    //int chunk_size = (1<<15);
    int chunk_size = (1<<29);
    int nearest_power_2 = 1;
    for (nearest_power_2 = 1; (nearest_power_2<<1) <= size; nearest_power_2 = (nearest_power_2 << 1)){}
    int remaining_non_power_2 = size - nearest_power_2;
    int level;
    if (rank >= size - 2 * remaining_non_power_2){
        int myCount;
        int nghrCount;
        level = 0;
        int neighbor_rank;
        int sendOffset;
        int recvOffset;
        if (rank < nearest_power_2){
            neighbor_rank = rank + remaining_non_power_2;
            myCount = (count >> 1);
            nghrCount = count - myCount;
            sendOffset = myCount;
            recvOffset = 0;
        } else {
            nghrCount = (count >> 1);
            myCount = count - nghrCount;
            neighbor_rank = rank - remaining_non_power_2;
            sendOffset = 0;
            recvOffset = nghrCount;
        }
        for (int i = 0; i < std::max(nghrCount, myCount); i += chunk_size) {
            this->PointToPointSendRecv((char*)(&grad_buffer[i+sendOffset]),
                                 std::min(chunk_size, nghrCount-i) * per_element_size / sizeof(char),
                                 horovod_datatype,
                                 neighbor_rank,
                                 level * 1000 + layerid,
                                 (char*)(&recv_buffer[i+recvOffset]),
                                 std::min(chunk_size, myCount-i) * per_element_size / sizeof(char),
                                 horovod_datatype,
                                 neighbor_rank,
                                 level * 1000 + layerid,
                                 communicator);
        }
        DispatchScaledAdd(horovod_datatype, myCount, 1.0, &grad_buffer[recvOffset] , 1.0, &recv_buffer[recvOffset], AdasumOp<Communicator_type>::global_state_, layerid);

        if (rank < nearest_power_2) {
            for (int i = 0; i < nghrCount; i += chunk_size) {
                this->PointToPointRecv((char*)(&grad_buffer[i+sendOffset]),
                                 std::min(chunk_size, nghrCount-i) * per_element_size / sizeof(char),
                                 horovod_datatype,
                                 neighbor_rank,
                                 level * 1000 + layerid,
                                 communicator);
            }
        } else {
            for (int i = 0; i < myCount; i += chunk_size)
                this->PointToPointSend((char*)(&grad_buffer[i+recvOffset]),
                                 std::min(chunk_size, myCount-i) * per_element_size / sizeof(char),
                                 horovod_datatype,
                                 neighbor_rank,
                                 level * 1000 + layerid,
                                 communicator);
        }
    }

    int orgSize = size;
    size = nearest_power_2;
    if (rank < nearest_power_2){
        int myCount = count;
        int comm_index;
        for (level = 1, comm_index = 0; level < size; level = (level << 1), comm_index++){
            int neighbor_rank = rank ^ level;
            int nghrCount = 0;
            int sendOffset = 0;
            int recvOffset = 0;
            int firstHalfMyCount = (myCount >> 1);
            int secondHalfMyCount = myCount - firstHalfMyCount;
            if ((rank & level) != 0) {
                myCount = secondHalfMyCount;
                nghrCount = firstHalfMyCount;
                sendOffset = 0;
                recvOffset = nghrCount;
            } else {
                myCount = firstHalfMyCount;
                nghrCount = secondHalfMyCount;
                sendOffset = myCount;
                recvOffset = 0;
            }
            for (int i = 0; i < std::max(myCount,nghrCount); i += chunk_size)
                this->PointToPointSendRecv((char*)(&grad_buffer[i+sendOffset]),
                                     std::min(chunk_size, nghrCount-i) * per_element_size / sizeof(char),
                                     horovod_datatype,
                                     neighbor_rank,
                                     level * 1000 + layerid,
                                     (char*)(&recv_buffer[i+recvOffset]),
                                     std::min(chunk_size, myCount-i) * per_element_size / sizeof(char),
                                     horovod_datatype,
                                     neighbor_rank,
                                     level * 1000 + layerid,
                                     communicator);

            if ((rank & level) != 0) {
                grad_buffer = &grad_buffer[nghrCount];
                recv_buffer = &recv_buffer[nghrCount];
            }
            this->PairwiseReduceWithComm(grad_buffer, recv_buffer, myCount, layerid, horovod_datatype, reduction_comms[comm_index], (rank & level) == 0);        
        }

            for (level = (size >> 1); level > 0; level = (level >> 1)) {
                int neighbor_rank = rank ^ level;
                int nghrCount = myCount;
                int levelNP = (level << 1);
                int levelSizeDeterminer = levelNP - 1;
                int countRemainer = (count & levelSizeDeterminer);
                int myLevelRank = (rank & levelSizeDeterminer);
                int nghrLevelRank = (neighbor_rank & levelSizeDeterminer);
                if ((myLevelRank >= (levelNP - countRemainer)) && (nghrLevelRank < (levelNP - countRemainer))){
                    nghrCount -= 1;
                } else if ((myLevelRank < (levelNP - countRemainer)) && (nghrLevelRank >= (levelNP - countRemainer))){
                    nghrCount += 1;
                }

                if ((rank & level) == 0) {
                    recv_buffer = &grad_buffer[myCount];
                } else {
                    recv_buffer = &grad_buffer[-nghrCount];
                }
                for (int i = 0; i < std::max(myCount,nghrCount); i += chunk_size)
                    this->PointToPointSendRecv((char*)(&grad_buffer[i]),
                             std::min(chunk_size, myCount-i) * per_element_size / sizeof(char),
                             horovod_datatype,
                             neighbor_rank,
                             level * 1000 + layerid,
                             (char*)(&recv_buffer[i]),
                             std::min(chunk_size, nghrCount-i) * per_element_size / sizeof(char),
                             horovod_datatype,
                             neighbor_rank,
                             level * 1000 + layerid,
                             communicator);
                if ((rank & level) != 0) {
                    grad_buffer = &grad_buffer[-nghrCount];
                }
                myCount += nghrCount;
            }
    }
    size = orgSize;

    if (rank >= size - 2 * remaining_non_power_2){
        level = 0;
        int neighbor_rank;
        if (rank < nearest_power_2) {
            neighbor_rank = rank + remaining_non_power_2;
            for (int i = 0; i < count; i += chunk_size) {
                this->PointToPointSend((char*)(&grad_buffer[i]),
                std::min(chunk_size, count-i) * per_element_size / sizeof(char),
                horovod_datatype,
                neighbor_rank,
                level * 1000 + layerid,
                communicator);
            }
        } else {
            neighbor_rank = rank - remaining_non_power_2;
            for (int i = 0; i < count; i += chunk_size)
                this->PointToPointRecv((char*)(&grad_buffer[i]),
                std::min(chunk_size, count-i) * per_element_size / sizeof(char),
                horovod_datatype,
                neighbor_rank,
                level * 1000 + layerid,
                communicator);
        }
    }

  }

  void PairwiseReduceWithComm(void* a, void* b, int count, int layerid, DataType horovod_datatype, Communicator_type& comm, bool isLeftNeighbor){
    double dotProduct = 0.;
    double anormsq = 0.;
    double bnormsq = 0.;
    
    DispatchComputeDotAndNormSqrds(a, b, horovod_datatype, count, dotProduct, anormsq, bnormsq, this->global_state_, layerid);

    double reduce_vals[3], temp_buffer[3];
    if (isLeftNeighbor) { 
        reduce_vals[0] = anormsq;
        reduce_vals[1] = bnormsq;
    } else {
        reduce_vals[1] = anormsq;
        reduce_vals[0] = bnormsq;
    }
    reduce_vals[2] = dotProduct;

    this->P2pAllreduce(reduce_vals, temp_buffer, sizeof(reduce_vals), DataType::HOROVOD_FLOAT64, comm, layerid);

    if (isLeftNeighbor) { 
        anormsq = reduce_vals[0];
        bnormsq = reduce_vals[1];
    } else {
        anormsq = reduce_vals[1];
        bnormsq = reduce_vals[0];
    }
    dotProduct = reduce_vals[2];

    double acoeff = 1;
    double bcoeff = 1;
    if (anormsq >= 1e-8f)
        acoeff = 1.0 - dotProduct / anormsq * 0.5;
    if (bnormsq >= 1e-8f)
        bcoeff = 1.0 - dotProduct / bnormsq * 0.5;

    DispatchScaledAdd(horovod_datatype, count, acoeff, (uint16_t*)a, bcoeff, (uint16_t*)b, this->global_state_, layerid);
  }

  void DispatchSyncAllreduce(void* gradient_buffer,
                      void* recv_buffer,
                      Communicator_type* node_comm,
                      Communicator_type* reduction_comm_pool,
                      int layerid,
                      TensorTableEntry entry) {
      switch(entry.tensor->dtype()) {
          case DataType::HOROVOD_FLOAT16:
            SyncAllreduce((uint16_t*)gradient_buffer, (uint16_t*)recv_buffer, *node_comm, reduction_comm_pool, layerid, entry);
            break;
          case DataType::HOROVOD_FLOAT32:
            SyncAllreduce((float*)gradient_buffer, (float*)recv_buffer, *node_comm, reduction_comm_pool, layerid, entry);
            break;
          case DataType::HOROVOD_FLOAT64:
            SyncAllreduce((double*)gradient_buffer, (double*)recv_buffer, *node_comm, reduction_comm_pool, layerid, entry);
            break;
          default:
            throw std::logic_error("Unsupported data type");
      }
  }

  // over-write ComputeDotAndNormSqrds for float16
  inline void static ComputeDotAndNormSqrdsfp16(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, int len, double& dotProduct, double& anormsq, double& bnormsq, HorovodGlobalState *global_state, int layerid) {
      int i;
      __m256d dotProductVec = _mm256_setzero_pd();
      __m256d anormVec = _mm256_setzero_pd();
      __m256d bnormVec = _mm256_setzero_pd();
      for (i = 0; i < len - 7; i += 8) {
          __m256 aVec = MmLoaduPh(&a[i]);
          __m256 bVec = MmLoaduPh(&b[i]);
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
          __m256 aVec = MmLoaduPhPartial(&a[i], len - i);
          __m256 bVec = MmLoaduPhPartial(&b[i], len - i);
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
  
      dotProduct = Mm256ReductionPd(dotProductVec);
      anormsq = Mm256ReductionPd(anormVec);
      bnormsq = Mm256ReductionPd(bnormVec);
  }

  inline void static ScaledAddfp16(int len, double acoeff, uint16_t* __restrict__ a, double bcoeff, uint16_t* __restrict__ b, HorovodGlobalState *global_state, int layerid) {
      int i;
      __m256 acoeffVec = _mm256_set1_ps((float)(acoeff));
      __m256 bcoeffVec = _mm256_set1_ps((float)bcoeff);
      for (i = 0; i < len - 7; i += 8) {
          __m256 aVec = MmLoaduPh(&a[i]);
          __m256 bVec = MmLoaduPh(&b[i]);
          aVec = _mm256_mul_ps(acoeffVec, aVec);
          MmStorePh(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec));
      }
      if (i < len) {
          __m256 aVec = MmLoaduPhPartial(&a[i], len - i);
          __m256 bVec = MmLoaduPhPartial(&b[i], len - i);
          aVec = _mm256_mul_ps(acoeffVec, aVec);
          MmStorePhPartial(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec), len - i);
      }
  }

  void virtual MemcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) {
    assert(dest != nullptr);
    assert(src != nullptr);
    memcpy(dest, src, buffer_len);
  }


private:

  // reduce 4xfloat64 into one double
  inline double static Mm256ReductionPd(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128
    
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar      
  }

  // load 8 float16s from a and return the __m256 register
  inline __m256 static MmLoaduPh(const uint16_t* a) {
      __m128i r = _mm_loadu_si128((__m128i*)(a));
      return _mm256_cvtph_ps(r);
  }

  // store 8 float16 from val into a 
  inline void static MmStorePh(uint16_t* a, __m256 val) {
      __m128i r = _mm256_cvtps_ph(val, 0);
      _mm_storeu_si128((__m128i*)a, r);
  }

  // load len (< 8) float16s from a, fill the rest with 0s, and return the __m256 register
  inline __m256 static MmLoaduPhPartial(const uint16_t* a, int len) {
      short e[8];
      std::memset(e, 0, sizeof(e));
      std::memcpy(e, a, std::min(len, 8) * sizeof(short));
      __m128i es = _mm_set_epi16(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
      return _mm256_cvtph_ps(es);
  }

  // store the first len (< 8) float16s from val and store into a
  inline void static MmStorePhPartial(uint16_t* a, __m256 val, int len) {
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

#endif // HOROVOD_ADASUM_OPERATIONS_H
