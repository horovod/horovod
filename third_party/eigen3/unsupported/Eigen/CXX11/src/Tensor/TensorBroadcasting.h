// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_BROADCASTING_H
#define EIGEN_CXX11_TENSOR_TENSOR_BROADCASTING_H

namespace Eigen {

/** \class TensorBroadcasting
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor broadcasting class.
  *
  *
  */
namespace internal {
template<typename Broadcast, typename XprType>
struct traits<TensorBroadcastingOp<Broadcast, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions;
  static const int Layout = XprTraits::Layout;
};

template<typename Broadcast, typename XprType>
struct eval<TensorBroadcastingOp<Broadcast, XprType>, Eigen::Dense>
{
  typedef const TensorBroadcastingOp<Broadcast, XprType>& type;
};

template<typename Broadcast, typename XprType>
struct nested<TensorBroadcastingOp<Broadcast, XprType>, 1, typename eval<TensorBroadcastingOp<Broadcast, XprType> >::type>
{
  typedef TensorBroadcastingOp<Broadcast, XprType> type;
};

template <typename Dims>
struct is_input_scalar {
  static const bool value = false;
};
template <>
struct is_input_scalar<Sizes<> > {
  static const bool value = true;
};
#ifndef EIGEN_EMULATE_CXX11_META_H
template <typename std::size_t... Indices>
struct is_input_scalar<Sizes<Indices...> > {
  static const bool value = (Sizes<Indices...>::total_size == 1);
};
#endif

}  // end namespace internal



template<typename Broadcast, typename XprType>
class TensorBroadcastingOp : public TensorBase<TensorBroadcastingOp<Broadcast, XprType>, ReadOnlyAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorBroadcastingOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorBroadcastingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorBroadcastingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorBroadcastingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorBroadcastingOp(const XprType& expr, const Broadcast& broadcast)
      : m_xpr(expr), m_broadcast(broadcast) {}

    EIGEN_DEVICE_FUNC
    const Broadcast& broadcast() const { return m_broadcast; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

  protected:
    typename XprType::Nested m_xpr;
    const Broadcast m_broadcast;
};


// Eval as rvalue
template<typename Broadcast, typename ArgType, typename Device>
struct TensorEvaluator<const TensorBroadcastingOp<Broadcast, ArgType>, Device>
{
  typedef TensorBroadcastingOp<Broadcast, ArgType> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions InputDimensions;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;

  enum {
    IsAligned = true,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    RawAccess = false
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
    : m_broadcast(op.broadcast()),m_impl(op.expression(), device)
  {
    // The broadcasting op doesn't change the rank of the tensor. One can't broadcast a scalar
    // and store the result in a scalar. Instead one should reshape the scalar into a a N-D
    // tensor with N >= 1 of 1 element first and then broadcast.
    EIGEN_STATIC_ASSERT((NumDims > 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
    const InputDimensions& input_dims = m_impl.dimensions();
    const Broadcast& broadcast = op.broadcast();
    for (int i = 0; i < NumDims; ++i) {
      eigen_assert(input_dims[i] > 0);
      m_dimensions[i] = input_dims[i] * broadcast[i];
    }

    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_inputStrides[0] = 1;
      m_outputStrides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_inputStrides[i] = m_inputStrides[i-1] * input_dims[i-1];
        m_outputStrides[i] = m_outputStrides[i-1] * m_dimensions[i-1];
      }
    } else {
      m_inputStrides[NumDims-1] = 1;
      m_outputStrides[NumDims-1] = 1;
      for (int i = NumDims-2; i >= 0; --i) {
        m_inputStrides[i] = m_inputStrides[i+1] * input_dims[i+1];
        m_outputStrides[i] = m_outputStrides[i+1] * m_dimensions[i+1];
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* /*data*/) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE CoeffReturnType coeff(Index index) const
  {
    if (internal::is_input_scalar<typename internal::remove_all<InputDimensions>::type>::value) {
      return m_impl.coeff(0);
    }

    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return coeffColMajor(index);
    } else {
      return coeffRowMajor(index);
    }
  }

  // TODO: attempt to speed this up. The integer divisions and modulo are slow
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeffColMajor(Index index) const
  {
    Index inputIndex = 0;
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = index / m_outputStrides[i];
      if (internal::index_statically_eq<Broadcast>(i, 1)) {
        eigen_assert(idx < m_impl.dimensions()[i]);
        inputIndex += idx * m_inputStrides[i];
      } else {
        if (internal::index_statically_eq<InputDimensions>(i, 1)) {
          eigen_assert(idx % m_impl.dimensions()[i] == 0);
        } else {
          inputIndex += (idx % m_impl.dimensions()[i]) * m_inputStrides[i];
        }
      }
      index -= idx * m_outputStrides[i];
    }
    if (internal::index_statically_eq<Broadcast>(0, 1)) {
      eigen_assert(index < m_impl.dimensions()[0]);
      inputIndex += index;
    } else {
      if (internal::index_statically_eq<InputDimensions>(0, 1)) {
        eigen_assert(index % m_impl.dimensions()[0] == 0);
      } else {
        inputIndex += (index % m_impl.dimensions()[0]);
      }
    }
    return m_impl.coeff(inputIndex);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeffRowMajor(Index index) const
  {
    Index inputIndex = 0;
    for (int i = 0; i < NumDims - 1; ++i) {
      const Index idx = index / m_outputStrides[i];
      if (internal::index_statically_eq<Broadcast>(i, 1)) {
        eigen_assert(idx < m_impl.dimensions()[i]);
        inputIndex += idx * m_inputStrides[i];
      } else {
        if (internal::index_statically_eq<InputDimensions>(i, 1)) {
          eigen_assert(idx % m_impl.dimensions()[i] == 0);
        } else {
          inputIndex += (idx % m_impl.dimensions()[i]) * m_inputStrides[i];
        }
      }
      index -= idx * m_outputStrides[i];
    }
    if (internal::index_statically_eq<Broadcast>(NumDims-1, 1)) {
      eigen_assert(index < m_impl.dimensions()[NumDims-1]);
      inputIndex += index;
    } else {
      if (internal::index_statically_eq<InputDimensions>(NumDims-1, 1)) {
        eigen_assert(index % m_impl.dimensions()[NumDims-1] == 0);
      } else {
        inputIndex += (index % m_impl.dimensions()[NumDims-1]);
      }
    }
    return m_impl.coeff(inputIndex);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketReturnType packet(Index index) const
  {
    if (internal::is_input_scalar<typename internal::remove_all<InputDimensions>::type>::value) {
      return internal::pset1<PacketReturnType>(m_impl.coeff(0));
    }

    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return packetColMajor<LoadMode>(index);
    } else {
      return packetRowMajor<LoadMode>(index);
    }
  }

  // Ignore the LoadMode and always use unaligned loads since we can't guarantee
  // the alignment at compile time.
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packetColMajor(Index index) const
  {
    EIGEN_STATIC_ASSERT((PacketSize > 1), YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+PacketSize-1 < dimensions().TotalSize());

    const Index originalIndex = index;

    Index inputIndex = 0;
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = index / m_outputStrides[i];
      if (internal::index_statically_eq<Broadcast>(i, 1)) {
        eigen_assert(idx < m_impl.dimensions()[i]);
        inputIndex += idx * m_inputStrides[i];
      } else {
        if (internal::index_statically_eq<InputDimensions>(i, 1)) {
          eigen_assert(idx % m_impl.dimensions()[i] == 0);
        } else {
          inputIndex += (idx % m_impl.dimensions()[i]) * m_inputStrides[i];
        }
      }
      index -= idx * m_outputStrides[i];
    }
    Index innermostLoc;
    if (internal::index_statically_eq<Broadcast>(0, 1)) {
      eigen_assert(index < m_impl.dimensions()[0]);
      innermostLoc = index;
    } else {
      if (internal::index_statically_eq<InputDimensions>(0, 1)) {
        eigen_assert(index % m_impl.dimensions()[0] == 0);
        innermostLoc = 0;
      } else {
        innermostLoc = index % m_impl.dimensions()[0];
      }
    }
    inputIndex += innermostLoc;

    // Todo: this could be extended to the second dimension if we're not
    // broadcasting alongside the first dimension, and so on.
    if (innermostLoc + PacketSize <= m_impl.dimensions()[0]) {
      return m_impl.template packet<Unaligned>(inputIndex);
    } else {
      EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type values[PacketSize];
      values[0] = m_impl.coeff(inputIndex);
      for (int i = 1; i < PacketSize; ++i) {
        values[i] = coeffColMajor(originalIndex+i);
      }
      PacketReturnType rslt = internal::pload<PacketReturnType>(values);
      return rslt;
    }
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packetRowMajor(Index index) const
  {
    EIGEN_STATIC_ASSERT((PacketSize > 1), YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+PacketSize-1 < dimensions().TotalSize());

    const Index originalIndex = index;

    Index inputIndex = 0;
    for (int i = 0; i < NumDims - 1; ++i) {
      const Index idx = index / m_outputStrides[i];
      if (internal::index_statically_eq<Broadcast>(i, 1)) {
        eigen_assert(idx < m_impl.dimensions()[i]);
        inputIndex += idx * m_inputStrides[i];
      } else {
        if (internal::index_statically_eq<InputDimensions>(i, 1)) {
          eigen_assert(idx % m_impl.dimensions()[i] == 0);
        } else {
          inputIndex += (idx % m_impl.dimensions()[i]) * m_inputStrides[i];
        }
      }
      index -= idx * m_outputStrides[i];
    }
    Index innermostLoc;
    if (internal::index_statically_eq<Broadcast>(NumDims-1, 1)) {
      eigen_assert(index < m_impl.dimensions()[NumDims-1]);
      innermostLoc = index;
    } else {
      if (internal::index_statically_eq<InputDimensions>(NumDims-1, 1)) {
        eigen_assert(index % m_impl.dimensions()[NumDims-1] == 0);
        innermostLoc = 0;
      } else {
        innermostLoc = index % m_impl.dimensions()[NumDims-1];
      }
    }
    inputIndex += innermostLoc;

    // Todo: this could be extended to the second dimension if we're not
    // broadcasting alongside the first dimension, and so on.
    if (innermostLoc + PacketSize <= m_impl.dimensions()[NumDims-1]) {
      return m_impl.template packet<Unaligned>(inputIndex);
    } else {
      EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type values[PacketSize];
      values[0] = m_impl.coeff(inputIndex);
      for (int i = 1; i < PacketSize; ++i) {
        values[i] = coeffRowMajor(originalIndex+i);
      }
      PacketReturnType rslt = internal::pload<PacketReturnType>(values);
      return rslt;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost
  costPerCoeff(bool vectorized) const {
    double compute_cost = TensorOpCost::AddCost<Index>();
    if (NumDims > 0) {
      for (int i = NumDims - 1; i > 0; --i) {
        compute_cost += TensorOpCost::DivCost<Index>();
        if (internal::index_statically_eq<Broadcast>(i, 1)) {
          compute_cost +=
              TensorOpCost::MulCost<Index>() + TensorOpCost::AddCost<Index>();
        } else {
          if (!internal::index_statically_eq<InputDimensions>(i, 1)) {
            compute_cost += TensorOpCost::MulCost<Index>() +
                            TensorOpCost::ModCost<Index>() +
                            TensorOpCost::AddCost<Index>();
          }
        }
        compute_cost +=
            TensorOpCost::MulCost<Index>() + TensorOpCost::AddCost<Index>();
      }
    }
    return m_impl.costPerCoeff(vectorized) +
           TensorOpCost(0, 0, compute_cost, vectorized, PacketSize);
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return NULL; }

  const TensorEvaluator<ArgType, Device>& impl() const { return m_impl; }

  Broadcast functor() const { return m_broadcast; }

 protected:
  const Broadcast m_broadcast;
  Dimensions m_dimensions;
  array<Index, NumDims> m_outputStrides;
  array<Index, NumDims> m_inputStrides;
  TensorEvaluator<ArgType, Device> m_impl;
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_BROADCASTING_H
