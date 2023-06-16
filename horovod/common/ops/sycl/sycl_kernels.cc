// Copyright (C) 2023 Intel CORPORATION. All rights reserved.
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

#include "sycl_kernels.h"

namespace horovod {
namespace common {

template <typename T, typename TS> class ScaleBufferSYCLKernelImpl;

template <typename TL, typename T, typename TS>
void BatchedScaledMemcpyDatatype(size_t idx, const T* input, T* output,
                                 size_t size, const TS scale_factor,
                                 int groups_num) {
  const int64_t num_words = size / sizeof(TL);
  const TL* read_ptr = reinterpret_cast<const TL*>(input);
  TL* write_ptr = reinterpret_cast<TL*>(output);
  for (size_t i = idx; i < num_words; i += groups_num) {
    // Load word
    TL word = read_ptr[i];
    T* val = reinterpret_cast<T*>(&word);

    // Scale elements in word
    for (int j = 0; j < sizeof(TL) / sizeof(T); ++j) {
      val[j] *= scale_factor;
    }

    // Write word
    write_ptr[i] = word;
  }

  // Deal with any remaining elements
  size_t remainder = (size % sizeof(TL)) / sizeof(T);
  if (remainder > 0 && idx < remainder) {
    const T* input_r = reinterpret_cast<const T*>(read_ptr + num_words);
    T* output_r = reinterpret_cast<T*>(write_ptr + num_words);
    output_r[idx] = scale_factor * input_r[idx];
  }
}

template <typename T>
void BatchedMemcpyDatatype(size_t idx, void* in, void* out, size_t size,
                           int groups_num) {
  const T* input = reinterpret_cast<const T*>(in);
  T* output = reinterpret_cast<T*>(out);
  const size_t num_elements = size / sizeof(T);

  for (size_t i = idx; i < num_elements; i += groups_num) {
    output[i] = input[i];
  }

  // Deal with any remaining bytes
  size_t remainder = size % sizeof(T);
  if (remainder > 0 && idx < remainder) {
    const unsigned char* input_r =
        reinterpret_cast<const unsigned char*>(input + num_elements);
    unsigned char* output_r =
        reinterpret_cast<unsigned char*>(output + num_elements);
    output_r[idx] = input_r[idx];
  }
}

template <typename T, typename TS> struct BatchedScaledMemcpySYCLKernel {
  BatchedScaledMemcpySYCLKernel(BatchedD2DParams& params, TS scale_factor,
                                int groups_per_copy)
      : params_(params), scale_factor_(scale_factor),
        groups_per_copy_(groups_per_copy) {}
  void operator()(sycl::nd_item<1> item) const {
    size_t local_id = item.get_local_id(0);
    size_t group_size = item.get_local_range(0);
    size_t group_id = item.get_group(0);

    const size_t idx = group_size * (group_id % groups_per_copy_) + local_id;
    int cur_index = group_id / groups_per_copy_;
    int groups_num = group_size * groups_per_copy_;

    T* output = reinterpret_cast<T*>(params_.out[cur_index]);
    const T* input = reinterpret_cast<const T*>(params_.in[cur_index]);
    const size_t size = params_.sizes[cur_index];

    size_t align_in = reinterpret_cast<size_t>(input) % BATCHED_D2D_PADDING;
    size_t align_out = reinterpret_cast<size_t>(output) % BATCHED_D2D_PADDING;

    // Select load/store size based on the misaligned buffer
    size_t align = (align_out == 0) ? align_in : align_out;
    if (align_in && align_out) {
      // If both are misaligned, use datatype size
      align = sizeof(T);
    }

    if (align % 16 == 0) {
      BatchedScaledMemcpyDatatype<sycl::ulonglong2>(idx, input, output, size,
                                                    scale_factor_, groups_num);
    } else if (align % 8 == 0) {
      BatchedScaledMemcpyDatatype<unsigned long long>(
          idx, input, output, size, scale_factor_, groups_num);
    } else if (align % 4 == 0) {
      BatchedScaledMemcpyDatatype<unsigned int>(idx, input, output, size,
                                                scale_factor_, groups_num);
    } else if (align % 2 == 0) {
      BatchedScaledMemcpyDatatype<unsigned short>(idx, input, output, size,
                                                  scale_factor_, groups_num);
    } else {
      BatchedScaledMemcpyDatatype<unsigned char>(idx, input, output, size,
                                                 scale_factor_, groups_num);
    }
  }

private:
  BatchedD2DParams params_;
  TS scale_factor_;
  int groups_per_copy_;
};

template <typename T> struct BatchedMemcpySYCLKernel {
  BatchedMemcpySYCLKernel(BatchedD2DParams& params, int groups_per_copy)
      : params_(params), groups_per_copy_(groups_per_copy) {}
  void operator()(sycl::nd_item<1> item) const {
    size_t local_id = item.get_local_id(0);
    size_t group_size = item.get_local_range(0);
    size_t group_id = item.get_group(0);

    const size_t idx = group_size * (group_id % groups_per_copy_) + local_id;
    int cur_index = group_id / groups_per_copy_;
    int groups_num = group_size * groups_per_copy_;

    void* output = params_.out[cur_index];
    void* input = params_.in[cur_index];
    const size_t size = params_.sizes[cur_index];

    size_t align_in = reinterpret_cast<size_t>(input) % BATCHED_D2D_PADDING;
    size_t align_out = reinterpret_cast<size_t>(output) % BATCHED_D2D_PADDING;

    // Select load/store size based on the misaligned buffer
    size_t align = (align_out == 0) ? align_in : align_out;
    if (align_in && align_out) {
      // If both are misaligned, use unsigned char (this should not occur for
      // Allreduces as fusion buffer locations should be aligned by applying
      // BATCHED_D2D_PADDING during construction.)
      align = 1;
    }

    if (align % 16 == 0) {
      BatchedMemcpyDatatype<sycl::ulonglong2>(idx, input, output, size,
                                              groups_num);
    } else if (align % 8 == 0) {
      BatchedMemcpyDatatype<unsigned long long>(idx, input, output, size,
                                                groups_num);
    } else if (align % 4 == 0) {
      BatchedMemcpyDatatype<unsigned int>(idx, input, output, size, groups_num);
    } else if (align % 2 == 0) {
      BatchedMemcpyDatatype<unsigned short>(idx, input, output, size,
                                            groups_num);
    } else {
      BatchedMemcpyDatatype<unsigned char>(idx, input, output, size,
                                           groups_num);
    }
  }

private:
  BatchedD2DParams params_;
  int groups_per_copy_;
};

#define GROUPS_PER_COPY_D2D_KERNEL 8
template <typename T, typename TS>
void BatchedScaledD2DMemcpy(BatchedD2DParams params, int num_copies,
                            TS scale_factor, gpuStream_t stream) {
  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();
  int32_t num_workitems =
      max_group_size * num_copies * GROUPS_PER_COPY_D2D_KERNEL;

  stream->submit([&](sycl::handler& cgh) {
    BatchedScaledMemcpySYCLKernel<T, TS> task(params, scale_factor,
                                              GROUPS_PER_COPY_D2D_KERNEL);
    cgh.parallel_for<BatchedScaledMemcpySYCLKernel<T, TS>>(
        sycl::nd_range<1>(num_workitems, max_group_size), task);
  });
}

void BatchedScaledD2DMemcpySYCLImpl(BatchedD2DParams& params, int num_copies,
                                    double scale_factor, DataType dtype,
                                    gpuStream_t stream) {
  float float_scale_factor = (float)scale_factor;
  switch (dtype) {
  case HOROVOD_UINT8:
    BatchedScaledD2DMemcpy<uint8_t, float>(params, num_copies,
                                           float_scale_factor, stream);
    break;
  case HOROVOD_INT8:
    BatchedScaledD2DMemcpy<int8_t, float>(params, num_copies,
                                          float_scale_factor, stream);
    break;
  case HOROVOD_INT32:
    BatchedScaledD2DMemcpy<int32_t, float>(params, num_copies,
                                           float_scale_factor, stream);
    break;
  case HOROVOD_INT64:
    BatchedScaledD2DMemcpy<int64_t, float>(params, num_copies,
                                           float_scale_factor, stream);
    break;
  case HOROVOD_FLOAT16:
    BatchedScaledD2DMemcpy<sycl::half, sycl::half>(
        params, num_copies, float_scale_factor, stream);
    break;
  case HOROVOD_FLOAT32:
    BatchedScaledD2DMemcpy<float, float>(params, num_copies, float_scale_factor,
                                         stream);
    break;
  case HOROVOD_FLOAT64:
    BatchedScaledD2DMemcpy<double, double>(params, num_copies, scale_factor,
                                           stream);
    break;
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " not supported by BatchedScaledD2DMemcpySYCLImpl.");
  }
}

template <typename T, typename TS>
void ScaleBufferSYCLKernel(const T* input, T* output, int64_t num_elements,
                           TS scale_factor, gpuStream_t stream) {
  const int wg_size =
      stream->get_device().get_info<sycl::info::device::max_work_group_size>();
  const int num_workgroups = (num_elements + wg_size - 1) / wg_size;
  stream->submit([&](sycl::handler& h) {
    sycl::range<1> global(num_workgroups * wg_size);
    sycl::range<1> local(wg_size);
    h.parallel_for<ScaleBufferSYCLKernelImpl<T, TS>>(
        sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
          auto id = it.get_global_linear_id();
          if (id >= num_elements)
            return;
          output[id] = scale_factor * input[id];
        });
  });
}

void ScaleBufferSYCLImpl(const void* fused_input_data, void* buffer_data,
                         int64_t num_elements, double scale_factor,
                         DataType dtype, gpuStream_t stream) {
  auto float_scale_factor = (float)scale_factor;
  switch (dtype) {
  case HOROVOD_UINT8:
    ScaleBufferSYCLKernel((const uint8_t*)fused_input_data,
                          (uint8_t*)buffer_data, num_elements,
                          float_scale_factor, stream);
    break;
  case HOROVOD_INT8:
    ScaleBufferSYCLKernel((const int8_t*)fused_input_data, (int8_t*)buffer_data,
                          num_elements, float_scale_factor, stream);
    break;
  case HOROVOD_INT32:
    ScaleBufferSYCLKernel((const int32_t*)fused_input_data,
                          (int32_t*)buffer_data, num_elements,
                          float_scale_factor, stream);
    break;
  case HOROVOD_INT64:
    ScaleBufferSYCLKernel((const int64_t*)fused_input_data,
                          (int64_t*)buffer_data, num_elements,
                          float_scale_factor, stream);
    break;
  case HOROVOD_FLOAT16:
    ScaleBufferSYCLKernel((const sycl::half*)fused_input_data,
                          (sycl::half*)buffer_data, num_elements,
                          (sycl::half)float_scale_factor, stream);
    break;
  case HOROVOD_FLOAT32:
    ScaleBufferSYCLKernel((const float*)fused_input_data, (float*)buffer_data,
                          num_elements, float_scale_factor, stream);
    break;
  case HOROVOD_FLOAT64:
    ScaleBufferSYCLKernel((const double*)fused_input_data, (double*)buffer_data,
                          num_elements, scale_factor, stream);
    break;
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " not supported by ScaleBufferSyclImpl.");
  }
}

void BatchedD2DMemcpySYCLImpl(BatchedD2DParams& params, int num_copies,
                              gpuStream_t stream) {
  int max_group_size =
      (stream->get_device())
          .template get_info<sycl::info::device::max_work_group_size>();
  int32_t num_workitems =
      max_group_size * num_copies * GROUPS_PER_COPY_D2D_KERNEL;

  stream->submit([&](sycl::handler& cgh) {
    BatchedMemcpySYCLKernel<unsigned char> task(params,
                                                GROUPS_PER_COPY_D2D_KERNEL);
    cgh.parallel_for<BatchedMemcpySYCLKernel<unsigned char>>(
        sycl::nd_range<1>(num_workitems, max_group_size), task);
  });
}
} // namespace common
} // namespace horovod
