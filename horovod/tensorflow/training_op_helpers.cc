/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/training_op_helpers.h"

#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {


void MaybeForwardRefInputToRefOutput(OpKernelContext* ctx, int input,
                                     int output) {
  if (ctx->input_dtype(input) != DT_RESOURCE) {
    ctx->forward_ref_input_to_ref_output(input, output);
  }
}

}  // end namespace tensorflow
