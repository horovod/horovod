load("@pip_deps//:requirements.bzl", "requirement")
load("@pip_deps_tf21//:requirements.bzl", requirement21 = "requirement")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

cc_library(
    name = "common_ops_cuda_headers",
    srcs = [
        "common/ops/cuda/cuda_kernels.h",
    ]
)

cc_library(
    name="common_ops_cuda",
    srcs=[
        "common/ops/cuda/cuda_kernels.h",
        "common/ops/cuda/cuda_kernels.cu"
    ],
    hdrs=[
        "common/ops/cuda/cuda_kernels.h",
        "common/message.h",
        "common/group_table.h"
    ],
    deps=[
        "@local_config_cuda//cuda:cuda_headers",
	"@local_config_cuda//cuda"
    ],
    alwayslink = True,
    copts = [
        "-pthread", 
        "-DTENSORFLOW_USE_NVCC=1", 
        "-DGOOGLE_CUDA=1", 
        "-x cuda",
        "--cuda-gpu-arch=sm_62",
        "--cuda-gpu-arch=sm_70",
        "--cuda-gpu-arch=sm_80",
        "-Ilocal_config_cuda/cuda/include",
	"--cuda-path=external/local_config_cuda/cuda",
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "common_mpi",
    srcs = glob(["common/mpi/mpi_*.cc"]),
    hdrs = glob(
        include = [
            "common/mpi/*.h",
            "common/*.h",
            "common/ops/*.h",
            "common/optim/*.h",
            "common/utils/*.h"
        ],
        exclude = [
            "common/ops/ddl_operations.h",
            "common/mpi/ddl_mpi_context_manager.h"
        ]
    ),
    copts = [
        "-Iexternal/mpi",
        "-DHAVE_MPI=true",
        "-DHAVE_NCCL=true",
    ],
    deps = [
        "@eigen_archive//:eigen3",
        "@boost//:fiber",
        "@boost//:lockfree",
        "@mpi//:mpi"
    ],
    alwayslink = True
)

cc_library(
    name = "common_ops_optim",
    srcs = [
        "common/optim/bayesian_optimization.cc",
        "common/optim/gaussian_process.cc",
    ],
    hdrs = [
        "common/optim/bayesian_optimization.h",
        "common/optim/gaussian_process.h",
    ],
    deps = [
        "@eigen_archive//:eigen3",
        "//third_party/lbfgs",
    ]
)

cc_library(
    name = "common_ops_utils",
    srcs = ["common/utils/env_parser.cc"],
    hdrs = [
        "common/utils/env_parser.h",
        "common/stall_inspector.h",
        "common/response_cache.h",
        "common/common.h",
        "common/message.h",
        "common/group_table.h",
        "common/operations.h",
        "common/nvtx_op_range.h",
        "common/half.h",
        "common/logging.h",
        "common/controller.h",
        "common/global_state.h",
        "common/fusion_buffer_manager.h",
        "common/hashes.h",
        "common/parameter_manager.h",
        "common/process_set.h",
        "common/tensor_queue.h",
        "common/timeline.h",
    ],
    deps = [
        "@mpi//:mpi"
    ],
    local_defines = ["HAVE_MPI", "HAVE_NCCL"]
)

cc_library(
    name = "common_ops_adasum",
    srcs = glob(["common/ops/adasum/*.cc"]),
    hdrs = glob([
        "common/ops/adasum/*.h",
        "common/mpi/mpi_context.h",
        "common/common.h",
        "common/message.h",
        "common/group_table.h",
        "common/nvtx_op_range.h",
        "common/half.h",
        "common/logging.h",
        "common/controller.h",
        "common/global_state.h",
        "common/fusion_buffer_manager.h",
        "common/hashes.h",
        "common/parameter_manager.h",
        "common/process_set.h",
        "common/response_cache.h",
        "common/tensor_queue.h",
        "common/timeline.h",
    ]),
    deps = [
        "@mpi//:mpi",
        "@eigen_archive//:eigen3",
        "@boost//:fiber",
        "@boost//:lockfree",
        ":common_ops_optim",
        ":common_ops_utils",
    ],
    local_defines = [],
    alwayslink = True
)

cc_library(
    name = "common_ops",
    srcs = glob(
        include = ["common/ops/*.cc"],
        exclude = [
            "common/ops/ddl_operations.cc",
            "common/ops/hip_operations.cc",
            "common/ops/ccl_operations.cc",
            "common/ops/gpu_context_impl.cc",
            "common/ops/gloo_operations.cc",
        ]
    ),
    hdrs = glob(
        include = [
            "common/ops/*.h",
            "common/hashes.h",
            "mpi/mpi_context.h",
            "common/thread_pool.h",
            "common/ops/gpu_context_impl.cc",
        ],
        exclude = [
            "common/ops/ddl_operations.h",
            "common/ops/hip_operations.h",
            "common/ops/ccl_operations.h",
            "common/ops/gloo_operations.h",
        ]
    ),
    deps = [
        ":common_ops_adasum",
        ":common_ops_cuda_headers",
        "@mpi//:mpi",
        "@eigen_archive//:eigen3",
        "@local_config_nccl//:nccl",
    ],
    copts = ["-Inccl/src"],
    local_defines = [
        "HAVE_MPI",
        "HAVE_CUDA",
        "HAVE_GPU",
        "HAVE_NCCL",
        "HOROVOD_GPU_ALLREDUCE=N"
    ],
    alwayslink = True
)

cc_binary(
    name = "tf_mpi_ops.so",
    srcs = [
        "tensorflow/mpi_ops.cc",
        "tensorflow/xla_mpi_ops.cc",
        "common/common.h",
        "common/message.h",
        "common/group_table.h",
        "common/nvtx_op_range.h",
        "common/operations.h",
    ],
    copts = ["-DTENSORFLOW_VERSION=2009000000"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/kernels:resource_variable_ops",
    ],
    linkshared = 1,
)

tf_custom_op_library(
    name = "tf_mpi_ops",
    srcs = [
        "tensorflow/mpi_ops.cc",
        "tensorflow/xla_mpi_ops.cc",
        "common/common.h",
        "common/message.h",
        "common/group_table.h",
        "common/nvtx_op_range.h",
        "common/operations.h",
    ], 
    copts = ["-DTENSORFLOW_VERSION=2009000000"],
    deps = [
    ],
)

cc_library(
    name = "common",
    srcs = glob(["common/*.cc"]),
    hdrs = glob([
        "common/**/*.h",
        "common/*.h"
    ]),
    deps = [
        "@eigen_archive//:eigen3",
        "@boost//:fiber",
        "@boost//:lockfree",
        "//third_party/flatbuffers",
        "@mpi//:mpi",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_nccl//:nccl",
    ],
    copts = [
        "-Iexternal/eigen",
        "-Iexternal/mpi",
    ],
    local_defines = [
        "HAVE_MPI",
        "HAVE_CUDA",
        #"HAVE_NCCL",
        #"HAVE_GPU"
    ],
    alwayslink = True
)

tf_custom_op_library(
    name = 'tensorflow/mpi_lib.cpython-37m-x86_64-linux-gnu.so',
    srcs = [
        "tensorflow/mpi_ops.cc",
        #"tensorflow/xla_mpi_ops.cc",
        "common/common.h",
        "common/message.h",
        "common/group_table.h",
        "common/nvtx_op_range.h",
        "common/operations.h",
    ],
    copts = ["-DTENSORFLOW_VERSION=2006000000"],
    deps = [
        ":common_ops_cuda",
        ":common_mpi",
        ":common",
        ":common_ops",
        "@org_tensorflow//tensorflow/core/kernels:training_op_helpers_headers",
        #"@org_tensorflow//tensorflow/compiler/tf2xla:xla_compiler_headers_lib",
        #"@org_tensorflow//tensorflow/compiler/xla:xla_headers_lib",
        #"@org_tensorflow//tensorflow/compiler/xla/service:xla_service_headers_lib",
    ],
    local_defines = ["EXT_SUFFIX"]
)


py_library(
    name = "runner_common",
    srcs = glob(["runner/common/**/*.py"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "runner_driver",
    srcs = glob(["runner/driver/*.py"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "runner_elastic",
    srcs = glob(["runner/elastic/*.py"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "runner_http",
    srcs = glob(["runner/http/*.py"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "runner_task",
    srcs = glob(["runner/task/*.py"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "runner_util",
    srcs = glob(["runner/util/*.py"]),
    visibility = ["//visibility:public"],
)


py_library(
    name = "runner_py",
    srcs = glob(["runner/*.py"]),
    deps = [
        ":runner_common",
        ":runner_driver",
        ":runner_elastic",
        ":runner_http",
        ":runner_task",
        ":runner_util",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "common_py",
    srcs = glob(["common/*.py"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "tensorflow_keras",
    srcs = glob(["tensorflow/keras/*.py", "_keras/*.py"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "horovod_tf_py",
    srcs = [
        "tensorflow/compression.py",
        "tensorflow/elastic.py",
        "tensorflow/functions.py",
        "tensorflow/gradient_aggregation_eager.py",
        "tensorflow/gradient_aggregation.py",
        "tensorflow/mpi_ops.py",
        "tensorflow/sync_batch_norm.py",
        "tensorflow/util.py",
        "tensorflow/__init__.py",
    ],
    data = [
        ":tensorflow/mpi_lib.cpython-37m-x86_64-linux-gnu.so",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/python",
        ":common_py",
        ":runner_py",
        ":tensorflow_keras",
        requirement("psutil"),
        requirement("cloudpickle"),
        requirement("keras"),
        requirement("h5py"),
        requirement21("tensorflow-estimator"),
    ],
)
