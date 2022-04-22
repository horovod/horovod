# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import subprocess
import sys

def is_mx_cuda():
    try:
        from mxnet import runtime
        features = runtime.Features()
        return features.is_enabled('CUDA')
    except Exception:
        if 'linux' in sys.platform:
            try:
                import mxnet as mx
                mx_libs = mx.libinfo.find_lib_path()
                for mx_lib in mx_libs:
                    output = subprocess.check_output(['readelf', '-d', mx_lib])
                    if 'cuda' in str(output):
                        return True
                return False
            except Exception:
                return False
    return False

def is_mx_mkldnn():
    return is_mx_dnn('MKLDNN')

def is_mx_onednn():
    return is_mx_dnn('ONEDNN')

def is_mx_dnn(dnn_flavour: str):
    """
    Detects if MXNet is build with given DNN flavour (MKLDNN or oneDNN) support.
    MXNET ≥ 2.0.0 uses oneDNN (renamed from MKLDNN), < 2.0.0 MKLDNN.
    """
    dnn_flavour_lower = dnn_flavour.lower()
    dnn_flavour = dnn_flavour.upper()
    try:
        from mxnet import runtime
        features = runtime.Features()
        return features.is_enabled(dnn_flavour)
    except Exception:
        msg = f'INFO: Cannot detect if {dnn_flavour} is enabled in MXNet. Please ' \
              f'set MXNET_USE_{dnn_flavour}=1 if {dnn_flavour} is ' \
              f'enabled in your MXNet build.'
        if 'linux' not in sys.platform:
            # MKLDNN / oneDNN is only enabled by default in MXNet Linux build. Return
            # False by default for non-linux build but still allow users to
            # enable it by using MXNET_USE_MKLDNN / MXNET_USE_ONEDNN env variable.
            print(msg, file=sys.stderr)
            return os.environ.get(f'MXNET_USE_{dnn_flavour}', '0') == '1'
        else:
            try:
                import mxnet as mx
                mx_libs = mx.libinfo.find_lib_path()
                for mx_lib in mx_libs:
                    output = subprocess.check_output(['readelf', '-d', mx_lib])
                    if dnn_flavour_lower in str(output):
                        return True
                return False
            except Exception:
                print(msg, file=sys.stderr)
            return os.environ.get(f'MXNET_USE_{dnn_flavour}', '0') == '1'

def get_nvcc_bin():
    cuda_home = os.environ.get('HOROVOD_CUDA_HOME', '/usr/local/cuda')
    cuda_nvcc = os.path.join(cuda_home, 'bin', 'nvcc')

    for nvcc_bin in ['nvcc', cuda_nvcc]:
        try:
            subprocess.check_output([nvcc_bin, '--version'])
            return nvcc_bin
        except Exception:
            pass

    raise RuntimeError('Cannot find `nvcc`. `nvcc` is required to build Horovod with GPU operations. '
                       'Make sure it is added to your path or in $HOROVOD_CUDA_HOME/bin.')

def get_nvcc_flags():
    default_flags = ['-O3', '-Xcompiler', '-fPIC']
    cc_list_env = os.environ.get('HOROVOD_BUILD_CUDA_CC_LIST')

    # Invoke nvcc and extract all supported compute capabilities for CUDA toolkit version
    nvcc_bin = get_nvcc_bin()
    full_cc_list = subprocess.check_output(f"{nvcc_bin} --help | "
                                           f"sed -n -e '/gpu-architecture <arch>/,/gpu-code <code>/ p' | "
                                           f"sed -n -e '/Allowed values/,/gpu-code <code>/ p' | "
                                           f"grep -i sm_ | "
                                           f"grep -Eo 'sm_[0-9]+' | "
                                           f"sed -e s/sm_//g | "
                                           f"sort -g -u | "
                                           f"tr '\n' ' '",
                                           shell=True).strip().split()
    full_cc_list = [int(i) for i in full_cc_list]

    # Build native kernels for specified compute capabilities
    cc_list = full_cc_list if cc_list_env is None else [int(x) for x in cc_list_env.split(',')]
    cc_list = sorted(cc_list)
    for cc in cc_list[:-1]:
        default_flags += ['-gencode', 'arch=compute_{cc},code=sm_{cc}'.format(cc=cc)]
    # Build PTX for maximum specified compute capability
    default_flags += ['-gencode', 'arch=compute_{cc},code=\\"sm_{cc},compute_{cc}\\"'.format(cc=cc_list[-1])]

    return default_flags
