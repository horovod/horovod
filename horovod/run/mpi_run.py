# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import print_function
import six
import traceback
import sys
import os
from horovod.run.common.util import env as env_util, safe_shell_exec, secret, codec

# Open MPI Flags
_OMPI_FLAGS = ['-mca pml ob1', '-mca btl ^openib']
# Spectrum MPI Flags
_SMPI_FLAGS = ['-gpu', '-disable_gdr']
# MPICH Flags
_MPICH_FLAGS = []
# Threshold for large cluster MPI issues:
_LARGE_CLUSTER_THRESHOLD = 64

try:
    from shlex import quote
except ImportError:
    from pipes import quote


def _get_mpi_implementation_flags():
    output = six.StringIO()
    command = 'mpirun --version'
    try:
        exit_code = safe_shell_exec.execute(command, stdout=output,
                                            stderr=output)
        output_msg = output.getvalue()
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return None
    finally:
        output.close()

    if exit_code == 0:
        if 'Open MPI' in output_msg:
            return list(_OMPI_FLAGS)
        elif 'IBM Spectrum MPI' in output_msg:
            return list(_SMPI_FLAGS)
        elif 'MPICH' in output_msg:
            return list(_MPICH_FLAGS)
        print('Open MPI/Spectrum MPI/MPICH not found in output of mpirun --version.',
              file=sys.stderr)
        return None
    else:
        print("Was not able to run %s:\n%s" % (command, output_msg),
              file=sys.stderr)
        return None


def mpi_run(settings, common_intfs, env, command, stdout=None, stderr=None, run_func=safe_shell_exec.execute):
    """
    Runs mpi_run.

    Args:
        settings: Settings for running MPI.
                  Note: settings.num_proc and settings.hosts must not be None.
        common_intfs: Interfaces to include by MPI.
        env: Environment dictionary to use for running MPI.
        command: Command and arguments to run as a list of string.
        stdout: Stdout of the mpi process.
                Only used when settings.run_func_mode is True.
        stderr: Stderr of the mpi process.
                Only used when settings.run_func_mode is True.
        run_func: Run function to use. Must have arguments 'command' and 'env'.
                  Only used when settings.run_func_mode is True.
                  Defaults to safe_shell_exec.execute.
    """
    mpi_impl_flags = _get_mpi_implementation_flags()
    if mpi_impl_flags is None:
        raise Exception(
            'horovodrun convenience script does not find an installed MPI.\n\n'
            'Choose one of:\n'
            '1. Install Open MPI 4.0.0+ or IBM Spectrum MPI or MPICH and re-install Horovod '
            '(use --no-cache-dir pip option).\n'
            '2. Run distributed '
            'training script using the standard way provided by your'
            ' MPI distribution (usually mpirun, srun, or jsrun).\n'
            '3. Use built-in gloo option (horovodrun --gloo ...).')

    ssh_port_arg = '-mca plm_rsh_args \"-p {ssh_port}\"'.format(
            ssh_port=settings.ssh_port) if settings.ssh_port else ''

    # if user does not specify any hosts, mpirun by default uses local host.
    # There is no need to specify localhost.
    hosts_arg = '-H {hosts}'.format(hosts=settings.hosts)

    tcp_intf_arg = '-mca btl_tcp_if_include {common_intfs}'.format(
        common_intfs=','.join(common_intfs)) if common_intfs else ''
    nccl_socket_intf_arg = '-x NCCL_SOCKET_IFNAME={common_intfs}'.format(
        common_intfs=','.join(common_intfs)) if common_intfs else ''

    # On large cluster runs (e.g. Summit), we need extra settings to work around OpenMPI issues
    if settings.num_hosts and settings.num_hosts >= _LARGE_CLUSTER_THRESHOLD:
        mpi_impl_flags.append('-mca plm_rsh_no_tree_spawn true')
        mpi_impl_flags.append('-mca plm_rsh_num_concurrent {}'.format(settings.num_proc))

    # Pass all the env variables to the mpirun command.
    mpirun_command = (
        'mpirun --allow-run-as-root --tag-output '
        '-np {num_proc} {hosts_arg} '
        '-bind-to none -map-by slot '
        '{mpi_args} '
        '{ssh_port_arg} '
        '{tcp_intf_arg} '
        '{nccl_socket_intf_arg} '
        '{output_filename_arg} '
        '{env} {extra_mpi_args} {command}'  # expect a lot of environment variables
        .format(num_proc=settings.num_proc,
                hosts_arg=hosts_arg,
                mpi_args=' '.join(mpi_impl_flags),
                tcp_intf_arg=tcp_intf_arg,
                nccl_socket_intf_arg=nccl_socket_intf_arg,
                ssh_port_arg=ssh_port_arg,
                output_filename_arg='--output-filename ' + settings.output_filename
                                    if settings.output_filename else '',
                env=' '.join('-x %s' % key for key in sorted(env.keys())
                             if env_util.is_exportable(key)),

                extra_mpi_args=settings.extra_mpi_args if settings.extra_mpi_args else '',
                command=' '.join(quote(par) for par in command))
    )

    if settings.verbose >= 2:
        print(mpirun_command)

    # Execute the mpirun command.
    if settings.run_func_mode:
        exit_code = run_func(command=mpirun_command, env=env, stdout=stdout, stderr=stderr)
        if exit_code != 0:
            raise RuntimeError("mpirun failed with exit code {exit_code}".format(exit_code=exit_code))
    else:
        os.execve('/bin/sh', ['/bin/sh', '-c', mpirun_command], env)

