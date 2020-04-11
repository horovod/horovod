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

import copy
import os
import sys

from horovod.run.common.util import env as env_util, safe_shell_exec, tiny_shell_exec

# MPI implementations
_OMPI_IMPL = 'OpenMPI'
_SMPI_IMPL = 'SpectrumMPI'
_MPICH_IMPL = 'MPICH'
_UNKNOWN_IMPL = 'Unknown'
_MISSING_IMPL = 'Missing'

# Open MPI Flags
_OMPI_FLAGS = ['-mca pml ob1', '-mca btl ^openib']
# Spectrum MPI Flags
_SMPI_FLAGS = []
_SMPI_FLAGS_TCP = ['-tcp']
# MPICH Flags
_MPICH_FLAGS = []

# Threshold for large cluster MPI issues:
_LARGE_CLUSTER_THRESHOLD = 64
# No process binding args
_NO_BINDING_ARGS = ['-bind-to none', '-map-by slot']
# Process socket binding args
_SOCKET_BINDING_ARGS = ['-bind-to socket', '-map-by socket', '-rank-by core']

# MPI not found error message
_MPI_NOT_FOUND_ERROR_MSG= ('horovod does not find an installed MPI.\n\n'
                           'Choose one of:\n'
                           '1. Install Open MPI 4.0.0+ or IBM Spectrum MPI or MPICH and re-install Horovod '
                           '(use --no-cache-dir pip option).\n'
                           '2. Run distributed '
                           'training script using the standard way provided by your'
                           ' MPI distribution (usually mpirun, srun, or jsrun).\n'
                           '3. Use built-in gloo option (horovodrun --gloo ...).')

try:
    from shlex import quote
except ImportError:
    from pipes import quote


def mpi_available():
    return _get_mpi_implementation() not in {_UNKNOWN_IMPL, _MISSING_IMPL}


def is_open_mpi():
    return _get_mpi_implementation() == _OMPI_IMPL


def is_spectrum_mpi():
    return _get_mpi_implementation() == _SMPI_IMPL


def is_mpich():
    return _get_mpi_implementation() == _MPICH_IMPL


def _get_mpi_implementation():
    """
    Detects the available MPI implementation by invoking `mpirun --version`.
    This command is executed by the given execute function, which takes the
    command as the only argument and returns (output, exit code). Output
    represents the stdout and stderr as a string.

    Returns one of:
    - _OMPI_IMPL, _SMPI_IMPL or _MPICH_IMPL for known implementations
    - _UNKNOWN_IMPL for any unknown implementation
    - _MISSING_IMPL if `mpirun --version` could not be executed.

    :return: string representing identified implementation
    """
    command = 'mpirun --version'
    res = tiny_shell_exec.execute(command)
    if res is None:
        return _MISSING_IMPL
    (output, exit_code) = res

    if exit_code == 0:
        if 'Open MPI' in output or 'OpenRTE' in output:
            return _OMPI_IMPL
        elif 'IBM Spectrum MPI' in output:
            return _SMPI_IMPL
        elif 'MPICH' in output:
            return _MPICH_IMPL

        print('Unknown MPI implementation given in output of mpirun --version:', file=sys.stderr)
        print(output, file=sys.stderr)
        return _UNKNOWN_IMPL
    else:
        print('Was unable to run {command}:'.format(command=command), file=sys.stderr)
        print(output, file=sys.stderr)
        return _MISSING_IMPL


def _get_mpi_implementation_flags(tcp_flag):
    if is_open_mpi():
        return list(_OMPI_FLAGS), list(_NO_BINDING_ARGS)
    elif is_spectrum_mpi():
        return list(_SMPI_FLAGS) if not tcp_flag else list(_SMPI_FLAGS_TCP), list(_SOCKET_BINDING_ARGS)
    elif is_mpich():
        return list(_MPICH_FLAGS), list(_NO_BINDING_ARGS)
    else:
        return None, None


def mpi_run(settings, nics, env, command, stdout=None, stderr=None):
    """
    Runs mpi_run.

    Args:
        settings: Settings for running MPI.
                  Note: settings.num_proc and settings.hosts must not be None.
        nics: Interfaces to include by MPI.
        env: Environment dictionary to use for running command.
        command: Command and arguments to run as a list of string.
        stdout: Stdout of the mpi process.
                Only used when settings.run_func_mode is True.
        stderr: Stderr of the mpi process.
                Only used when settings.run_func_mode is True.
    """
    mpi_impl_flags, impl_binding_args = _get_mpi_implementation_flags(settings.tcp_flag)
    if mpi_impl_flags is None:
        raise Exception(_MPI_NOT_FOUND_ERROR_MSG)

    ssh_port_arg = '-mca plm_rsh_args \"-p {ssh_port}\"'.format(
            ssh_port=settings.ssh_port) if settings.ssh_port else ''

    # if user does not specify any hosts, mpirun by default uses local host.
    # There is no need to specify localhost.
    hosts_arg = '-H {hosts}'.format(hosts=settings.hosts)

    tcp_intf_arg = '-mca btl_tcp_if_include {nics}'.format(
        nics=','.join(nics)) if nics else ''
    nccl_socket_intf_arg = '-x NCCL_SOCKET_IFNAME={nics}'.format(
        nics=','.join(nics)) if nics else ''

    # On large cluster runs (e.g. Summit), we need extra settings to work around OpenMPI issues
    if settings.num_hosts and settings.num_hosts >= _LARGE_CLUSTER_THRESHOLD:
        mpi_impl_flags.append('-mca plm_rsh_no_tree_spawn true')
        mpi_impl_flags.append('-mca plm_rsh_num_concurrent {}'.format(settings.num_hosts))

    binding_args = settings.binding_args if settings.binding_args else ' '.join(impl_binding_args)

    # Pass all the env variables to the mpirun command.
    mpirun_command = (
        'mpirun --allow-run-as-root --tag-output '
        '-np {num_proc} {hosts_arg} '
        '{binding_args} '
        '{mpi_args} '
        '{ssh_port_arg} '
        '{tcp_intf_arg} '
        '{nccl_socket_intf_arg} '
        '{output_filename_arg} '
        '{env} {extra_mpi_args} {command}'  # expect a lot of environment variables
        .format(num_proc=settings.num_proc,
                hosts_arg=hosts_arg,
                binding_args=binding_args,
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

    # we need the driver's PATH in env to run mpirun,
    # env for mpirun is different to env encoded in mpirun_command
    if 'PATH' not in env and 'PATH' in os.environ:
        env = copy.copy(env)  # copy env so we do not leak env modifications
        env['PATH'] = os.environ['PATH']

    # Execute the mpirun command.
    if settings.run_func_mode:
        exit_code = safe_shell_exec.execute(mpirun_command, env=env, stdout=stdout, stderr=stderr)
        if exit_code != 0:
            raise RuntimeError("mpirun failed with exit code {exit_code}".format(exit_code=exit_code))
    else:
        os.execve('/bin/sh', ['/bin/sh', '-c', mpirun_command], env)
