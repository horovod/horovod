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

import copy
import os
import sys

from shlex import quote

from horovod.runner.common.util import env as env_util, hosts, safe_shell_exec, tiny_shell_exec

# MPI implementations
_OMPI_IMPL = 'OpenMPI'
_SMPI_IMPL = 'SpectrumMPI'
_MPICH_IMPL = 'MPICH'
_IMPI_IMPL = "IntelMPI"
_UNKNOWN_IMPL = 'Unknown'
_MISSING_IMPL = 'Missing'

# Open MPI Flags
_OMPI_FLAGS = ['-mca pml ob1', '-mca btl ^openib']
# Spectrum MPI Flags
_SMPI_FLAGS = []
_SMPI_FLAGS_TCP = ['-tcp']
# MPICH Flags
_MPICH_FLAGS = []
# Intel MPI Flags
_IMPI_FLAGS = []

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


def mpi_available(env=None):
    return _get_mpi_implementation(env) not in {_UNKNOWN_IMPL, _MISSING_IMPL}


def is_open_mpi(env=None):
    return _get_mpi_implementation(env) == _OMPI_IMPL


def is_spectrum_mpi(env=None):
    return _get_mpi_implementation(env) == _SMPI_IMPL


def is_mpich(env=None):
    return _get_mpi_implementation(env) == _MPICH_IMPL


def is_intel_mpi(env=None):
    return _get_mpi_implementation(env) == _IMPI_IMPL


def _get_mpi_implementation(env=None):
    """
    Detects the available MPI implementation by invoking `mpirun --version`.
    This command is executed by the given execute function, which takes the
    command as the only argument and returns (output, exit code). Output
    represents the stdout and stderr as a string.

    Returns one of:
    - _OMPI_IMPL, _SMPI_IMPL, _MPICH_IMPL or _IMPI_IMPL for known implementations
    - _UNKNOWN_IMPL for any unknown implementation
    - _MISSING_IMPL if `mpirun --version` could not be executed.

    :param env: environment variable to use to run mpirun
    :return: string representing identified implementation
    """
    command = 'mpirun --version'
    res = tiny_shell_exec.execute(command, env)
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
        elif 'Intel(R) MPI' in output:
            return _IMPI_IMPL

        print('Unknown MPI implementation given in output of mpirun --version:', file=sys.stderr)
        print(output, file=sys.stderr)
        return _UNKNOWN_IMPL
    else:
        print('Was unable to run {command}:'.format(command=command), file=sys.stderr)
        print(output, file=sys.stderr)
        return _MISSING_IMPL


def _get_mpi_implementation_flags(tcp_flag, env=None):
    if is_open_mpi(env):
        return list(_OMPI_FLAGS), list(_NO_BINDING_ARGS), _OMPI_IMPL
    elif is_spectrum_mpi(env):
        return (list(_SMPI_FLAGS_TCP) if tcp_flag else list(_SMPI_FLAGS)), list(_SOCKET_BINDING_ARGS), _SMPI_IMPL
    elif is_mpich(env):
        return list(_MPICH_FLAGS), list(_NO_BINDING_ARGS), _MPICH_IMPL
    elif is_intel_mpi(env):
        return list(_IMPI_FLAGS), [], _IMPI_IMPL
    else:
        return None, None, None


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
    if env is not None and not isinstance(env, dict):
        raise Exception('env argument must be a dict, not {type}: {env}'
                        .format(type=type(env), env=env))

    mpi_impl_flags, impl_binding_args, mpi = _get_mpi_implementation_flags(settings.tcp_flag, env=env)
    if mpi_impl_flags is None:
        raise Exception(_MPI_NOT_FOUND_ERROR_MSG)

    impi_or_mpich = mpi in (_IMPI_IMPL, _MPICH_IMPL)

    ssh_args = []
    if settings.ssh_port:
        ssh_args += [f'-p {settings.ssh_port}']
    if settings.ssh_identity_file:
        ssh_args += [f'-i {settings.ssh_identity_file}']

    mpi_ssh_args = ''
    if ssh_args:
        joined_ssh_args = ' '.join(ssh_args)
        mpi_ssh_args = f'-bootstrap=ssh -bootstrap-exec-args \"{joined_ssh_args}\"' if impi_or_mpich else f'-mca plm_rsh_args \"{joined_ssh_args}\"'

    tcp_intf_arg = '-mca btl_tcp_if_include {nics}'.format(
        nics=','.join(nics)) if nics and not impi_or_mpich else ''
    nccl_socket_intf_arg = '-{opt} NCCL_SOCKET_IFNAME={nics}'.format(
        opt='genv' if impi_or_mpich else 'x',
        nics=','.join(nics)) if nics else ''

    # On large cluster runs (e.g. Summit), we need extra settings to work around OpenMPI issues
    host_names, host_to_slots = hosts.parse_hosts_and_slots(settings.hosts)
    if not impi_or_mpich and host_names and len(host_names) >= _LARGE_CLUSTER_THRESHOLD:
        mpi_impl_flags.append('-mca plm_rsh_no_tree_spawn true')
        mpi_impl_flags.append('-mca plm_rsh_num_concurrent {}'.format(len(host_names)))

    # if user does not specify any hosts, mpirun by default uses local host.
    # There is no need to specify localhost.
    hosts_arg = '-{opt} {hosts}'.format(opt='hosts' if impi_or_mpich else 'H',
                hosts=','.join(host_names) if host_names and impi_or_mpich else settings.hosts)

    ppn_arg = ' '
    if host_to_slots and impi_or_mpich:
        ppn = host_to_slots[host_names[0]]
        for h_name in host_names[1:]:
            if ppn != host_to_slots[h_name]:
                raise Exception('''Different slots in -hosts parameter are not supported in Intel(R) MPI.
                                 Use -machinefile <machine_file> for this purpose.''')
        ppn_arg = ' -ppn {} '.format(ppn)

    if settings.prefix_output_with_timestamp and not impi_or_mpich:
        mpi_impl_flags.append('--timestamp-output')

    binding_args = settings.binding_args if settings.binding_args and not impi_or_mpich else ' '.join(impl_binding_args)

    basic_args = '-l' if impi_or_mpich else '--allow-run-as-root --tag-output'

    output = []
    if settings.output_filename:
        output.append('-outfile-pattern' if impi_or_mpich else '--output-filename')
        output.append(settings.output_filename)

    env_list = '' if impi_or_mpich else ' '.join(
                    '-x %s' % key for key in sorted(env.keys()) if env_util.is_exportable(key))

    # Pass all the env variables to the mpirun command.
    mpirun_command = (
        'mpirun {basic_args} '
        '-np {num_proc}{ppn_arg}{hosts_arg} '
        '{binding_args} '
        '{mpi_args} '
        '{mpi_ssh_args} '
        '{tcp_intf_arg} '
        '{nccl_socket_intf_arg} '
        '{output_filename_arg} '
        '{env} {extra_mpi_args} {command}'  # expect a lot of environment variables
        .format(basic_args=basic_args,
                num_proc=settings.num_proc,
                ppn_arg=ppn_arg,
                hosts_arg=hosts_arg,
                binding_args=binding_args,
                mpi_args=' '.join(mpi_impl_flags),
                tcp_intf_arg=tcp_intf_arg,
                nccl_socket_intf_arg=nccl_socket_intf_arg,
                mpi_ssh_args=mpi_ssh_args,
                output_filename_arg=' '.join(output),
                env=env_list,
                extra_mpi_args=settings.extra_mpi_args if settings.extra_mpi_args else '',
                command=' '.join(quote(par) for par in command))
    )

    if settings.verbose >= 2:
        print(mpirun_command)

    # we need the driver's PATH and PYTHONPATH in env to run mpirun,
    # env for mpirun is different to env encoded in mpirun_command
    for var in ['PATH', 'PYTHONPATH']:
        if var not in env and var in os.environ:
            # copy env so we do not leak env modifications
            env = copy.copy(env)
            # copy var over from os.environ
            env[var] = os.environ[var]

    # Execute the mpirun command.
    if settings.run_func_mode:
        exit_code = safe_shell_exec.execute(mpirun_command, env=env, stdout=stdout, stderr=stderr)
        if exit_code != 0:
            raise RuntimeError("mpirun failed with exit code {exit_code}".format(exit_code=exit_code))
    else:
        os.execve('/bin/sh', ['/bin/sh', '-c', mpirun_command], env)
