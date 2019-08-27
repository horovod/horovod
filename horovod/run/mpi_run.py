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

try:
    from shlex import quote
except ImportError:
    from pipes import quote


def _is_open_mpi_installed():
    output = six.StringIO()
    command = 'mpirun --version'
    try:
        exit_code = safe_shell_exec.execute(command, stdout=output,
                                            stderr=output)
        output_msg = output.getvalue()
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return False
    finally:
        output.close()

    if exit_code == 0:
        if 'Open MPI' not in output_msg:
            print('Open MPI not found in output of mpirun --version.',
                  file=sys.stderr)
            return False
        else:
            return True
    else:
        print("Was not able to run %s:\n%s" % (command, output_msg),
              file=sys.stderr)
        return False


def mpi_run(settings, common_intfs, env):
    if not _is_open_mpi_installed():
        raise Exception(
            'horovodrun convenience script does not find an installed OpenMPI.\n\n'
            'Choose one of:\n'
            '1. Install Open MPI 4.0.0+ and re-install Horovod '
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

    # Pass all the env variables to the mpirun command.
    mpirun_command = (
        'mpirun --allow-run-as-root --tag-output '
        '-np {num_proc} {hosts_arg} '
        '-bind-to none -map-by slot '
        '-mca pml ob1 -mca btl ^openib '
        '{ssh_port_arg} '
        '{tcp_intf_arg} '
        '{nccl_socket_intf_arg} '
        '{env} {command}'  # expect a lot of environment variables
        .format(num_proc=settings.num_proc,
                hosts_arg=hosts_arg,
                tcp_intf_arg=tcp_intf_arg,
                nccl_socket_intf_arg=nccl_socket_intf_arg,
                ssh_port_arg=ssh_port_arg,
                env=' '.join('-x %s' % key for key in env.keys()
                             if env_util.is_exportable(key)),
                command=' '.join(quote(par) for par in settings.command))
    )

    if settings.verbose >= 2:
        print(mpirun_command)
    # Execute the mpirun command.
    os.execve('/bin/sh', ['/bin/sh', '-c', mpirun_command], env)
