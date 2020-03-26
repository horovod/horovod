# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

import argparse
import hashlib
import os
import re
import sys
import textwrap

try:
    from shlex import quote
except ImportError:
    from pipes import quote

import six
import yaml
import cloudpickle

import horovod

from horovod.common.util import (extension_available,
                                 gloo_built, mpi_built,
                                 nccl_built, ddl_built, ccl_built)
from horovod.run.common.util import config_parser, safe_shell_exec, timeout, secret
from horovod.run.common.util import settings as hvd_settings
from horovod.run.driver import driver_service
from horovod.run.util import cache, threads, network, lsf
from horovod.run.gloo_run import gloo_run
from horovod.run.mpi_run import mpi_run
from horovod.run.js_run import js_run, is_jsrun_installed
from horovod.run.http.http_client import read_data_from_kvstore, put_data_into_kvstore
from horovod.run.http.http_server import KVStoreServer


# Cached information of horovodrun functions be stored in this directory
CACHE_FOLDER = os.path.join(os.path.expanduser('~'), '.horovod')

# Cache entries will be stale if they are older than this number of minutes
CACHE_STALENESS_THRESHOLD_MINUTES = 60

# Number of retries for sshing into the hosts
SSH_RETRIES = 5


@cache.use_cache()
def _check_all_hosts_ssh_successful(host_addresses, ssh_port=None):
    """
    checks if ssh can successfully be performed to all the hosts.
    :param host_addresses: list of addresses to ssh into. for example,
        ['worker-0','worker-1']
        ['10.11.11.11', '10.11.11.12']
    :type host_addresses: list(strings)
    :return: Returns True if all ssh was successful into all the addresses.
    """

    def exec_command(command):
        exit_code = 1
        output_msg = ''

        # Try ssh 5 times
        for i in range(SSH_RETRIES):
            output = six.StringIO()
            try:
                exit_code = safe_shell_exec.execute(command,
                                                    stdout=output,
                                                    stderr=output)
                if exit_code == 0:
                    break
                else:
                    output_msg = output.getvalue()
            finally:
                output.close()
        return exit_code, output_msg

    ssh_port_arg = '-p {ssh_port}'.format(
        ssh_port=ssh_port) if ssh_port else ''

    ssh_command_format = 'ssh -o StrictHostKeyChecking=no {host} {ssh_port_arg} date'

    args_list = [[ssh_command_format.format(host=host_address,
                                            ssh_port_arg=ssh_port_arg)]
                 for host_address in host_addresses]
    ssh_exit_codes = \
        threads.execute_function_multithreaded(exec_command,
                                               args_list)

    ssh_successful_to_all_hosts = True
    for index, ssh_status in six.iteritems(ssh_exit_codes):
        exit_code, output_msg = ssh_status[0], ssh_status[1]
        if exit_code != 0:
            print('ssh not successful for host {host}:\n{msg_output}'
                  .format(host=host_addresses[index],
                          msg_output=output_msg))

            ssh_successful_to_all_hosts = False
    if not ssh_successful_to_all_hosts:
        exit(1)
    return True


def check_build(verbose):
    def get_check(value):
        return 'X' if value else ' '

    output = '''{verbose_newline}\
    Horovod v{version}:

    Available Frameworks:
        [{tensorflow}] TensorFlow
        [{torch}] PyTorch
        [{mxnet}] MXNet

    Available Controllers:
        [{mpi}] MPI
        [{gloo}] Gloo

    Available Tensor Operations:
        [{nccl_ops}] NCCL
        [{ddl_ops}] DDL
        [{ccl_ops}] CCL
        [{mpi_ops}] MPI
        [{gloo_ops}] Gloo\
    '''.format(verbose_newline='\n' if verbose else '',
               version=horovod.__version__,
               tensorflow=get_check(extension_available('tensorflow', verbose=verbose)),
               torch=get_check(extension_available('torch', verbose=verbose)),
               mxnet = get_check(extension_available('mxnet', verbose=verbose)),
               mpi=get_check(mpi_built(verbose=verbose)),
               gloo=get_check(gloo_built(verbose=verbose)),
               nccl_ops=get_check(nccl_built(verbose=verbose)),
               ddl_ops=get_check(ddl_built(verbose=verbose)),
               mpi_ops=get_check(mpi_built(verbose=verbose)),
               ccl_ops=get_check(ccl_built(verbose=verbose)),
               gloo_ops=get_check(gloo_built(verbose=verbose)))
    print(textwrap.dedent(output))
    os._exit(0)


def make_check_build_action(np_arg):
    class CheckBuildAction(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            # If -cb is specified, make -np optional
            np_arg.required = False
            args.check_build = True

    return CheckBuildAction


def make_override_action(override_args):
    class StoreOverrideAction(argparse.Action):
        def __init__(self,
                     option_strings,
                     dest,
                     default=None,
                     type=None,
                     choices=None,
                     required=False,
                     help=None):
            super(StoreOverrideAction, self).__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=1,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help)

        def __call__(self, parser, args, values, option_string=None):
            override_args.add(self.dest)
            setattr(args, self.dest, values[0])

    return StoreOverrideAction


def make_override_bool_action(override_args, bool_value):
    class StoreOverrideBoolAction(argparse.Action):
        def __init__(self,
                     option_strings,
                     dest,
                     required=False,
                     help=None):
            super(StoreOverrideBoolAction, self).__init__(
                option_strings=option_strings,
                dest=dest,
                const=bool_value,
                nargs=0,
                default=None,
                required=required,
                help=help)

        def __call__(self, parser, args, values, option_string=None):
            override_args.add(self.dest)
            setattr(args, self.dest, self.const)

    return StoreOverrideBoolAction


def make_override_true_action(override_args):
    return make_override_bool_action(override_args, True)


def make_override_false_action(override_args):
    return make_override_bool_action(override_args, False)


def parse_args():
    override_args = set()

    parser = argparse.ArgumentParser(description='Horovod Runner')

    parser.add_argument('-v', '--version', action='version', version=horovod.__version__,
                        help='Shows Horovod version.')

    np_arg = parser.add_argument('-np', '--num-proc', action='store', dest='np',
                                 type=int, required=not lsf.LSFUtils.using_lsf(),
                                 help='Total number of training processes.')

    parser.add_argument('-cb', '--check-build', action=make_check_build_action(np_arg), nargs=0,
                        help='Shows which frameworks and libraries have been built into Horovod.')

    parser.add_argument('-p', '--ssh-port', action='store', dest='ssh_port',
                        type=int, help='SSH port on all the hosts.')

    parser.add_argument('--disable-cache', action='store_true',
                        dest='disable_cache',
                        help='If the flag is not set, horovodrun will perform '
                             'the initialization checks only once every 60 '
                             'minutes -- if the checks successfully pass. '
                             'Otherwise, all the checks will run every time '
                             'horovodrun is called.')

    parser.add_argument('--start-timeout', action='store',
                        dest='start_timeout', type=int,
                        help='Horovodrun has to perform all the checks and '
                             'start the processes before the specified '
                             'timeout. The default value is 30 seconds. '
                             'Alternatively, The environment variable '
                             'HOROVOD_START_TIMEOUT can also be used to '
                             'specify the initialization timeout.')

    parser.add_argument('--network-interface', action='store', dest='nics',
                        help='Network interfaces that can be used for communication separated by '
                             'comma. If not specified, Horovod will find the common NICs among all '
                             'the workers and use it; example, --network-interface "eth0,eth1".')

    parser.add_argument('--output-filename', action='store',
                        help='For Gloo, writes stdout / stderr of all processes to a filename of the form '
                             '<output_filename>/rank.<rank>/<stdout | stderr>. The <rank> will be padded with 0 '
                             'characters to ensure lexicographical order. For MPI, delegates its behavior to mpirun.')

    parser.add_argument('--verbose', action='store_true',
                        dest='verbose',
                        help='If this flag is set, extra messages will '
                             'be printed.')

    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Command to be executed.')

    parser.add_argument('--config-file', action='store', dest='config_file',
                        help='Path to YAML file containing runtime parameter configuration for Horovod. '
                             'Note that this will override any command line arguments provided before '
                             'this argument, and will be overridden by any arguments that come after it.')

    group_params = parser.add_argument_group('tuneable parameter arguments')
    group_params.add_argument('--fusion-threshold-mb', action=make_override_action(override_args),type=int,
                              help='Fusion buffer threshold in MB. This is the maximum amount of '
                                   'tensor data that can be fused together into a single batch '
                                   'during allreduce / allgather. Setting 0 disables tensor fusion. '
                                   '(default: 64)')
    group_params.add_argument('--cycle-time-ms', action=make_override_action(override_args), type=float,
                              help='Cycle time in ms. This is the delay between each tensor fusion '
                                   'cycle. The larger the cycle time, the more batching, but the '
                                   'greater latency between each allreduce / allgather operations. '
                                   '(default: 5')
    group_params.add_argument('--cache-capacity', action=make_override_action(override_args), type=int,
                              help='Maximum number of tensor names that will be cached to reduce amount '
                                   'of coordination required between workers before performing allreduce / '
                                   'allgather. (default: 1024')

    group_hierarchical_allreduce = group_params.add_mutually_exclusive_group()
    group_hierarchical_allreduce.add_argument('--hierarchical-allreduce',
                                              action=make_override_true_action(override_args),
                                              help='Perform hierarchical allreduce between workers instead of '
                                                   'ring allreduce. Hierarchical allreduce performs a local '
                                                   'allreduce / gather within a host, then a parallel cross allreduce '
                                                   'between equal local ranks across workers, and finally a '
                                                   'local gather.')
    group_hierarchical_allreduce.add_argument('--no-hierarchical-allreduce', dest='hierarchical_allreduce',
                                              action=make_override_false_action(override_args),
                                              help='Explicitly disable hierarchical allreduce to prevent autotuning '
                                                   'from adjusting it.')

    group_hierarchical_allgather = group_params.add_mutually_exclusive_group()
    group_hierarchical_allgather.add_argument('--hierarchical-allgather',
                                              action=make_override_true_action(override_args),
                                              help='Perform hierarchical allgather between workers instead of '
                                                   'ring allgather. See hierarchical allreduce for algorithm details.')
    group_hierarchical_allgather.add_argument('--no-hierarchical-allgather', dest='hierarchical_allgather',
                                              action=make_override_false_action(override_args),
                                              help='Explicitly disable hierarchical allgather to prevent autotuning '
                                                   'from adjusting it.')

    group_autotune = parser.add_argument_group('autotune arguments')
    group_autotune_enabled = group_autotune.add_mutually_exclusive_group()
    group_autotune_enabled.add_argument('--autotune', action=make_override_true_action(override_args),
                                        help='Perform autotuning to select parameter argument values that maximimize '
                                             'throughput for allreduce / allgather. Any parameter explicitly set will '
                                             'be held constant during tuning.')
    group_autotune_enabled.add_argument('--no-autotune', dest='autotune',
                                        action=make_override_false_action(override_args), help=argparse.SUPPRESS)
    group_autotune.add_argument('--autotune-log-file', action=make_override_action(override_args),
                                help='Comma-separated log of trials containing each hyperparameter and the '
                                     'score of the trial. The last row will always contain the best value '
                                     'found.')
    group_autotune.add_argument('--autotune-warmup-samples', action=make_override_action(override_args),
                                type=int, default=3,
                                help='Number of samples to discard before beginning the optimization process '
                                     'during autotuning. Performance during the first few batches can be '
                                     'affected by initialization and cache warmups. (default: %(default)s)')
    group_autotune.add_argument('--autotune-steps-per-sample', action=make_override_action(override_args),
                                type=int, default=10,
                                help='Number of steps (approximate) to record before observing a sample. The sample '
                                     'score is defined to be the median score over all batches within the sample. The '
                                     'more batches per sample, the less variance in sample scores, but the longer '
                                     'autotuning will take. (default: %(default)s)')
    group_autotune.add_argument('--autotune-bayes-opt-max-samples', action=make_override_action(override_args),
                                type=int, default=20,
                                help='Maximum number of samples to collect for each Bayesian optimization process. '
                                     '(default: %(default)s)')
    group_autotune.add_argument('--autotune-gaussian-process-noise', action=make_override_action(override_args),
                                type=float, default=0.8,
                                help='Regularization value [0, 1] applied to account for noise in samples. '
                                     '(default: %(default)s)')

    group_timeline = parser.add_argument_group('timeline arguments')
    group_timeline.add_argument('--timeline-filename', action=make_override_action(override_args),
                                help='JSON file containing timeline of Horovod events used for debugging '
                                     'performance. If this is provided, timeline events will be recorded, '
                                     'which can have a negative impact on training performance.')
    group_timeline_cycles = group_timeline.add_mutually_exclusive_group()
    group_timeline_cycles.add_argument('--timeline-mark-cycles', action=make_override_true_action(override_args),
                                       help='Mark cycles on the timeline. Only enabled if the timeline filename '
                                            'is provided.')
    group_timeline_cycles.add_argument('--no-timeline-mark-cycles', dest='timeline_mark_cycles',
                                       action=make_override_false_action(override_args), help=argparse.SUPPRESS)

    group_stall_check = parser.add_argument_group('stall check arguments')
    group_stall_check_enabled = group_stall_check.add_mutually_exclusive_group()
    group_stall_check_enabled.add_argument('--no-stall-check', action=make_override_true_action(override_args),
                                           help='Disable the stall check. The stall check will log a warning when '
                                                'workers have stalled waiting for other ranks to submit tensors.')
    group_stall_check_enabled.add_argument('--stall-check', dest='no_stall_check',
                                           action=make_override_false_action(override_args), help=argparse.SUPPRESS)
    group_stall_check.add_argument('--stall-check-warning-time-seconds', action=make_override_action(override_args),
                                   type=int, default=60,
                                   help='Seconds until the stall warning is logged to stderr. (default: %(default)s)')
    group_stall_check.add_argument('--stall-check-shutdown-time-seconds', action=make_override_action(override_args),
                                   type=int, default=0,
                                   help='Seconds until Horovod is shutdown due to stall. Shutdown will only take '
                                        'place if this value is greater than the warning time. (default: %(default)s)')

    group_library_options = parser.add_argument_group('library arguments')
    group_mpi_threads_disable = group_library_options.add_mutually_exclusive_group()
    group_mpi_threads_disable.add_argument('--mpi-threads-disable', action=make_override_true_action(override_args),
                                           help='Disable MPI threading support. Only applies when running in MPI '
                                                'mode. In some cases, multi-threaded MPI can slow down other '
                                                'components, but is necessary if you wish to run mpi4py on top '
                                                'of Horovod.')
    group_mpi_threads_disable.add_argument('--no-mpi-threads-disable', dest='mpi_threads_disable',
                                           action=make_override_false_action(override_args), help=argparse.SUPPRESS)
    group_library_options.add_argument('--mpi-args', action='store', dest='mpi_args',
                                       help='Extra MPI arguments to pass to mpirun. '
                                       'They need to be passed with the equal sign to avoid parsing issues. '
                                       'e.g. --mpi-args="--map-by ppr:6:node"')
    group_library_options.add_argument('--tcp', action='store_true', dest='tcp_flag',
                                       help='If this flag is set, only TCP is used for communication.')
    group_library_options.add_argument('--binding-args', action='store', dest='binding_args',
                                       help='Process binding arguments. Default is socket for Spectrum MPI '
                                       'and no binding for other cases. e.g. --binding-args="--rankfile myrankfile"')
    group_library_options.add_argument('--num-nccl-streams', action=make_override_action(override_args),
                                       type=int, default=1,
                                       help='Number of NCCL streams. Only applies when running with NCCL support. '
                                            '(default: %(default)s)')
    group_library_options.add_argument('--ccl-bgt-affinity', action=make_override_action(override_args),
                                       type=int, default=0,
                                       help='CCL background thread affinity. Only applies when running with CCL '
                                            'support. (default: %(default)s)')
    group_library_options.add_argument('--gloo-timeout-seconds', action=make_override_action(override_args),
                                       type=int, default=30,
                                       help='Timeout in seconds for Gloo operations to complete. '
                                            '(default: %(default)s)')

    group_logging = parser.add_argument_group('logging arguments')
    group_logging.add_argument('--log-level', action=make_override_action(override_args),
                               choices=config_parser.LOG_LEVELS,
                               help='Minimum level to log to stderr from the Horovod backend. (default: WARNING).')
    group_logging_timestamp = group_logging.add_mutually_exclusive_group()
    group_logging_timestamp.add_argument('--log-hide-timestamp', action=make_override_true_action(override_args),
                                         help='Hide the timestamp from Horovod log messages.')
    group_logging_timestamp.add_argument('--no-log-hide-timestamp', dest='log_hide_timestamp',
                                         action=make_override_false_action(override_args), help=argparse.SUPPRESS)

    group_hosts_parent = parser.add_argument_group('host arguments')
    group_hosts = group_hosts_parent.add_mutually_exclusive_group()
    group_hosts.add_argument('-H', '--hosts', action='store', dest='hosts',
                             help='List of host names and the number of available slots '
                                  'for running processes on each, of the form: <hostname>:<slots> '
                                  '(e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run on host1, '
                                  '4 on host2, and 1 on host3). If not specified, defaults to using '
                                  'localhost:<np>')
    group_hosts.add_argument('-hostfile', '--hostfile', action='store', dest='hostfile',
                             help='Path to a host file containing the list of host names and the number of '
                                  'available slots. Each line of the file must be of the form: '
                                  '<hostname> slots=<slots>')

    group_controller_parent = parser.add_argument_group('controller arguments')
    group_controller = group_controller_parent.add_mutually_exclusive_group()
    group_controller.add_argument('--gloo', action='store_true', dest='use_gloo',
                                  help='Run Horovod using the Gloo controller. This will '
                                       'be the default if Horovod was not built with MPI support.')
    group_controller.add_argument('--mpi', action='store_true', dest='use_mpi',
                                  help='Run Horovod using the MPI controller. This will '
                                       'be the default if Horovod was built with MPI support.')
    group_controller.add_argument('--jsrun', action='store_true', dest='use_jsrun',
                                  help='Launch Horovod processes with jsrun and use the MPI controller. '
                                       'This will be the default if jsrun is installed and Horovod '
                                       'was built with MPI support.')

    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config_parser.set_args_from_config(args, config, override_args)
    config_parser.validate_config_args(args)

    return args


class HorovodArgs(object):
    def __init__(self):
        self.np = 1
        self.check_build = None
        self.ssh_port = None
        self.disable_cache = None
        self.start_timeout = None
        self.output_filename = None
        self.verbose = None
        self.command = None
        self.run_func = None
        self.config_file = None
        self.nics = None

        # tuneable parameter arguments
        self.fusion_threshold_mb = None
        self.cycle_time_ms = None,
        self.cache_capacity = None,

        # hierrachy
        self.hierarchical_allreduce = None
        self.hierarchical_allgather = None

        # autotune arguments
        self.autotune = None
        self.autotune_log_file = None
        self.autotune_warmup_samples = None
        self.autotune_steps_per_sample = None
        self.autotune_bayes_opt_max_samples = None
        self.autotune_gaussian_process_noise = None

        # timeline arguments
        self.timeline_filename = None
        self.timeline_mark_cycles = None

        # stall check arguments
        self.no_stall_check = None
        self.stall_check_warning_time_seconds = None
        self.stall_check_shutdown_time_seconds = None

        # library arguments
        self.mpi_threads_disable = None
        self.mpi_args = None
        self.tcp_flag = None
        self.binding_args = None
        self.num_nccl_streams = None
        self.ccl_bgt_affinity = None
        self.gloo_timeout_seconds = None

        # logging arguments
        self.log_level = None
        self.log_hide_timestamp = None

        # host arguments
        self.hosts = None
        self.hostfile = None

        # controller arguments
        self.use_gloo = None
        self.use_mpi = None
        self.use_jsrun = None


def parse_host_files(filename):
    """
    Transform the hostfile into a format of
    <IP address> or <host name>:<Number of GPUs>
    :param filename: Should be in <IP address> or <host name> slots=<number of GPUs>
    :return: Comma separated string of <IP address> or <host name>:<Number of GPUs>
    """
    hosts = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            hostname = line.split()[0]
            slots = line.split('=')[1]
            hosts.append('{name}:{slots}'.format(name=hostname, slots=slots))
    return ','.join(hosts)


def parse_host_names(hosts):
    host_list = hosts.split(',')
    all_host_names = []
    pattern = re.compile(r'^[\w.-]+:[0-9]+$')
    for host in host_list:
        if not pattern.match(host.strip()):
            raise ValueError('Invalid host input, please make sure it has '
                             'format as : worker-0:2,worker-1:2.')
        all_host_names.append(host.strip().split(':')[0])
    return all_host_names


def _run(args):
    if args.check_build:
        check_build(args.verbose)

    # If LSF is used, use default values from job config
    if lsf.LSFUtils.using_lsf():
        if not args.np:
            args.np = lsf.LSFUtils.get_num_processes()
        if not args.hosts and not args.hostfile:
            args.hosts = ','.join('{host}:{np}'.format(host=host, np=lsf.LSFUtils.get_num_gpus())
                         for host in lsf.LSFUtils.get_compute_hosts())

    # if hosts are not specified, either parse from hostfile, or default as
    # localhost
    if not args.hosts:
        if args.hostfile:
            args.hosts = parse_host_files(args.hostfile)
        else:
            # Set hosts to localhost if not specified
            args.hosts = 'localhost:{np}'.format(np=args.np)

    all_host_names = parse_host_names(args.hosts)

    nics_set = set(args.nics.split(',')) if args.nics else None

    # horovodrun has to finish all the checks before this timeout runs out.
    if args.start_timeout:
        start_timeout = args.start_timeout
    else:
        # Lookup default timeout from the environment variable.
        start_timeout = int(os.getenv('HOROVOD_START_TIMEOUT', '30'))

    tmout = timeout.Timeout(start_timeout,
                            message='Timed out waiting for {activity}. Please '
                                    'check connectivity between servers. You '
                                    'may need to increase the --start-timeout '
                                    'parameter if you have too many servers.')
    settings = hvd_settings.Settings(verbose=2 if args.verbose else 0,
                                     ssh_port=args.ssh_port,
                                     extra_mpi_args=args.mpi_args,
                                     tcp_flag=args.tcp_flag,
                                     binding_args=args.binding_args,
                                     key=secret.make_secret_key(),
                                     timeout=tmout,
                                     num_hosts=len(all_host_names),
                                     num_proc=args.np,
                                     hosts=args.hosts,
                                     output_filename=args.output_filename,
                                     run_func_mode=args.run_func is not None,
                                     nics=nics_set)

    # This cache stores the results of checks performed by horovodrun
    # during the initialization step. It can be disabled by setting
    # --disable-cache flag.
    fn_cache = None
    if not args.disable_cache:
        params = ''
        if args.np:
            params += str(args.np) + ' '
        if args.hosts:
            params += str(args.hosts) + ' '
        if args.ssh_port:
            params += str(args.ssh_port)
        parameters_hash = hashlib.md5(params.encode('utf-8')).hexdigest()
        fn_cache = cache.Cache(CACHE_FOLDER, CACHE_STALENESS_THRESHOLD_MINUTES,
                               parameters_hash)

    if settings.verbose >= 2:
        print('Filtering local host names.')
    remote_host_names = network.filter_local_addresses(all_host_names)
    if settings.verbose >= 2:
        print('Remote host found: ' + ' '.join(remote_host_names))

    if len(remote_host_names) > 0:
        if settings.verbose >= 2:
            print('Checking ssh on all remote hosts.')
        # Check if we can ssh into all remote hosts successfully.
        _check_all_hosts_ssh_successful(remote_host_names, args.ssh_port,
                                        fn_cache=fn_cache)
        if settings.verbose >= 2:
            print('SSH was successful into all the remote hosts.')

    nics = driver_service.get_common_interfaces(settings, all_host_names,
                                                remote_host_names, fn_cache)

    if args.run_func:
        # get the driver IPv4 address
        driver_ip = network._get_driver_ip(nics)
        run_func_server = KVStoreServer(verbose=settings.verbose)
        run_func_server_port = run_func_server.start_server()
        pickled_exec_func = cloudpickle.dumps(args.run_func)
        put_data_into_kvstore(driver_ip, run_func_server_port,
                              'runfunc', 'func', pickled_exec_func)

        command = [sys.executable, '-m', 'horovod.run.run_task', str(driver_ip), str(run_func_server_port)]

        try:
            _launch_job(args, remote_host_names, settings, nics, command)
            results = [None] * args.np
            # TODO: make it parallel to improve performance
            for i in range(args.np):
                pickled_result = read_data_from_kvstore(driver_ip, run_func_server_port,
                                                        'runfunc_result', str(i))
                results[i] = cloudpickle.loads(pickled_result)
            return results
        finally:
            run_func_server.shutdown_server()
    else:
        command = args.command
        _launch_job(args, remote_host_names, settings, nics, command)
        return None


def run_controller(use_gloo, gloo_run, use_mpi, mpi_run, use_jsrun, js_run, verbosity):
    verbose = verbosity is not None and verbosity >= 2
    if use_gloo:
        if not gloo_built(verbose=verbose):
            raise ValueError('Gloo support has not been built.  If this is not expected, ensure CMake is installed '
                             'and reinstall Horovod with HOROVOD_WITH_GLOO=1 to debug the build error.')
        gloo_run()
    elif use_mpi:
        if not mpi_built(verbose=verbose):
            raise ValueError('MPI support has not been built.  If this is not expected, ensure MPI is installed '
                             'and reinstall Horovod with HOROVOD_WITH_MPI=1 to debug the build error.')
        mpi_run()
    elif use_jsrun:
        if not mpi_built(verbose=verbose):
            raise ValueError('MPI support has not been built.  If this is not expected, ensure MPI is installed '
                             'and reinstall Horovod with HOROVOD_WITH_MPI=1 to debug the build error.')
        if not lsf.LSFUtils.using_lsf():
            raise ValueError('Horovod did not detect an LSF job.  The jsrun launcher can only be used in that environment. '
                             'Please, pick a different launcher for other environments.')
        js_run()
    else:
        if mpi_built(verbose=verbose):
            if lsf.LSFUtils.using_lsf() and is_jsrun_installed():
                js_run()
            else:
                mpi_run()
        elif gloo_built(verbose=verbose):
            gloo_run()
        else:
            raise ValueError('Neither MPI nor Gloo support has been built. Try reinstalling Horovod ensuring that '
                             'either MPI is installed (MPI) or CMake is installed (Gloo).')


def _launch_job(args, remote_host_names, settings, nics, command):
    env = os.environ.copy()
    config_parser.set_env_from_args(env, args)
    driver_ip = network._get_driver_ip(nics)

    def gloo_run_fn():
        gloo_run(settings, remote_host_names, nics, env, driver_ip, command)

    def mpi_run_fn():
        mpi_run(settings, nics, env, command)

    def js_run_fn():
        js_run(settings, nics, env, command)

    run_controller(args.use_gloo, gloo_run_fn,
                   args.use_mpi, mpi_run_fn,
                   args.use_jsrun, js_run_fn,
                   args.verbose)


def run_commandline():
    args = parse_args()
    args.run_func = None
    _run(args)


def run(
        func,
        args=(),
        kwargs=None,
        np=1,
        hosts=None,
        hostfile=None,
        start_timeout=None,
        ssh_port=None,
        disable_cache=None,
        output_filename=None,
        verbose=None,
        use_gloo=None,
        use_mpi=None,
        mpi_args=None,
        network_interface=None):
    """
    Launch a Horovod job to run the specified process function and get the return value.

    :param func: The function to be run in Horovod job processes. The function return value will
                 be collected as the corresponding Horovod process return value.
                 This function must be compatible with pickle.
    :param args: Arguments to pass to `func`.
    :param kwargs: Keyword arguments to pass to `func`.
    :param np: Number of Horovod processes.
    :param hosts: List of host names and the number of available slots
                  for running processes on each, of the form: <hostname>:<slots>
                  (e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run on host1,
                  4 on host2, and 1 on host3). If not specified, defaults to using localhost:<np>
    :param hostfile: Path to a host file containing the list of host names and the number of
                     available slots. Each line of the file must be of the form:
                     <hostname> slots=<slots>
    :param start_timeout: Horovodrun has to perform all the checks and
                          start the processes before the specified
                          timeout. The default value is 30 seconds.
                          Alternatively, The environment variable
                          HOROVOD_START_TIMEOUT can also be used to
                          specify the initialization timeout.
    :param ssh_port: SSH port on all the hosts.
    :param disable_cache: If the flag is not set, horovodrun will perform
                          the initialization checks only once every 60
                          minutes -- if the checks successfully pass.
                          Otherwise, all the checks will run every time
                          horovodrun is called.'
    :param output_filename: For Gloo, writes stdout / stderr of all processes to a filename of the form
                            <output_filename>/rank.<rank>/<stdout | stderr>. The <rank> will be padded with 0
                            characters to ensure lexicographical order.
                            For MPI, delegates its behavior to mpirun.
    :param verbose: If this flag is set, extra messages will be printed.
    :param use_gloo: Run Horovod using the Gloo controller. This will
                     be the default if Horovod was not built with MPI support.
    :param use_mpi: Run Horovod using the MPI controller. This will
                    be the default if Horovod was built with MPI support.
    :param mpi_args: Extra arguments for the MPI controller. This is only used when use_mpi is True.
    :param network_interface: Network interfaces to use for communication separated by comma. If
                             not specified, Horovod will find the common NICs among all the
                             workers and use those; example, eth0,eth1.
    :return: Return a list which contains values return by all Horovod processes.
             The index of the list corresponds to the rank of each Horovod process.
    """

    if kwargs is None:
        kwargs = {}

    def wrapped_func():
        return func(*args, **kwargs)

    if hosts is not None and hostfile is not None:
        raise ValueError('Argument hosts and hostfile only allow one provided.')

    if use_gloo and use_mpi:
        raise ValueError('Argument use_gloo and use_mpi only allow one set True.')

    hargs = HorovodArgs()

    hargs.np = np
    hargs.hosts = hosts
    hargs.hostfile = hostfile
    hargs.start_timeout = start_timeout
    hargs.ssh_port = ssh_port
    hargs.mpi_args = mpi_args
    hargs.disable_cache = disable_cache
    hargs.output_filename = output_filename
    hargs.verbose = verbose
    hargs.use_gloo = use_gloo
    hargs.use_mpi = use_mpi
    hargs.nics = network_interface
    hargs.run_func = wrapped_func

    return _run(hargs)


if __name__ == '__main__':
    run_commandline()
