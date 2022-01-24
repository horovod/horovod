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

import argparse
import hashlib
import logging
import io
import os
import sys
import textwrap
import warnings

import yaml

import horovod

from horovod.common.util import (extension_available,
                                 gloo_built, mpi_built,
                                 nccl_built, ddl_built, ccl_built)
from horovod.runner.common.util import config_parser, hosts, safe_shell_exec, secret, timeout
from horovod.runner.common.util import settings as hvd_settings
from horovod.runner.driver import driver_service
from horovod.runner.elastic import settings as elastic_settings
from horovod.runner.elastic import discovery
from horovod.runner.util import cache, threads, network, lsf
from horovod.runner.gloo_run import gloo_run, gloo_run_elastic
from horovod.runner.mpi_run import mpi_run
from horovod.runner.js_run import js_run, is_jsrun_installed
from horovod.runner.http.http_client import read_data_from_kvstore, put_data_into_kvstore
from horovod.runner.http.http_server import KVStoreServer
from horovod.runner.util.remote import get_remote_command

# Cached information of horovod functions be stored in this directory
CACHE_FOLDER = os.path.join(os.path.expanduser('~'), '.horovod')

# Cache entries will be stale if they are older than this number of minutes
CACHE_STALENESS_THRESHOLD_MINUTES = 60

# Number of attempts for sshing into the hosts
SSH_ATTEMPTS = 5

SSH_CONNECT_TIMEOUT_S = 10


@cache.use_cache()
def _check_all_hosts_ssh_successful(host_addresses, ssh_port=None, ssh_identity_file=None):
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
        for i in range(SSH_ATTEMPTS):
            output = io.StringIO()
            try:
                exit_code = safe_shell_exec.execute(command,
                                                    stdout=output,
                                                    stderr=output)
                if exit_code == 0:
                    break
                output_msg = output.getvalue()
            finally:
                output.close()
        return exit_code, output_msg

    args_list = [[get_remote_command(local_command='true',
                                     host=host_address,
                                     port=ssh_port,
                                     identity_file=ssh_identity_file,
                                     timeout_s=SSH_CONNECT_TIMEOUT_S)]
                 for host_address in host_addresses]
    ssh_exit_codes = \
        threads.execute_function_multithreaded(exec_command,
                                               args_list)

    ssh_successful_to_all_hosts = True
    for index, ssh_status in ssh_exit_codes.items():
        exit_code, output_msg = ssh_status[0], ssh_status[1]
        if exit_code != 0:
            print('ssh not successful for host {host}:\n{msg_output}'
                  .format(host=host_addresses[index],
                          msg_output=output_msg))

            ssh_successful_to_all_hosts = False
    if not ssh_successful_to_all_hosts:
        return None  # we could return False here but do not want it to be cached
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
               mxnet=get_check(extension_available('mxnet', verbose=verbose)),
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


def make_deprecated_bool_action(override_args, bool_value, replacement_option):
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
            deprecated_option = '|'.join(self.option_strings)
            warnings.warn(f'Argument {deprecated_option} has been replaced by {replacement_option} and will be removed in v0.21.0',
                          DeprecationWarning)
            override_args.add(self.dest)
            setattr(args, self.dest, self.const)

    return StoreOverrideBoolAction


def parse_args():
    override_args = set()

    parser = argparse.ArgumentParser(description='Horovod Runner')

    parser.add_argument('-v', '--version', action='version', version=horovod.__version__,
                        help='Shows Horovod version.')

    np_arg = parser.add_argument('-np', '--num-proc', action='store', dest='np',
                                 type=int, required=not lsf.LSFUtils.using_lsf(),
                                 help='Total number of training processes. In elastic mode, '
                                      'number of processes required before training can start.')

    parser.add_argument('-cb', '--check-build', action=make_check_build_action(np_arg), nargs=0,
                        help='Shows which frameworks and libraries have been built into Horovod.')

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

    group_ssh = parser.add_argument_group('SSH arguments')
    group_ssh.add_argument('-p', '--ssh-port', action='store', dest='ssh_port',
                           type=int, help='SSH port on all the hosts.')
    group_ssh.add_argument('-i', '--ssh-identity-file', action='store', dest='ssh_identity_file',
                           help='File on the driver from which the identity (private key) is read.')

    group_params = parser.add_argument_group('tuneable parameter arguments')
    group_params.add_argument('--fusion-threshold-mb', action=make_override_action(override_args), type=int,
                              help='Fusion buffer threshold in MB. This is the maximum amount of '
                                   'tensor data that can be fused together into a single batch '
                                   'during allreduce / allgather. Setting 0 disables tensor fusion. '
                                   '(default: 128)')
    group_params.add_argument('--cycle-time-ms', action=make_override_action(override_args), type=float,
                              help='Cycle time in ms. This is the delay between each tensor fusion '
                                   'cycle. The larger the cycle time, the more batching, but the '
                                   'greater latency between each allreduce / allgather operations. '
                                   '(default: 1')
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
                                type=int,
                                help='Number of samples to discard before beginning the optimization process '
                                     'during autotuning. Performance during the first few batches can be '
                                     'affected by initialization and cache warmups. (default: 3')
    group_autotune.add_argument('--autotune-steps-per-sample', action=make_override_action(override_args),
                                type=int,
                                help='Number of steps (approximate) to record before observing a sample. The sample '
                                     'score is defined to be the median score over all batches within the sample. The '
                                     'more batches per sample, the less variance in sample scores, but the longer '
                                     'autotuning will take. (default: 10')
    group_autotune.add_argument('--autotune-bayes-opt-max-samples', action=make_override_action(override_args),
                                type=int,
                                help='Maximum number of samples to collect for each Bayesian optimization process. '
                                     '(default: 20')
    group_autotune.add_argument('--autotune-gaussian-process-noise', action=make_override_action(override_args),
                                type=float,
                                help='Regularization value [0, 1] applied to account for noise in samples. '
                                     '(default: 0.8')

    group_elastic = parser.add_argument_group('elastic arguments')
    group_elastic.add_argument('--min-np', action='store', dest='min_np', type=int,
                               help='Minimum number of processes running for training to continue. If number of '
                                    'available processes dips below this threshold, then training will wait for '
                                    'more instances to become available. Defaults to --num-proc.')
    group_elastic.add_argument('--max-np', action='store', dest='max_np', type=int,
                               help='Maximum number of training processes, beyond which no additional '
                                    'processes will be created. If not specified, then will be unbounded.')
    group_elastic.add_argument('--slots-per-host', action='store', dest='slots', type=int,
                               help='Number of slots for processes per host. Normally 1 slot per GPU per host. '
                                    'If slots are provided by the output of the host discovery script, then '
                                    'that value will override this parameter.')
    group_elastic.add_argument('--elastic-timeout', action='store', dest='elastic_timeout', type=int,
                               help='Timeout for elastic initialisation after re-scaling the cluster. '
                                    'The default value is 600 seconds. Alternatively, '
                                    'The environment variable HOROVOD_ELASTIC_TIMEOUT '
                                    'can also be used to.')
    group_elastic.add_argument('--reset-limit', action='store', dest='reset_limit', type=int,
                               help='Maximum number of times that the training job can scale up or down '
                                    'the number of workers after which the job is terminated. (default: None)')
    group_elastic.add_argument('--blacklist-cooldown-range', action='store', dest='cooldown_range', type=int, nargs=2,
                               help='Range of seconds(min, max) a failing host will remain in blacklist. (default: None)')

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
                                   type=int,
                                   help='Seconds until the stall warning is logged to stderr. (default: 60')
    group_stall_check.add_argument('--stall-check-shutdown-time-seconds', action=make_override_action(override_args),
                                   type=int,
                                   help='Seconds until Horovod is shutdown due to stall. Shutdown will only take '
                                        'place if this value is greater than the warning time. (default: 0')

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
                                       type=int,
                                       help='Number of NCCL streams. Only applies when running with NCCL support. '
                                            '(default: %(default)s)')
    group_library_options.add_argument('--thread-affinity', action=make_override_action(override_args),
                                       type=int,
                                       help='Horovod background thread affinity. '
                                            '(default: 0')
    group_library_options.add_argument('--gloo-timeout-seconds', action=make_override_action(override_args),
                                       type=int,
                                       help='Timeout in seconds for Gloo operations to complete. '
                                            '(default: 30')

    group_logging = parser.add_argument_group('logging arguments')
    group_logging.add_argument('--log-level', action=make_override_action(override_args),
                               choices=config_parser.LOG_LEVELS,
                               help='Minimum level to log to stderr from the Horovod backend. (default: WARNING).')
    group_logging_timestamp = group_logging.add_mutually_exclusive_group()
    group_logging_timestamp.add_argument('--log-with-timestamp',
                                         action=make_override_true_action(override_args),
                                         help=argparse.SUPPRESS)
    group_logging_timestamp.add_argument('--log-without-timestamp', dest='log_with_timestamp',
                                         action=make_override_false_action(override_args),
                                         help='Hide the timestamp from Horovod internal log messages.')
    group_logging_timestamp.add_argument('-prefix-timestamp', '--prefix-output-with-timestamp', action='store_true',
                                         dest='prefix_output_with_timestamp',
                                         help='Timestamp each line of output to stdout, stderr, and stddiag.')
    group_logging_timestamp.add_argument('--log-hide-timestamp',
                                         dest='log_with_timestamp',
                                         action=make_deprecated_bool_action(override_args, False, '--log-without-timestamp'),
                                         help=argparse.SUPPRESS)
    group_logging_timestamp.add_argument('--no-log-hide-timestamp',
                                         dest='log_with_timestamp',
                                         action=make_deprecated_bool_action(override_args, True, '--log-with-timestamp'),
                                         help=argparse.SUPPRESS)

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
    group_hosts.add_argument('--host-discovery-script', action=make_override_action(override_args),
                             help='Used for elastic training (autoscaling and fault tolerance). '
                                  'An executable script that will print to stdout every available host (one per '
                                  'newline character) that can be used to run worker processes. Optionally '
                                  'specifies number of slots on the same line as the hostname as: "hostname:slots".'
                                  'Providing a discovery script enables elastic training (see elastic arguments).'
                                  'The job will fail immediately if execution of the script returns a non-zero exit '
                                  'code on the first call. Subsequent calls will be retried until timeout.')

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

    args.run_func = None
    args.executable = None

    if args.check_build:
        check_build(args.verbose)

    return args


def _run_static(args):
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
                                     ssh_identity_file=args.ssh_identity_file,
                                     extra_mpi_args=args.mpi_args,
                                     tcp_flag=args.tcp_flag,
                                     binding_args=args.binding_args,
                                     key=secret.make_secret_key(),
                                     start_timeout=tmout,
                                     num_proc=args.np,
                                     hosts=args.hosts,
                                     output_filename=args.output_filename,
                                     run_func_mode=args.run_func is not None,
                                     nics=args.nics,
                                     prefix_output_with_timestamp=args.prefix_output_with_timestamp)

    # This cache stores the results of checks performed by horovod
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
        if args.ssh_identity_file:
            params += args.ssh_identity_file
        parameters_hash = hashlib.md5(params.encode('utf-8')).hexdigest()
        fn_cache = cache.Cache(CACHE_FOLDER, CACHE_STALENESS_THRESHOLD_MINUTES,
                               parameters_hash)

    all_host_names, _ = hosts.parse_hosts_and_slots(args.hosts)
    if settings.verbose >= 2:
        print('Filtering local host names.')
    remote_host_names = network.filter_local_addresses(all_host_names)
    if settings.verbose >= 2:
        print('Remote host found: ' + ' '.join(remote_host_names))

    if len(remote_host_names) > 0:
        if settings.verbose >= 2:
            print('Checking ssh on all remote hosts.')
        # Check if we can ssh into all remote hosts successfully.
        if not _check_all_hosts_ssh_successful(remote_host_names, args.ssh_port, args.ssh_identity_file,
                                               fn_cache=fn_cache):
            raise RuntimeError('could not connect to some hosts via ssh')
        if settings.verbose >= 2:
            print('SSH was successful into all the remote hosts.')

    nics = driver_service.get_common_interfaces(settings, all_host_names,
                                                remote_host_names, fn_cache)

    if args.run_func:
        # get the driver IPv4 address
        driver_ip = network.get_driver_ip(nics)
        run_func_server = KVStoreServer(verbose=settings.verbose)
        run_func_server_port = run_func_server.start_server()
        put_data_into_kvstore(driver_ip, run_func_server_port,
                              'runfunc', 'func', args.run_func)

        executable = args.executable or sys.executable
        command = [executable, '-m', 'horovod.runner.run_task', str(driver_ip), str(run_func_server_port)]

        try:
            _launch_job(args, settings, nics, command)
            results = [None] * args.np
            # TODO: make it parallel to improve performance
            for i in range(args.np):
                results[i] = read_data_from_kvstore(driver_ip, run_func_server_port,
                                                    'runfunc_result', str(i))
            return results
        finally:
            run_func_server.shutdown_server()
    else:
        command = args.command
        _launch_job(args, settings, nics, command)
        return None


def _run_elastic(args):
    # construct host discovery component
    if args.host_discovery_script:
        discover_hosts = discovery.HostDiscoveryScript(args.host_discovery_script, args.slots)
    elif args.hosts:
        _, available_host_slots = hosts.parse_hosts_and_slots(args.hosts)
        if len(available_host_slots) < 2:
            raise ValueError('Cannot run in fault tolerance mode with fewer than 2 hosts.')
        discover_hosts = discovery.FixedHosts(available_host_slots)
    else:
        raise ValueError('One of --host-discovery-script, --hosts, or --hostnames must be provided')

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
    settings = elastic_settings.ElasticSettings(discovery=discover_hosts,
                                                min_np=args.min_np or args.np,
                                                max_np=args.max_np,
                                                elastic_timeout=args.elastic_timeout,
                                                reset_limit=args.reset_limit,
                                                cooldown_range=args.cooldown_range,
                                                num_proc=args.np,
                                                verbose=2 if args.verbose else 0,
                                                ssh_port=args.ssh_port,
                                                ssh_identity_file=args.ssh_identity_file,
                                                extra_mpi_args=args.mpi_args,
                                                key=secret.make_secret_key(),
                                                start_timeout=tmout,
                                                output_filename=args.output_filename,
                                                run_func_mode=args.run_func is not None,
                                                nics=args.nics,
                                                prefix_output_with_timestamp=args.prefix_output_with_timestamp)

    if not gloo_built(verbose=(settings.verbose >= 2)):
        raise ValueError('Gloo support is required to use elastic training, but has not been built.  Ensure CMake is '
                         'installed and reinstall Horovod with HOROVOD_WITH_GLOO=1 to debug the build error.')

    env = os.environ.copy()
    config_parser.set_env_from_args(env, args)
    gloo_run_elastic(settings, env, args.command)


def is_gloo_used(use_gloo=None, use_mpi=None, use_jsrun=None):
    # determines whether run_controller will run gloo
    # for the given (use_gloo, _, use_mpi, _, use_jsrun, _, _)
    return use_gloo or (not use_mpi and not use_jsrun and not mpi_built())


def run_controller(use_gloo, gloo_run, use_mpi, mpi_run, use_jsrun, js_run, verbosity):
    # keep logic in sync with is_gloo_used(...)
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
            raise ValueError(
                'Horovod did not detect an LSF job.  The jsrun launcher can only be used in that environment. '
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


def _is_elastic(args):
    return args.host_discovery_script is not None or args.min_np is not None


def _launch_job(args, settings, nics, command):
    env = os.environ.copy()
    config_parser.set_env_from_args(env, args)

    def gloo_run_fn():
        driver_ip = network.get_driver_ip(nics)
        gloo_run(settings, nics, env, driver_ip, command)

    def mpi_run_fn():
        mpi_run(settings, nics, env, command)

    def js_run_fn():
        js_run(settings, nics, env, command)

    run_controller(args.use_gloo, gloo_run_fn,
                   args.use_mpi, mpi_run_fn,
                   args.use_jsrun, js_run_fn,
                   args.verbose)


def _run(args):
    # If LSF is used, use default values from job config
    if lsf.LSFUtils.using_lsf():
        if not args.np:
            args.np = lsf.LSFUtils.get_num_processes()
        if not args.hosts and not args.hostfile and not args.host_discovery_script:
            args.hosts = ','.join('{host}:{np}'.format(host=host, np=lsf.LSFUtils.get_num_gpus())
                                  for host in lsf.LSFUtils.get_compute_hosts())

    # if hosts are not specified, either parse from hostfile, or default as
    # localhost
    if not args.hosts and not args.host_discovery_script:
        if args.hostfile:
            args.hosts = hosts.parse_host_files(args.hostfile)
        else:
            # Set hosts to localhost if not specified
            args.hosts = 'localhost:{np}'.format(np=args.np)

    # Convert nics into set
    args.nics = set(args.nics.split(',')) if args.nics else None

    if _is_elastic(args):
        return _run_elastic(args)
    else:
        return _run_static(args)


def run_commandline():
    args = parse_args()

    if args.log_level:
        logging.addLevelName(logging.NOTSET, 'TRACE')
        logging.basicConfig(level=logging.getLevelName(args.log_level))

    _run(args)


if __name__ == '__main__':
    run_commandline()
