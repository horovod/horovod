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
# =============================================================================


class _HorovodArgs(object):
    def __init__(self):
        self.np = 1
        self.check_build = None
        self.ssh_port = None
        self.ssh_identity_file = None
        self.disable_cache = None
        self.start_timeout = None
        self.nic = None
        self.output_filename = None
        self.verbose = None
        self.command = None
        self.run_func = None
        self.config_file = None
        self.nics = None
        self.executable = None

        # tuneable parameter arguments
        self.fusion_threshold_mb = None
        self.cycle_time_ms = None,
        self.cache_capacity = None,

        # hierarchy
        self.hierarchical_allreduce = None
        self.hierarchical_allgather = None

        # autotune arguments
        self.autotune = None
        self.autotune_log_file = None
        self.autotune_warmup_samples = None
        self.autotune_steps_per_sample = None
        self.autotune_bayes_opt_max_samples = None
        self.autotune_gaussian_process_noise = None

        # elastic arguments
        self.min_np = None
        self.max_np = None
        self.slots = None
        self.elastic_timeout = None
        self.reset_limit = None
        self.cooldown_range = None

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
        self.thread_affinity = None
        self.gloo_timeout_seconds = None

        # logging arguments
        self.log_level = None
        self.log_with_timestamp = None
        self.prefix_output_with_timestamp = None

        # host arguments
        self.hosts = None
        self.hostfile = None
        self.host_discovery_script = None

        # controller arguments
        self.use_gloo = None
        self.use_mpi = None
        self.use_jsrun = None


def run(
        func,
        args=(),
        kwargs=None,
        np=1,
        min_np=None,
        max_np=None,
        slots=None,
        reset_limit=None,
        cooldown_range=None,
        hosts=None,
        hostfile=None,
        start_timeout=None,
        ssh_port=None,
        ssh_identity_file=None,
        disable_cache=None,
        output_filename=None,
        verbose=None,
        use_gloo=None,
        use_mpi=None,
        mpi_args=None,
        network_interface=None,
        executable=None):
    """
    Launch a Horovod job to run the specified process function and get the return value.

    :param func: The function to be run in Horovod job processes. The function return value will
                 be collected as the corresponding Horovod process return value.
                 This function must be compatible with pickle.
    :param args: Arguments to pass to `func`.
    :param kwargs: Keyword arguments to pass to `func`.
    :param np: Number of Horovod processes.
    :param min_np: Minimum number of processes running for training to continue. If number of
                   available processes dips below this threshold, then training will wait for
                   more instances to become available. Defaults to np
    :param max_np: Maximum number of training processes, beyond which no additional processes
                   will be created. If not specified, then will be unbounded.
    :param slots: Number of slots for processes per host. Normally 1 slot per GPU per host.
                  If slots are provided by the output of the host discovery script, then that
                  value will override this parameter.
    :param reset_limit: Maximum number of times that the training job can scale up or down the number of workers after
                        which the job is terminated. A reset event occurs when workers are added or removed from the
                        job after the initial registration. So a reset_limit of 0 would mean the job cannot change
                        membership after its initial set of workers. A reset_limit of 1 means it can resize at most
                        once, etc.
    :param cooldown_range: Range of seconds(min, max) a failing host will remain in blacklist.

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
    :param ssh_identity_file: SSH identity (private key) file.
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
    :param executable: Optional executable to run when launching the workers. Defaults to `sys.executable`.
    :return: Return a list which contains values return by all Horovod processes.
             The index of the list corresponds to the rank of each Horovod process.
    """
    from .launch import _run

    if kwargs is None:
        kwargs = {}

    def wrapped_func():
        return func(*args, **kwargs)

    if hosts is not None and hostfile is not None:
        raise ValueError('Argument hosts and hostfile only allow one provided.')

    if use_gloo and use_mpi:
        raise ValueError('Argument use_gloo and use_mpi only allow one set True.')

    hargs = _HorovodArgs()

    hargs.np = np
    hargs.min_np = min_np
    hargs.max_np = max_np
    hargs.slots = slots
    hargs.reset_limit = reset_limit
    hargs.cooldown_range = cooldown_range
    hargs.hosts = hosts
    hargs.hostfile = hostfile
    hargs.start_timeout = start_timeout
    hargs.ssh_port = ssh_port
    hargs.ssh_identity_file = ssh_identity_file
    hargs.mpi_args = mpi_args
    hargs.disable_cache = disable_cache
    hargs.output_filename = output_filename
    hargs.verbose = verbose
    hargs.use_gloo = use_gloo
    hargs.use_mpi = use_mpi
    hargs.nics = network_interface
    hargs.run_func = wrapped_func
    hargs.executable = executable

    return _run(hargs)
