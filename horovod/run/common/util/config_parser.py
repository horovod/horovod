# Parameter knobs
HOROVOD_FUSION_THRESHOLD = 'HOROVOD_FUSION_THRESHOLD'
HOROVOD_CYCLE_TIME = 'HOROVOD_CYCLE_TIME'
HOROVOD_CACHE_CAPACITY = 'HOROVOD_CACHE_CAPACITY'
HOROVOD_HIERARCHICAL_ALLREDUCE = 'HOROVOD_HIERARCHICAL_ALLREDUCE'
HOROVOD_HIERARCHICAL_ALLGATHER = 'HOROVOD_HIERARCHICAL_ALLGATHER'

# Autotune knobs
HOROVOD_AUTOTUNE = 'HOROVOD_AUTOTUNE'
HOROVOD_AUTOTUNE_LOG = 'HOROVOD_AUTOTUNE_LOG'
HOROVOD_AUTOTUNE_WARMUP_SAMPLES = 'HOROVOD_AUTOTUNE_WARMUP_SAMPLES'
HOROVOD_AUTOTUNE_CYCLES_PER_SAMPLE = 'HOROVOD_AUTOTUNE_CYCLES_PER_SAMPLE'
HOROVOD_AUTOTUNE_BAYES_OPT_MAX_SAMPLES = 'HOROVOD_AUTOTUNE_BAYES_OPT_MAX_SAMPLES'
HOROVOD_AUTOTUNE_GAUSSIAN_PROCESS_NOISE = 'HOROVOD_AUTOTUNE_GAUSSIAN_PROCESS_NOISE'

# Timeline knobs
HOROVOD_TIMELINE = 'HOROVOD_TIMELINE'
HOROVOD_TIMELINE_MARK_CYCLES = 'HOROVOD_TIMELINE_MARK_CYCLES'

# Stall check knobs
HOROVOD_STALL_CHECK_DISABLE = 'HOROVOD_STALL_CHECK_DISABLE'
HOROVOD_STALL_CHECK_TIME_SECONDS = 'HOROVOD_STALL_CHECK_TIME_SECONDS'
HOROVOD_STALL_SHUTDOWN_TIME_SECONDS = 'HOROVOD_STALL_SHUTDOWN_TIME_SECONDS'

# Library options knobs
HOROVOD_MPI_THREADS_DISABLE = 'HOROVOD_MPI_THREADS_DISABLE'
HOROVOD_NUM_NCCL_STREAMS = 'HOROVOD_NUM_NCCL_STREAMS'
HOROVOD_MLSL_BGT_AFFINITY = 'HOROVOD_MLSL_BGT_AFFINITY'


def _set_arg_from_config(args, arg_base_name, config, arg_prefix=''):
    value = config.get(arg_base_name)
    if value is not None:
        setattr(args, arg_prefix + arg_base_name, value)


def set_args_from_config(args, config):
    # Controller
    controller = config.get('controller')
    if controller:
        if controller.lower() == 'gloo':
            args.use_gloo = True
        elif controller.lower() == 'mpi':
            args.use_mpi = True
        else:
            raise ValueError('No such controller supported: {}'.format(controller))

    # Params
    params = config.get('params')
    if params:
        _set_arg_from_config(args, 'fusion_threshold_mb', params)
        _set_arg_from_config(args, 'cycle_time_ms', params)
        _set_arg_from_config(args, 'cache_capacity', params)
        _set_arg_from_config(args, 'hierarchical_allreduce', params)
        _set_arg_from_config(args, 'hierarchical_allgather', params)

    # Autotune
    autotune = config.get('autotune')
    if autotune:
        _set_arg_from_config(args, 'autotune', autotune)
        _set_arg_from_config(args, 'log_file', autotune, arg_prefix='autotune_')
        _set_arg_from_config(args, 'warmup_samples', autotune, arg_prefix='autotune_')
        _set_arg_from_config(args, 'batches_per_sample', autotune, arg_prefix='autotune_')
        _set_arg_from_config(args, 'bayes_opt_max_samples', autotune, arg_prefix='autotune_')
        _set_arg_from_config(args, 'gaussian_process_noise', autotune, arg_prefix='autotune_')

    # Timeline
    timeline = config.get('timeline')
    if timeline and timeline.get('enabled'):
        _set_arg_from_config(args, 'filename', timeline, arg_prefix='timeline_')
        _set_arg_from_config(args, 'mark_cycles', timeline, arg_prefix='timeline_')

    # Stall Check
    stall_check = config.get('stall_check')
    if stall_check:
        args.stall_check_disable = not stall_check.get('enabled', True)
        _set_arg_from_config(args, 'stall_check_warning_time_seconds', stall_check,
                             arg_prefix='stall_check_')
        _set_arg_from_config(args, 'stall_check_shutdown_time_seconds', stall_check,
                             arg_prefix='stall_check_')

    # Library Options
    library_options = config.get('library_options')
    if library_options:
        _set_arg_from_config(args, 'mpi_threads_enabled', library_options)
        _set_arg_from_config(args, 'num_nccl_streams', library_options)
        _set_arg_from_config(args, 'mlsl_bgt_affinity', library_options)


def validate_config_args(args):
    if args.fusion_threshold_mb < 0:
        raise ValueError('fusion_threshold_mb {} must be > 0'.format(args.fusion_threshold_mb))


def _add_arg_to_env(env, env_key, arg_value, transform_fn=None):
    if arg_value is not None:
        value = arg_value
        if transform_fn:
            value = transform_fn(value)
        env[env_key] = value


def set_env_from_args(env, args):
    def identity(value):
        return 1 if value else 0

    def complement(value):
        return 0 if value else 1

    # Params
    _add_arg_to_env(env, HOROVOD_FUSION_THRESHOLD, args.fusion_threshold_mb, lambda v: v * 1024 * 1024)
    _add_arg_to_env(env, HOROVOD_CYCLE_TIME, args.cycle_time_ms)
    _add_arg_to_env(env, HOROVOD_CACHE_CAPACITY, args.cache_capacity)
    _add_arg_to_env(env, HOROVOD_HIERARCHICAL_ALLREDUCE, args.hierarchical_allreduce, identity)
    _add_arg_to_env(env, HOROVOD_HIERARCHICAL_ALLGATHER, args.hierarchical_allgather, identity)

    # Autotune
    if args.autotune:
        _add_arg_to_env(env, HOROVOD_AUTOTUNE, args.autotune, identity)
        _add_arg_to_env(env, HOROVOD_AUTOTUNE_LOG, args.autotune_log_file)
        _add_arg_to_env(env, HOROVOD_AUTOTUNE_WARMUP_SAMPLES, args.autotune_warmup_samples)
        _add_arg_to_env(env, HOROVOD_AUTOTUNE_CYCLES_PER_SAMPLE, args.autotune_batches_per_sample)
        _add_arg_to_env(env, HOROVOD_AUTOTUNE_BAYES_OPT_MAX_SAMPLES, args.autotune_bayes_opt_max_samples)
        _add_arg_to_env(env, HOROVOD_AUTOTUNE_GAUSSIAN_PROCESS_NOISE, args.autotune_gaussian_process_noise)

    # Timeline
    if args.timeline_filename:
        _add_arg_to_env(env, HOROVOD_TIMELINE, args.timeline_filename)
        _add_arg_to_env(env, HOROVOD_TIMELINE_MARK_CYCLES, args.timeline_mark_cycles)

    # Stall Check
    _add_arg_to_env(env, HOROVOD_STALL_CHECK_DISABLE, args.stall_check_enabled, complement)
    _add_arg_to_env(env, HOROVOD_STALL_CHECK_TIME_SECONDS, args.stall_check_warning_time_seconds)
    _add_arg_to_env(env, HOROVOD_STALL_SHUTDOWN_TIME_SECONDS, args.stall_check_shutdown_time_seconds)

    # Library Options
    _add_arg_to_env(env, HOROVOD_MPI_THREADS_DISABLE, args.mpi_threads_enabled, complement)
    _add_arg_to_env(env, HOROVOD_NUM_NCCL_STREAMS, args.num_nccl_streams)
    _add_arg_to_env(env, HOROVOD_MLSL_BGT_AFFINITY, args.mlsl_bgt_affinity)

    return env
