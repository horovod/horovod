import keras
import keras.backend as K
import tensorflow as tf

import horovod.tensorflow as hvd


class BroadcastGlobalVariablesCallback(keras.callbacks.Callback):
    """
    Keras Callback that will broadcast all global variables from root rank
    to all other processes during initialization.

    This is necessary to ensure consistent initialization of all workers when
    training is started with random weights or restored from a checkpoint.
    """

    def __init__(self, root_rank, device=''):
        """
        Construct a new BroadcastGlobalVariablesCallback that will broadcast all
        global variables from root rank to all other processes during initialization.

        Args:
            root_rank: Rank that will send data, other ranks will receive data.
            device: Device to be used for broadcasting. Uses GPU by default
                    if Horovod was build with HOROVOD_GPU_BROADCAST.
        """
        super(BroadcastGlobalVariablesCallback, self).__init__()
        self.root_rank = root_rank
        self.device = device

    def on_train_begin(self, logs=None):
        with tf.device(self.device):
            bcast_op = hvd.broadcast_global_variables(self.root_rank)
            K.get_session().run(bcast_op)


class MetricAverageCallback(keras.callbacks.Callback):
    """
    Keras Callback that will average metrics across all processes at the
    end of the epoch. Useful in conjuction with ReduceLROnPlateau,
    TensorBoard and other metrics-based callbacks.

    Note: This callback must be added to the callback list before the
    ReduceLROnPlateau, TensorBoard or other metrics-based callbacks.
    """

    def __init__(self, device=''):
        """
        Construct a new MetricAverageCallback that will average metrics
        across all processes at the end of the epoch.

        Args:
            device: Device to be used for allreduce. Uses GPU by default
                    if Horovod was build with HOROVOD_GPU_ALLREDUCE.
        """
        super(MetricAverageCallback, self).__init__()
        self.variables = {}
        self.allreduce_ops = {}
        self.device = device

    def _make_variable(self, metric, value):
        with tf.name_scope('MetricAverageCallback'):
            var = tf.Variable(value, name=metric)
            K.get_session().run(var.initializer)
            allreduce_op = hvd.allreduce(var, device_dense=self.device)
            return var, allreduce_op

    def _average_metrics_in_place(self, logs):
        logs = logs or {}
        reduced_logs = {}
        # Reduce every metric among workers. Sort metrics by name
        # to ensure consistent order.
        for metric, value in sorted(logs.items()):
            if metric not in self.variables:
                self.variables[metric], self.allreduce_ops[metric] = \
                    self._make_variable(metric, value)
            else:
                K.set_value(self.variables[metric], value)
            reduced_logs[metric] = \
                K.get_session().run(self.allreduce_ops[metric])
        # Override the reduced values back into logs dictionary
        # for other callbacks to use.
        for metric, value in reduced_logs.items():
            logs[metric] = value

    def on_epoch_end(self, epoch, logs=None):
        self._average_metrics_in_place(logs)


class LearningRateScheduleCallback(keras.callbacks.Callback):
    """
    LearningRateScheduleCallback sets learning rate between epochs `start_epoch` and
    `end_epoch` to be `initial_lr * multiplier`.  `multiplier` can be a constant or
    a function `f(epoch) = lr'`.

    If `multiplier` is a function and `staircase=True`, learning rate adjustment will
    happen at the beginning of each epoch and the epoch passed to the `multiplier`
    function will be an integer.

    If `multiplier` is a function and `staircase=False`, learning rate adjustment will
    happen at the beginning of each batch and the epoch passed to the `multiplier`
    function will be a floating number: `epoch' = epoch + batch / steps_per_epoch`.
    This functionality is useful for smooth learning rate adjustment schedulers, such
    as `LearningRateWarmupCallback`.

    `initial_lr` is the learning rate of the model optimizer at the start of the training.
    """

    def __init__(self, multiplier, start_epoch=0, end_epoch=None, staircase=True,
                 momentum_correction=True, steps_per_epoch=None):
        """
        Construct a new LearningRateScheduleCallback.

        Args:
            multiplier: A constant multiplier or a function `f(epoch) = lr'`
            start_epoch: The first epoch this adjustment will be applied to. Defaults to 0.
            end_epoch: The epoch this adjustment will stop applying (exclusive end).
                       Defaults to None.
            staircase: Whether to adjust learning rate at the start of epoch (`staircase=True`)
                       or at the start of every batch (`staircase=False`).
            momentum_correction: Apply momentum correction to optimizers that have momentum.
                                 Defaults to True.
            steps_per_epoch: The callback will attempt to autodetect number of batches per
                             epoch with Keras >= 2.0.0. Provide this value if you have an older
                             version of Keras.
        """
        super(LearningRateScheduleCallback, self).__init__()
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.staircase = staircase
        self.momentum_correction = momentum_correction
        self.initial_lr = None
        self.restore_momentum = None
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = None

        if not callable(multiplier):
            self.staircase = True
            self.multiplier = lambda epoch: multiplier
        else:
            self.multiplier = multiplier

    def _autodetect_steps_per_epoch(self):
        if self.params.get('steps'):
            # The number of steps is provided in the parameters.
            return self.params['steps']
        elif self.params.get('samples') and self.params.get('batch_size'):
            # Compute the number of steps per epoch using # of samples and a batch size.
            return self.params['samples'] // self.params['batch_size']
        else:
            raise ValueError('Could not autodetect the number of steps per epoch. '
                             'Please specify the steps_per_epoch parameter to the '
                             '%s() or upgrade to the latest version of Keras.'
                             % self.__class__.__name__)

    def _adjust_learning_rate(self, epoch):
        old_lr = K.get_value(self.model.optimizer.lr)
        new_lr = self.initial_lr * self.multiplier(epoch)
        K.set_value(self.model.optimizer.lr, new_lr)

        if hasattr(self.model.optimizer, 'momentum') and self.momentum_correction:
            # See the paper cited above for more information about momentum correction.
            self.restore_momentum = K.get_value(self.model.optimizer.momentum)
            K.set_value(self.model.optimizer.momentum,
                        self.restore_momentum * new_lr / old_lr)

    def _restore_momentum_if_needed(self):
        if self.restore_momentum:
            K.set_value(self.model.optimizer.momentum, self.restore_momentum)
            self.restore_momentum = None

    def on_train_begin(self, logs=None):
        self.initial_lr = K.get_value(self.model.optimizer.lr)
        if not self.staircase and not self.steps_per_epoch:
            self.steps_per_epoch = self._autodetect_steps_per_epoch()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if (self.current_epoch < self.start_epoch or
                (self.end_epoch is not None and self.current_epoch >= self.end_epoch)):
            # Outside of the adjustment scope.
            return

        if self.staircase and batch == 0:
            # Do on first batch of every epoch.
            self._adjust_learning_rate(self.current_epoch)
        elif not self.staircase:
            epoch = self.current_epoch + float(batch) / self.steps_per_epoch
            self._adjust_learning_rate(epoch)

    def on_batch_end(self, batch, logs=None):
        self._restore_momentum_if_needed()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Log current learning rate.
            logs['lr'] = K.get_value(self.model.optimizer.lr)


class LearningRateWarmupCallback(LearningRateScheduleCallback):
    """
    Implements gradual learning rate warmup:

        `lr = initial_lr / hvd.size()` ---> `lr = initial_lr`

    `initial_lr` is the learning rate of the model optimizer at the start of the training.

    This technique was described in the paper "Accurate, Large Minibatch SGD: Training
    ImageNet in 1 Hour". See https://arxiv.org/pdf/1706.02677.pdf for details.

    Math recap:
                                                 batch
        epoch               = full_epochs + ---------------
                                            steps_per_epoch

                               lr     size - 1
        lr'(epoch)          = ---- * (-------- * epoch + 1)
                              size     warmup

                               lr
        lr'(epoch = 0)      = ----
                              size

        lr'(epoch = warmup) = lr
    """

    def __init__(self, warmup_epochs=5, momentum_correction=True, steps_per_epoch=None,
                 verbose=0):
        """
        Construct a new LearningRateWarmupCallback that will gradually warm up the learning rate.

        Args:
            warmup_epochs: The number of epochs of the warmup phase. Defaults to 5.
            momentum_correction: Apply momentum correction to optimizers that have momentum.
                                 Defaults to True.
            steps_per_epoch: The callback will attempt to autodetect number of batches per
                             epoch with Keras >= 2.0.0. Provide this value if you have an older
                             version of Keras.
            verbose: verbosity mode, 0 or 1.
        """
        def multiplier(epoch):
            # Adjust epoch to produce round numbers at the end of each epoch, so that TensorBoard
            # learning rate graphs look better.
            epoch += 1. / self.steps_per_epoch
            return 1. / hvd.size() * (epoch * (hvd.size() - 1) / warmup_epochs + 1)
        self.verbose = verbose
        super(LearningRateWarmupCallback, self).__init__(
            multiplier, start_epoch=0, end_epoch=warmup_epochs, staircase=False,
            momentum_correction=momentum_correction, steps_per_epoch=steps_per_epoch)

    def on_epoch_end(self, epoch, logs=None):
        super(LearningRateWarmupCallback, self).on_epoch_end(epoch, logs)

        if epoch == self.end_epoch - 1 and self.verbose > 0:
            new_lr = K.get_value(self.model.optimizer.lr)
            print('\nEpoch %d: finished gradual learning rate warmup to %g.' %
                  (epoch + 1, new_lr))
