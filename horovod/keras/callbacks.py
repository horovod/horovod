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


class LRWarmupCallback(keras.callbacks.Callback):
    """
    Implements gradual learning rate warmup described in the paper
    "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour".
    See https://arxiv.org/pdf/1706.02677.pdf for details.

    Math recap:
                                                 batch
        epoch               = full_epochs + ---------------
                                            steps_per_epoch
                                    size - 1
        lr'(epoch)          = lr * (-------- * epoch + 1)
                                     warmup

        lr'(epoch = 0)      = lr
        lr'(epoch = warmup) = hvd.size() * lr
    """

    def __init__(self, warmup_epochs=5, momentum_correction=True, steps_per_epoch=None,
                 verbose=0):
        """
        Construct a new LRWarmupCallback that will gradually warmup learning rate.

        Args:
            warmup_epochs: The number of epochs of the warmup phase. Defaults to 5.
            momentum_correction: Apply momentum correction to optimizers that have momentum.
                                 Defaults to True.
            steps_per_epoch: The callback will attempt to autodetect number of batches per
                             epoch with Keras >= 2.0.0. Provide this value if you have an older
                             version of Keras.
            verbose: verbosity mode, 0 or 1.
        """
        super(LRWarmupCallback, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.momentum_correction = momentum_correction
        self.initial_lr = None
        self.restore_momentum = None
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose
        self.current_epoch = None

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
                             'LRWarmupCallback() or upgrade to the latest version of Keras.')

    def on_train_begin(self, logs=None):
        self.initial_lr = K.get_value(self.model.optimizer.lr)
        if not self.steps_per_epoch:
            self.steps_per_epoch = self._autodetect_steps_per_epoch()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if self.current_epoch > self.warmup_epochs:
            # Outside of adjustment scope.
            return

        if self.current_epoch == self.warmup_epochs and batch > 0:
            # Outside of adjustment scope, final adjustment is done on first batch.
            return

        old_lr = K.get_value(self.model.optimizer.lr)
        epoch = self.current_epoch + float(batch) / self.steps_per_epoch
        new_lr = self.initial_lr * \
            (epoch * (hvd.size() - 1) / self.warmup_epochs + 1)
        K.set_value(self.model.optimizer.lr, new_lr)

        if self.current_epoch == self.warmup_epochs and self.verbose:
            print('Epoch %d: finished gradual learning rate warmup to %s.' %
                  (epoch + 1, new_lr))

        if hasattr(self.model.optimizer, 'momentum') and self.momentum_correction:
            # See the paper cited above for more information about momentum correction.
            self.restore_momentum = K.get_value(self.model.optimizer.momentum)
            K.set_value(self.model.optimizer.momentum,
                        self.restore_momentum * new_lr / old_lr)

    def on_batch_end(self, batch, logs=None):
        if self.restore_momentum:
            K.set_value(self.model.optimizer.momentum, self.restore_momentum)
            self.restore_momentum = None
