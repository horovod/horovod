from packaging import version
import warnings
import horovod.tensorflow as hvd
import tensorflow as tf

class BroadcastGlobalVariablesCallbackImpl(object):
    def __init__(self, backend, root_rank, device='', *args):
        super(BroadcastGlobalVariablesCallbackImpl, self).__init__(*args)
        self.backend = backend
        self.root_rank = root_rank
        self.device = device
        self.broadcast_done = False
        self._local_vars = set()

    def register_local_var(self, var):
        """
        Registers a variable as worker local. Horovod will not perform broadcasting
        operation on this variable.
        """
        if version.parse(tf.__version__) < version.parse('2.0.0'):
            raise RuntimeError('Registering local variables for BroadcastGlobalVariablesCallback is '
                               'not supported in TensorFlow 1.*')

        self._local_vars.add(var.ref())

    def on_batch_end(self, *args):
        if self.broadcast_done:
            return

        with tf.device(self.device):
            if hvd._executing_eagerly() and hasattr(self.model, 'variables'):
                # TensorFlow 2.0 or TensorFlow eager execution
                broadcast_vars = [var for var in self.model.variables if var.ref() not in self._local_vars]
                hvd.broadcast_variables(broadcast_vars, root_rank=self.root_rank)
                hvd.broadcast_variables(self.model.optimizer.variables(), root_rank=self.root_rank)
            else:
                bcast_op = hvd.broadcast_global_variables(self.root_rank)
                self.backend.get_session().run(bcast_op)

        self.broadcast_done = True

class MetricAverageCallbackImpl(object):
    def __init__(self, backend, device='', *args):
        super(MetricAverageCallbackImpl, self).__init__(*args)
        self.backend = backend
        self.device = device
        self.variables = {}
        self.allreduce_ops = {}

        if version.parse("2.3") <= version.parse(tf.__version__) < version.parse("2.5"):
            warnings.warn("Some callbacks may not have access to the averaged metrics, "
                          "see https://github.com/horovod/horovod/issues/2440")

    def _make_variable(self, metric, value):
        with self.backend.name_scope('MetricAverageCallback'):
            var = self.backend.variable(value, name=metric)
            self.backend.get_session().run(var.initializer)
            allreduce_op = hvd.allreduce(var, device_dense=self.device)
            return var, allreduce_op

    def _average_metrics_in_place(self, logs):
        logs = logs or {}
        reduced_logs = {}
        # Reduce every metric among workers. Sort metrics by name to ensure consistent order.
        for metric, value in sorted(logs.items()):
            if hvd._executing_eagerly():
                reduced_logs[metric] = hvd.allreduce(self.backend.constant(value, name=metric)).numpy()
            else:
                if metric not in self.variables:
                    self.variables[metric], self.allreduce_ops[metric] = self._make_variable(metric, value)
                else:
                    self.backend.set_value(self.variables[metric], value)
                reduced_logs[metric] = self.backend.get_session().run(self.allreduce_ops[metric])
        # Override the reduced values back into logs dictionary for other callbacks to use.
        for metric, value in reduced_logs.items():
            logs[metric] = value

    def on_epoch_end(self, epoch, logs=None):
        self._average_metrics_in_place(logs)

class LearningRateScheduleCallbackImpl(object):
    def __init__(self, backend, initial_lr, multiplier, start_epoch=0, end_epoch=None, staircase=True,
                 momentum_correction=True, steps_per_epoch=None, *args):
        super(LearningRateScheduleCallbackImpl, self).__init__(*args)
        self.backend = backend
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.staircase = staircase
        self.momentum_correction = momentum_correction
        self.initial_lr = initial_lr
        self.restore_momentum = None
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = None

        if not callable(multiplier):
            # If multiplier is a constant, it corresponds to exponential decay
            self.multiplier = lambda epoch: multiplier ** (epoch - start_epoch)
        else:
            self.multiplier = multiplier

        if self.initial_lr is None:
            raise ValueError('Parameter `initial_lr` is required')

    def _autodetect_steps_per_epoch(self):
        if self.params.get('steps'):
            return self.params['steps']
        elif self.params.get('samples') and self.params.get('batch_size'):
            return self.params['samples'] // self.params['batch_size']
        else:
            raise ValueError('Could not autodetect the number of steps per epoch. '
                             'Please specify the steps_per_epoch parameter or upgrade to the latest version of Keras.')

    def _adjust_learning_rate(self, epoch):
        old_lr = self.backend.get_value(self.model.optimizer.lr)
        new_lr = self.initial_lr * self.multiplier(epoch)
        self.backend.set_value(self.model.optimizer.lr, new_lr)

        if hasattr(self.model.optimizer, 'momentum') and self.momentum_correction:
            self.restore_momentum = self.backend.get_value(self.model.optimizer.momentum)
            self.backend.set_value(self.model.optimizer.momentum,
                                   self.restore_momentum * new_lr / old_lr)

    def _restore_momentum_if_needed(self):
        if self.restore_momentum:
            self.backend.set_value(self.model.optimizer.momentum, self.restore_momentum)
            self.restore_momentum = None

    def on_train_begin(self, logs=None):
        if self.initial_lr is None:
            self.initial_lr = self.backend.get_value(self.model.optimizer.lr)
        if not self.staircase and not self.steps_per_epoch:
            self.steps_per_epoch = self._autodetect_steps_per_epoch()

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if (self.current_epoch < self.start_epoch or
                (self.end_epoch is not None and self.current_epoch >= self.end_epoch)):
            return

        if self.staircase and batch == 0:
            self._adjust_learning_rate(self.current_epoch)
        elif not self.staircase:
            epoch = self.current_epoch + float(batch) / self.steps_per_epoch
            self._adjust_learning_rate(epoch)

    def on_batch_end(self, batch, logs=None):
        self._restore_momentum_if_needed()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['lr'] = self.backend.get_value(self.model.optimizer.lr)

class LearningRateWarmupCallbackImpl(LearningRateScheduleCallbackImpl):
    def __init__(self, backend, initial_lr, warmup_epochs=5, momentum_correction=True, steps_per_epoch=None,
                 verbose=0, *args):
        def multiplier(epoch):
            epoch += 1. / self.steps_per_epoch
            return 1. / hvd.size() * (epoch * (hvd.size() - 1) / warmup_epochs + 1)
        super(LearningRateWarmupCallbackImpl, self).__init__(
            backend, initial_lr, multiplier, start_epoch=0, end_epoch=warmup_epochs, staircase=False,
            momentum_correction=momentum_correction, steps_per_epoch=steps_per_epoch, *args)
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        super(LearningRateWarmupCallbackImpl, self).on_epoch_end(epoch, logs)

        if epoch == self.end_epoch - 1 and self.verbose > 0 and hvd.rank() == 0:
            new_lr = self.backend.get_value(self.model.optimizer.lr)
            print('\nEpoch %d: finished gradual learning rate warmup to %g.' %
                  (epoch + 1, new_lr))
