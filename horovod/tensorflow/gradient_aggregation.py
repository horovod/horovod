import tensorflow as tf


def apply_op_to_not_none_tensors(tensor_op, tensors, *args):
    return [
        tensor_op(
            tensor,
            *args
        ) if tensor is not None else tensor for tensor in tensors]


def get_not_none_from_list(tensor_list):
    return [x for x in tensor_list if x is not None]


class LocalGradientAggregationHelper:
    """
    LocalGradientAggregationHelper aggregates gradient updates locally,
    and communicates the updates across machines only once every
    backward_passes_per_step. Only supports graph mode execution.
    """

    _OPTIMIZER_TYPE_KERAS = "optimizer_type_keras"
    _OPTIMIZER_TYPE_LEGACY = "optimizer_type_legacy"

    def __init__(
            self,
            backward_passes_per_step,
            allreduce_func,
            sparse_as_dense,
            average_aggregated_gradients,
            rank,
            optimizer_type):
        self._allreduce_grads = allreduce_func

        # backward_passes_per_step controls how often gradient updates are
        # synchronized.
        self.backward_passes_per_step = backward_passes_per_step
        if self.backward_passes_per_step <= 0:
            raise ValueError("backward_passes_per_step must be > 0")

        # average_aggregated_gradients controls whether gradient updates that are
        # aggregated, should be divided by `backward_passes_per_step`.
        self.average_aggregated_gradients = average_aggregated_gradients

        # This is going to be [N] data structure holding the aggregated gradient updates
        # N is the number of parameters.
        self.locally_aggregated_grads = []

        # Used to know when to allreduce and apply gradients. We allreduce when `self.counter`
        # is equal to `self.backward_passes_per_step`. We apply gradients when `self.counter` is
        # equal to 0.
        self.counter = None

        self.sparse_as_dense = sparse_as_dense
        self.rank = rank
        self.optimizer_type = optimizer_type

        # Contains the mapping of indexes of grad updates that are not None to their index in
        # locally_aggregated_grads which only contains not None gradients. When performing
        # gradient aggregation we have to remove them from the list of grads prior to passing
        # the list into a tf.cond().
        self.not_none_indexes = {}
        self.num_none_grad_updates = 0

    def _maybe_convert_grad(self, grad):
        # Handle IndexedSlices.
        if isinstance(grad, tf.IndexedSlices):
            if self.sparse_as_dense:
                return tf.convert_to_tensor(grad)
            else:
                raise ValueError(
                    "IndexedSlices are not supported when "
                    "`backward_passes_per_step` > 1 and "
                    "`sparse_as_dense` is False."
                )

        return grad

    def _init_aggregation_vars(self, grads):
        """
        Initializes the counter that is used when to communicate and aggregate gradients
        and the tensorflow variables that store the locally aggregated gradients.
        """
        variable_scope_name = "aggregation_variables_" + str(self.rank)
        with tf.compat.v1.variable_scope(variable_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            self.counter = tf.compat.v1.get_variable(
                "aggregation_counter", shape=(), dtype=tf.int32,
                trainable=False, initializer=tf.compat.v1.zeros_initializer(),
                collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES],
            )
            for idx, grad in enumerate(grads):
                grad = self._maybe_convert_grad(grad)

                # Handle grads that are None.
                if grad is None:
                    self.num_none_grad_updates += 1
                    continue
                self.not_none_indexes[idx] = len(self.locally_aggregated_grads)

                # Create shadow variable.
                grad_aggregation_variable_name = str(idx)
                zero_grad = tf.zeros(shape=grad.get_shape().as_list(), dtype=grad.dtype)
                grad_aggregation_variable = tf.compat.v1.get_variable(
                    grad_aggregation_variable_name,
                    trainable=False,
                    initializer=zero_grad,
                    collections=[
                        tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
                        "aggregating_collection"],
                )
                self.locally_aggregated_grads.append(grad_aggregation_variable)
            assert len(self.locally_aggregated_grads) + \
                self.num_none_grad_updates == len(grads)

        # We expect to get a `sess` when we need to manually do a `sess.run(...)`
        # for the variables to be initialized. This is the `tf.keras`
        # optimizers.
        if self.optimizer_type == self._OPTIMIZER_TYPE_KERAS:
            session = tf.compat.v1.keras.backend.get_session(op_input_list=())
            vars_init_op = tf.compat.v1.variables_initializer(
                [self.counter, *get_not_none_from_list(self.locally_aggregated_grads)]
            )
            session.run(vars_init_op)

    def _clear_grads(self):
        clear_ops_list = []
        for idx, grad_aggregator in enumerate(self.locally_aggregated_grads):
            clear_op = grad_aggregator.assign(grad_aggregator.initial_value)
            clear_ops_list.append(clear_op)
        return tf.group(*clear_ops_list)

    def _aggregate_grads(self, grads):
        aggregation_ops_list = []
        grads = get_not_none_from_list(grads)
        assert len(grads) == len(self.locally_aggregated_grads)

        # Apply new gradient updates to the local copy.
        for idx, grad in enumerate(grads):
            if self.sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)

            updated_grad_aggregator = self.locally_aggregated_grads[idx].assign_add(
                grad)
            aggregation_ops_list.append(updated_grad_aggregator)

        return aggregation_ops_list

    def _allreduce_grads_helper(self, vars):
        # Read in latest variables values.
        aggregated_grads = []
        aggregation_read_ops_list = []
        for idx, locally_aggregated_grad in enumerate(
                self.locally_aggregated_grads):
            aggregated_grads.append(locally_aggregated_grad.read_value())
            aggregation_read_ops_list.append(aggregated_grads[idx])
        aggregation_read_ops = tf.group(*aggregation_read_ops_list)

        with tf.control_dependencies([aggregation_read_ops]):
            averaged_gradients = self._allreduce_grads(aggregated_grads, vars)

            # Reset counter.
            with tf.control_dependencies([g.op for g in averaged_gradients if g is not None]):
                reset_op = self.counter.assign(
                    tf.constant(0), use_locking=True)

            # Divide by backward_passes_per_step if
            # average_aggregated_gradients is True.
            with tf.control_dependencies([reset_op]):
                gradient_divisor = self.backward_passes_per_step if \
                    self.average_aggregated_gradients else 1

                averaged_gradients = apply_op_to_not_none_tensors(
                    tf.divide,
                    averaged_gradients,
                    gradient_divisor,
                )
                return averaged_gradients

    def compute_gradients(self, grads, vars):
        """
        Applies the new gradient updates the locally aggregated gradients, and
        performs cross-machine communication every backward_passes_per_step
        times it is called.
        """
        self._init_aggregation_vars(grads)

        # Clear the locally aggregated gradients when the counter is at zero.
        clear_op = tf.cond(
            pred=tf.equal(self.counter, 0),
            true_fn=lambda: self._clear_grads(),
            false_fn=tf.no_op
        )

        # Add new gradients to the locally aggregated gradients.
        with tf.control_dependencies([clear_op]):
            aggregation_ops_list = self._aggregate_grads(grads)

        # Increment the counter once new gradients have been applied.
        aggregation_ops = tf.group(*aggregation_ops_list)
        with tf.control_dependencies([aggregation_ops]):
            update_counter = self.counter.assign_add(tf.constant(1))

        with tf.control_dependencies([update_counter]):
            grads = get_not_none_from_list(grads)
            assert len(grads) == len(self.locally_aggregated_grads)

            # Allreduce locally aggregated gradients when the counter is equivalent to
            # `backward_passes_per_step`. This the condition is true, it also resets
            # the counter back to 0.
            allreduced_grads = tf.cond(
                tf.equal(self.counter, self.backward_passes_per_step),
                lambda: self._allreduce_grads_helper(vars),
                lambda: [self._maybe_convert_grad(g) for g in grads]
            )

            # Handle case where there is only one variable.
            if not isinstance(allreduced_grads, (list, tuple)):
                allreduced_grads = (allreduced_grads,)
            assert len(allreduced_grads) == len(self.locally_aggregated_grads)

            # Insert gradients that are None back in.
            allreduced_grads = [
                allreduced_grads[self.not_none_indexes[idx]] if idx in self.not_none_indexes else None
                for idx in range(len(self.locally_aggregated_grads) + self.num_none_grad_updates)
            ]
            assert len(allreduced_grads) == len(
                self.locally_aggregated_grads) + self.num_none_grad_updates

        # If gradients have not been allreduced this batch, we return the gradients
        # that were submitted as the updates (the input).
        return allreduced_grads

    def apply_gradients(self, apply_grads_closure, optimizer, *args, **kwargs):
        """
        Apply updates every backward_passes_per_step, which lines up with
        the batches on which we communicated the locally aggregated gradients.
        """
        flattended_args0 = [item for tup in args[0] for item in tup]

        # If optimizer tracks iterations, we increment it on steps where we
        # are not going to call `apply_gradients()`.
        def increment_optimizer_iteration():
            if hasattr(optimizer, "_iterations") and optimizer._iterations is not None:
                return optimizer._iterations.assign_add(1).op
            return tf.no_op()

        with tf.control_dependencies([tf.group(*get_not_none_from_list(flattended_args0))]):
            train_op = tf.cond(
                pred=tf.equal(self.counter, 0),
                true_fn=apply_grads_closure,
                false_fn=increment_optimizer_iteration,
            )

        # Since we skip applying updates when the counter is not at zero we
        # still want to increment the global step if it is being tracked
        # (e.g., Tensorflow Estimators).
        def increment_global_step_counter():
            global_step_counter = kwargs.get('global_step')
            if global_step_counter is None:
                return tf.no_op()
            return global_step_counter.assign_add(
                tf.constant(1, dtype=tf.int64),
                use_locking=True,
                read_value=False
            )

        with tf.control_dependencies([train_op]):
            # Increment global step on iterations where we don't call `apply_gradients()`.
            return tf.cond(
                pred=tf.equal(self.counter, 0),
                true_fn=tf.no_op,
                false_fn=increment_global_step_counter,
            )
