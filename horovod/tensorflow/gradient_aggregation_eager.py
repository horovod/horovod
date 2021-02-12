from distutils.version import LooseVersion

import tensorflow as tf

_POST_TF_2_4_0 = LooseVersion(tf.__version__) >= LooseVersion('2.4.0')


class LocalGradientAggregationHelperEager:
    """
   LocalGradientAggregationHelperEager aggregates gradient updates
   locally, and communicates the updates across machines only once
   every backward_passes_per_step. Only supports eager execution.
   """

    def __init__(
        self,
        backward_passes_per_step,
        allreduce_func,
        sparse_as_dense,
        average_aggregated_gradients,
    ):
        self.allreduce_grads = allreduce_func
        self.sparse_as_dense = sparse_as_dense

        # backward_passes_per_step controls how often gradient updates are
        # synchronized.
        self.backward_passes_per_step = backward_passes_per_step
        if self.backward_passes_per_step <= 0:
            raise ValueError("backward_passes_per_step must be > 0")

        # average_aggregated_gradients controls whether gradient updates that are
        # aggregated, should be divided by `backward_passes_per_step`.
        self.average_aggregated_gradients = average_aggregated_gradients

        # This is going to be N data structure holding the aggregated gradient updates
        # for parameter updates. N is the number of parameters.
        self.locally_aggregated_grads = {}

        # Used to know when to allreduce and apply gradients. We allreduce when `self.counter`
        # is equal to `self.backward_passes_per_step`. We apply gradients when `self.counter`
        # is equal to 0.
        self.counter = tf.Variable(initial_value=0)

    @tf.function
    def compute_gradients(self, grads, vars):
        # On steps where allreduce happens, resulting_grads returns the allreduced
        # gradients, on other steps it returns the locally aggregated
        # gradients.
        resulting_grads = []

        for idx, grad in enumerate(grads):
            # Handle IndexedSlices.
            if self.sparse_as_dense and isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)
            elif isinstance(grad, tf.IndexedSlices):
                raise ValueError(
                    "IndexedSlices are not supported when "
                    "`backward_passes_per_step` > 1 and "
                    "`sparse_as_dense` is False."
                )

            # Create variables to store to aggregate gradients if they don't
            # already exist. Skip variables that are None.
            if idx not in self.locally_aggregated_grads.keys():
                if grad is not None:
                    self.locally_aggregated_grads[idx] = tf.Variable(
                        initial_value=tf.zeros_like(grad),
                        trainable=False,
                        dtype=grad.dtype,
                    )

            if grad is None:
                resulting_grads.append(None)
            else:
                self.locally_aggregated_grads[idx].assign_add(grad)
                resulting_grads.append(
                    self.locally_aggregated_grads[idx].read_value())
        assert len(self.locally_aggregated_grads) == len(grads)

        # Increment counter.
        self.counter.assign_add(1)

        def _all_reduce_and_clear_aggregated_variables(aggregated_gradients, vars):
            # Performs allreduce. If `average_aggregated_gradients` is
            # set to True divides result by `backward_passes_per_step`.
            reduced_gradients = self._allreduce_helper(aggregated_gradients, vars)
            assert len(reduced_gradients) == len(grads)

            self._clear_vars()
            return reduced_gradients

        def _do_nothing(aggregated_gradients):
            return aggregated_gradients

        resulting_grads = tf.cond(
            pred=tf.equal(self.counter, self.backward_passes_per_step),
            true_fn=lambda: _all_reduce_and_clear_aggregated_variables(resulting_grads, vars),
            false_fn=lambda: _do_nothing(resulting_grads),
        )

        return resulting_grads

    def _allreduce_helper(self, grads, vars):
        allreduced_grads = self.allreduce_grads(grads, vars)

        if not self.average_aggregated_gradients:
            return allreduced_grads

        scaled_allreduced_grads = []
        for grad in allreduced_grads:
            if grad is None:
                scaled_allreduced_grads.append(grad)
                continue

            scaled_allreduced_grads.append(
                grad / self.backward_passes_per_step)

        return scaled_allreduced_grads

    def _clear_vars(self):
        self.counter.assign(0)
        for idx in self.locally_aggregated_grads.keys():
            self.locally_aggregated_grads[idx].assign_add(
                -1 * self.locally_aggregated_grads[idx])

    def apply_gradients(self, apply_grads_closure, optimizer, *args, **kwargs):
        def increment_optimizer_iteration():
            if hasattr(optimizer, "_iterations") and optimizer._iterations is not None:
                return optimizer._iterations.assign_add(1).op
            return tf.no_op()

        def non_aggregation_step():
            if _POST_TF_2_4_0:
                # In TF 2.4+ `_aggregate_gradients()` is called from inside of `apply_gradients()`.
                # We account for this by calling `_aggregate_gradients()` for steps where we do
                # not call `apply_gradients()`.
                transformed_grads_and_vars = optimizer._transform_unaggregated_gradients(
                    args[0])
                _ = optimizer._aggregate_gradients(transformed_grads_and_vars)

            return increment_optimizer_iteration()

        def is_aggregation_step():
            if _POST_TF_2_4_0:
                # In TF 2.4+ we evaluate whether it's time to aggregated gradients before
                # calling `_aggregate_gradients()`.
                return tf.equal(tf.add(self.counter, 1), self.backward_passes_per_step)

            return tf.equal(self.counter, 0)

        return tf.cond(
            pred=is_aggregation_step(),
            true_fn=apply_grads_closure,
            false_fn=non_aggregation_step,
        )
