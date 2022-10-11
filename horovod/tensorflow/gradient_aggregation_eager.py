import os
import tensorflow as tf
from packaging import version
from horovod.tensorflow.mpi_ops import size_op
from horovod.tensorflow.mpi_ops import global_process_set


_POST_TF_2_4_0 = version.parse(tf.__version__) >= version.parse('2.4.0')
_IS_TF2 = version.parse(tf.__version__) >= version.parse('2.0.0')


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
        process_set=global_process_set,
        scale_local_gradients=True
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

        self.process_set = process_set
        self.scale_local_gradients = scale_local_gradients
        self._local_vars = set()

    def register_local_var(self, var):
        """Registers a source/variable as worker local. Horovod will not perform any global
        operations on gradients corresponding to these sources and will instead return the local
        gradient."""
        if _IS_TF2:
            self._local_vars.add(var.ref())
        else:
            self._local_vars.add(var)

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
        def __filtered_reduce_grads(grads, vars):
            rv = []
            rg = []
            if _IS_TF2:
                v2g = {var.ref(): grad for var, grad in zip(vars, grads)}
                for var, grad in zip(vars, grads):
                    if var.ref() not in self._local_vars:
                        rv.append(var)
                        rg.append(grad)
            else:
                v2g = {var: grad for var, grad in zip(vars, grads)}
                for var, grad in zip(vars, grads):
                    if var not in self._local_vars:
                        rv.append(var)
                        rg.append(grad)

            rg = self.allreduce_grads(rg, rv)
            horovod_size = size_op(process_set_id=self.process_set.process_set_id) if int(os.environ.get("HOROVOD_ELASTIC", 0)) else self.process_set.size()
            if _IS_TF2:
                for rv, rg in zip(rv, rg):
                    v2g[rv.ref()] = rg

                if self.scale_local_gradients and len(self._local_vars):
                    # Scale local gradients by a size factor. See pull/3695 and discussions/3705 for context.
                    for v_ref in v2g:
                        if v_ref in self._local_vars and v2g[v_ref] is not None:
                            v2g[v_ref] /= horovod_size

                return [v2g[rv.ref()] for rv in vars]
            else:
                for rv, rg in zip(rv, rg):
                    v2g[rv] = rg

                if self.scale_local_gradients and len(self._local_vars):
                    # Scale local gradients by a size factor. See pull/3695 and discussions/3705 for context.
                    for v in v2g:
                        if v in self._local_vars and v2g[v] is not None:
                            v2g[v] /= horovod_size

                return [v2g[rv] for rv in vars]

        allreduced_grads = __filtered_reduce_grads(grads, vars)

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
            self.locally_aggregated_grads[idx].assign(
                tf.zeros_like(self.locally_aggregated_grads[idx]))

    def apply_gradients(self, apply_grads_closure, optimizer, *args, **kwargs):
        def increment_optimizer_iteration():
            # (kvignesh1420): Since all `tf.OptimizerV2` instances have the `iterations`
            # property for modifying the underlying `optimizer._iterations`, it is safe to use
            # the property instead of the private variable. For instance, the keras
            # `LossScaleOptimizer` inherits `tf.Optimizer` and exposes the cleaner `iterations`
            # property instead of the unsafe `_iterations`.

            if hasattr(optimizer, "iterations") and optimizer.iterations is not None:
                return optimizer.iterations.assign_add(1).op
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
            return tf.equal(self.counter, 0)

        return tf.cond(
            pred=is_aggregation_step(),
            true_fn=apply_grads_closure,
            false_fn=non_aggregation_step,
        )
