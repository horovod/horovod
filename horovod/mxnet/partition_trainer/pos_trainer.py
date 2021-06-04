from horovod.mxnet.mpi_ops import allreduce, allreduce_
from horovod.mxnet.mpi_ops import broadcast, broadcast_
from horovod.mxnet.mpi_ops import init, shutdown
from horovod.mxnet.mpi_ops import size, local_size,  rank, local_rank

import mxnet as mx
from collections import OrderedDict, defaultdict
import types
import time
import warnings
from mxnet.gluon.parameter import Parameter


class POS_Trainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None,
                 gradient_predivide_factor=1.0, prefix=None):

        self._world_size = size()
        self._world_rank = rank()

        self._all_params = []
        self._all_param2idx = {}
        param_list = []
        if isinstance(params, (dict, OrderedDict)):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])
            params = param_list
        if not isinstance(params, (list, tuple)):
            raise ValueError(
                "First argument must be a list or dict of Parameters, " \
                "got %s." % (type(params)))
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise ValueError(
                    "First argument must be a list or dict of Parameters, " \
                    "got list of %s." % (type(param)))
            if param._uuid in self._all_param2idx:
                # Shared parameters have same uuid; only need to store one of the shared versions
                continue
            self._all_param2idx[param._uuid] = i
            self._all_params.append(param)
        self._partition_params, self._param2rank = self.partition_parameters(self._all_params)
        self._own_part = self._partition_params[self._world_rank]
        super(POS_Trainer, self).__init__(
            self._own_part, optimizer, optimizer_params=optimizer_params, kvstore=None)
        self._prefix = prefix if prefix else ""
        self._scale = gradient_predivide_factor / size()
        self._gradient_predivide_factor = gradient_predivide_factor


    def partition_parameters(self, params):
        world_size = self._world_size
        ## list for rank each would be
        partition_params = [[] for _ in range(world_size)]
        param2rank = {}
        sizes = [0 for _ in range(world_size)]
        for param in params:
            if param.grad_req != 'null':
                current_rank = sizes.index(min(sizes))
                partition_params[current_rank].append(param)
                num = 1
                param2rank[param._uuid] = current_rank
                for p in param.shape:
                    num *= p
                sizes[current_rank] += num
        return partition_params, param2rank

    def _allreduce_grads(self):
        for i, param in enumerate(self._all_params):
            if param.grad_req != 'null':
                allreduce_(param.list_grad()[0], average=False,
                           name=self._prefix + str(i), priority=-i,
                           prescale_factor=1.0 / self._gradient_predivide_factor)

    def step(self, batch_size, ignore_stale_grad=False):
        """
        inherit from trainer, only call boardcast to make sure all parameter are consistent
        Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.
        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        self._allreduce_grads()
        self._update(ignore_stale_grad)
        self.broadcast_params()

    def update(self, batch_size, ignore_stale_grad=False):
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        assert not (self._kvstore and self._update_on_kvstore), \
            'update() when parameters are updated on kvstore ' \
            'is not supported. Try setting `update_on_kvstore` ' \
            'to False when creating trainer.'

        self._check_and_rescale_grad(self._scale / batch_size)
        self._update(ignore_stale_grad)
        self.broadcast_params()



    def broadcast_params(self):
        for param in self._all_params:
            broadcast_(param.data(), self._param2rank[param._uuid], name=str(self._all_param2idx[param._uuid]))

    def _broadcast_parameters(self, params, root_rank=0, prefix=None):
        """
        Broadcasts the parameters from root rank to all other processes.
        Typical usage is to broadcast the `Module.get_params()` or the
        `Block.collect_params()`.
        Arguments:
            params: One of the following:
                - dict of parameters to broadcast
                - ParameterDict to broadcast
            root_rank: The rank of the process from which parameters will be
                       broadcasted to all other processes.
            prefix: The prefix of the parameters to broadcast.
                  If multiple `broadcast_parameters` are called in the same program,
                  they must be specified by different prefixes to avoid tensor name collision.
        """

        if size() == 1:
            return

        tensors = []
        names = []
        assert prefix is None or isinstance(prefix, str)
        prefix = prefix if prefix else ""
        try:
            from mxnet.gluon.parameter import ParameterDict
            valid_types = (dict, ParameterDict)
        except ImportError:
            valid_types = (dict,)
        if isinstance(params, valid_types):
            for name, p in sorted(params.items()):
                try:
                    if isinstance(p, mx.gluon.parameter.Parameter):
                        tensors.append(p.data())
                    else:
                        tensors.append(p)
                    names.append(prefix + str(name))
                except mx.gluon.parameter.DeferredInitializationError:
                    # Inject wrapper method with post-initialization broadcast to
                    # handle parameters with deferred initialization
                    # we use the key of params instead of param.name, since
                    # param.name is no longer unique in MXNet 2.0
                    new_init = _append_broadcast_init(p, root_rank, prefix + str(name))
                    p._init_impl = types.MethodType(new_init, p)
        else:
            raise ValueError('invalid params of type: %s' % type(params))

        # Run broadcasts.
        for tensor, name in zip(tensors, names):
            broadcast_(tensor, root_rank, name=name)

