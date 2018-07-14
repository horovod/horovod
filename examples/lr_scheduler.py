import logging
from math import cos, pi
# commit: https://github.com/rahul003/mxnet/commit/2330555c27ecd7e36d7900d2a1c0d0c398a34830
class LRScheduler(object):
    """Base class of a learning rate scheduler.
    A scheduler returns a new learning rate based on the number of updates that have
    been performed.
    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    """
    def __init__(self, base_lr=0.01, warmup_steps=0):
        self.base_lr = base_lr
        assert isinstance(warmup_steps, int)
        self.warmup_steps = warmup_steps

    def __call__(self, num_update):
        """Return a new learning rate.
        The ``num_update`` is the upper bound of the number of updates applied to
        every weight.
        Assume the optimizer has updated *i*-th weight by *k_i* times, namely
        ``optimizer.update(i, weight_i)`` is called by *k_i* times. Then::
            num_update = max([k_i for all i])
        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")

class FactorScheduler(LRScheduler):
    """Reduce the learning rate by a factor for every *n* steps.
    It returns a new learning rate by::
        base_lr * pow(factor, floor(num_update/step))
    Parameters
    ----------
    step : int
        Changes the learning rate for every n updates.
    factor : float, optional
        The factor to change the learning rate.
    stop_factor_lr : float, optional
        Stop updating the learning rate if it is less than this value.
    """
    def __init__(self, step, factor=1, stop_factor_lr=1e-8, base_lr=0.01):
        super(FactorScheduler, self).__init__(base_lr)
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.count = 0

    def __call__(self, num_update):
        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while num_update > self.count + self.step:
            self.count += self.step
            self.base_lr *= self.factor
            if self.base_lr < self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr
                logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                             "change in the future", num_update, self.base_lr)
            else:
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
        return self.base_lr

class MultiFactorScheduler(LRScheduler):
    """Reduce the learning rate by given a list of steps.
    Assume there exists *k* such that::
       step[k] <= num_update and num_update < step[k+1]
    Then calculate the new learning rate by::
       base_lr * pow(factor, k+1)
    Parameters
    ----------
    step: list of int
        The list of steps to schedule a change
    factor: float
        The factor to change the learning rate.
    """
    def __init__(self, step, factor=1, base_lr=0.01):
        super(MultiFactorScheduler, self).__init__(base_lr)
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr

class PolyScheduler(LRScheduler):
    """ Reduce the learning rate according to a polynomial of given power.
    Calculate the new learning rate by::
       final_lr + (start_lr - final_lr) * (1-nup/max_nup)^pwr
       if nup < max_nup, 0 otherwise.
    Parameters
    ----------
       max_update: maximum number of updates before the decay reaches final learning rate.
       base_lr:    base learning rate to start from
       pwr:   power of the decay term as a function of the current number of updates.
       final_lr:   final learning rate after all steps
       warmup_steps: number of warmup steps used before this scheduler starts decay
    """

    def __init__(self, max_update, base_lr=0.01, pwr=2, final_lr=0, warmup_steps=0):
        super(PolyScheduler, self).__init__(base_lr, warmup_steps)
        assert isinstance(max_update, int)
        if max_update < 1:
            raise ValueError("maximum number of updates must be strictly positive")
        self.power = pwr
        self.base_lr_orig = self.base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.max_steps = self.max_update - self.warmup_steps

    def __call__(self, num_update):
        if num_update <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                pow(1 - float(num_update - self.warmup_steps) / float(self.max_steps), self.power)
        return self.base_lr

class CosineScheduler(LRScheduler):
    """ Reduce the learning rate by given a list of steps.
    Calculate the new learning rate by::
       final_lr + (start_lr - final_lr) * (1+cos(pi * nup/max_nup))/2
       if nup < max_nup, 0 otherwise.
    Parameters
    ----------
       max_update: maximum number of updates before the decay reaches 0
       base_lr:    base learning rate
       final_lr:   final learning rate after all steps
       warmup_steps: number of warmup steps used before this scheduler starts decay
    """

    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0):
        super(CosineScheduler, self).__init__(base_lr, warmup_steps)
        assert isinstance(max_update, int)
        if max_update < 1:
            raise ValueError("maximum number of updates must be strictly positive")
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.max_steps = self.max_update - self.warmup_steps

    def __call__(self, num_update):
        if num_update <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                (1 + cos(pi * (num_update - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

class WarmupScheduler(LRScheduler):
    """Implement warmup of learning rate for given number of steps.
    Linear warmup starting from given base_lr to given scheduler's base_lr
    or constant warmup at base_lr
    Parameters
    ----------
    base_lr: float
            learning rate to begin warmup from if mode is linear.
            if mode is constant, stays at this lr
    warmup_steps: int
            number of warmup steps
    scheduler: LRScheduler
            scheduler following the warmup
    mode: str
            type of warmup, either linear or constant
    """
    def __init__(self, base_lr, warmup_steps, scheduler, mode='linear'):
        super(WarmupScheduler, self).__init__(base_lr, warmup_steps)
        self.scheduler = scheduler
        self.lr_final = self.scheduler.base_lr
        self.lr_begin = self.base_lr
        if self.lr_begin > self.lr_final:
            raise ValueError("Final lr has to be higher than beginning lr")
        if warmup_steps <= 0:
            raise ValueError("Warmup steps has to be positive")
        if mode not in ['linear', 'constant']:
            raise ValueError("Warmup scheduler supports only linear and constant modes")
        self.mode = mode
        self.lrs_updates = {}
        self.lr_difference = self.lr_final - self.lr_begin

    def __call__(self, num_update):
        if num_update not in self.lrs_updates:
            if num_update < self.warmup_steps:
                increase = self.lr_difference * float(num_update)/float(self.warmup_steps)
                self.lrs_updates[num_update] = self.lr_begin + increase
            else:
                # uses warmup steps of given scheduler to determine the number of
                # updates that scheduler should start after
                self.lrs_updates[num_update] = self.scheduler(num_update - self.scheduler.warmup_steps)
        return self.lrs_updates[num_update]
