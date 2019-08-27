.. inclusion-marker-start-do-not-remove

Autotune: Automated Performance Tuning
======================================

Horovod comes with several adjustable "knobs" that can affect runtime performance, including
``--fusion-threshold-mb`` and ``--cycle-time-ms`` (tensor fusion), ``--cache-capacity`` (response cache), and
hierarchical collective algorithms ``--hierarchical-allreduce`` and ``--hierarchical-allgather``.

Determining the best combination of these values to maximize performance (minimize time to convergence) can be a
matter of trial-and-error, as many factors including model complexity, network bandwidth, GPU memory, etc. can all
affect inputs per second throughput during training.

Horovod provides a mechanism to automate the process of selecting the best values for these "knobs" called
**autotuning**. The Horovod autotuning system uses
`Bayesian optimization <https://en.wikipedia.org/wiki/Bayesian_optimization>`_ to intelligently search through the
space of parameter combinations during training. This feature can be enabled by setting the ``--autotune`` flag for
``horovodrun``:

.. code-block:: bash

    $ horovodrun -np 4 --autotune python train.py

When autotuning is enabled, Horovod will spend the first steps / epochs of training experimenting with different
parameter values and collecting metrics on performance (measured in bytes allreduced / allgathered per unit of time).
Once the experiment reaches convergence, or a set number of samples have been collected, the system will record the best
combination of parameters discovered and continue to use them for the duration of training.

A log of all parameter combinations explored (and the best values selected) can be recorded by providing
the ``--autotune-log-file`` option to ``horovodrun``:

.. code-block:: bash

    $ horovodrun -np 4 --autotune --autotune-log-file /tmp/autotune_log.csv python train.py

By logging the best parameters to a file, you can opt to set the best parameters discovered on the command line
instead of re-running autotuning if training is paused and later resumed.

Note that some configurable parameters, like tensor compression, are not included as part of the autotuning process
because they can affect model convergence. The purpose of autotuning at this time is entirely to improve scaling
efficiency without making any tradeoffs on model performance.


Constant Parameters
-------------------

Sometimes you may wish to hold certain values constant and only tune the unspecified parameters. This can be
accomplished by explicitly setting those values on the command line or in the config file provided
by ``--config-file``:

.. code-block:: bash

    $ horovodrun -np 4 --autotune --cache-capacity 1024 --no-hierarchical-allgather python train.py

In the above example, parameters ``cache-capacity`` and ``hierarchical-allgather`` will not be adjusted by
autotuning.


Advanced Autotuning
-------------------

Enabling autotuning imposes a tradeoff between degraded performance during the early phases of training in exchange for
better performance later on. As such, it's generally recommended to use autotuning in situations where training is both
expected to take a long time (many epochs on very large datasets) and where scaling efficiency has been found lacking
using the default settings.

You can tune the autotuning system itself to change the number of warmup samples (discarded samples at the beginning),
steps per sample, and maximum samples:

.. code-block:: bash

    $ horovodrun -np 4 --autotune \
    --autotune-warmup-samples 5 --autotune-steps-per-sample 20 --autotune-bayes-opt-max-samples 40 \
    python train.py

Increasing these values will generally improve the accuracy of the autotuning process at the cost of greater time
spent in the autotuning process with degraded performance.

Finally, for those familiar with the underlying theory of Bayesian optimization and Gaussian processes, you can tune
the noise regularization term (alpha) to account for variance in your network or other system resources:

.. code-block:: bash

    $ horovodrun -np 4 --autotune --autotune-gaussian-process-noise 0.75 python train.py

.. inclusion-marker-end-do-not-remove
