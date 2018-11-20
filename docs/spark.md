## Horovod in Spark

(TODO)

Environment knobs:

* `HOROVOD_SPARK_START_TIMEOUT` - sets the default timeout for Spark
tasks to spawn, register and start running the code.  If executors for
Spark tasks are scheduled on-demand and can take long time to start,
it may be useful to increase this timeout on a system level.
