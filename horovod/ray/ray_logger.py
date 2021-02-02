import os
import warnings

_queue = None
_queue_set = False
warning_raised = False


def configure(queue):
    global _queue
    global _queue_set
    _queue = queue
    _queue_set = True


def log(info_dict):
    global warning_raised
    if os.environ.get("HOROVOD_RANK") == "0":
        if _queue_set:
            _queue.put(info_dict)
        elif not warning_raised:
            warnings.warn(
                "`ray_logger.log` called but logging is not configured.")
            warning_raised = True
