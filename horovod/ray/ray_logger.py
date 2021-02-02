"""Module for supporting multi-node callbacks.

This module maintains a set of global variables for each
Ray process.
"""

import os
import warnings

_queue = None
_queue_set = False
warning_raised = False


def configure(queue):
    """Sets the proper global variables."""

    global _queue
    global _queue_set
    _queue = queue
    # We maintain a _queue_set variable because somehow
    # the _queue variable makes a lot of unnecessary remote calls.
    _queue_set = True


def log(info_dict, chief_only=True):
    """Sends info dict back to the driver, where it will be processed.

    Args:
        info_dict (dict): Dictionary of serializable elements.
        chief_only (bool): Whether to only send dicts to the driver
            if on rank == 0.
    """
    global warning_raised
    global _queue_set
    should_log = os.environ.get("HOROVOD_RANK") == "0"
    if not chief_only:
        should_log = True
    if should_log:
        if _queue_set:
            _queue.put(info_dict)
        elif not warning_raised:
            warnings.warn(
                "`ray_logger.log` called but logging is not configured.")
            warning_raised = True
