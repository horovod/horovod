from typing import Dict, List
import socket
import ray
import os
from horovod.ray import ray_logger


class BaseHorovodWorker:
    executable = None

    def __init__(self, world_rank=0, world_size=1):
        os.environ["HOROVOD_HOSTNAME"] = self.node_id()
        os.environ["HOROVOD_RANK"] = str(world_rank)
        os.environ["HOROVOD_SIZE"] = str(world_size)

    def node_id(self) -> str:
        return ray.get_runtime_context().node_id.hex()

    def hostname(self) -> str:
        # TODO: This is probably not the right way to retrieve
        # the intended hostname.
        return socket.gethostname()

    def get_gpu_ids(self) -> List[int]:
        """Return list of CUDA device IDs available to this worker."""
        return ray.get_gpu_ids()

    def update_env_vars(self, env_vars: Dict[str, str]):
        """Update the env vars in the actor process."""
        sanitized = {k: str(v) for k, v in env_vars.items()}
        os.environ.update(sanitized)

    def env_vars(self):
        """Check the env vars in the actor process."""
        return dict(os.environ)

    def start_executable(self,
                         executable_cls: type = None,
                         executable_args: list = None,
                         executable_kwargs: dict = None):
        """Instantiates the executable class with provided args.

        If none, self.executable = None.

        Args:
            executable_cls (type): Class of object to be created on all
                workers.
            executable_args (list): Initialization arguments for the
                executable_cls.
            executable_kwargs (dict): Initialization arguments for the
                executable_cls.
        """
        executable_args = executable_args or []
        executable_kwargs = executable_kwargs or {}
        if executable_cls:
            self.executable = executable_cls(*executable_args,
                                             **executable_kwargs)

    def execute(self, func):
        """Executes an arbitrary function on self."""
        return func(self.executable)

    def set_queue(self, queue):
        """Sets the queue for multi-node logging."""
        ray_logger.configure(queue=queue)
