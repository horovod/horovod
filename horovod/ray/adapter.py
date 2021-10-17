from abc import ABC, abstractmethod
from typing import Dict, Callable, Any, Optional, List
from dataclasses import dataclass

@dataclass
class BaseParams:
    cpus_per_worker: int = 1
    use_gpu: bool = False
    gpus_per_worker: Optional[int] = None
    def __post_init__(self):
        if self.gpus_per_worker and not self.use_gpu:
            raise ValueError("gpus_per_worker is set, but use_gpu is False. "
                             "use_gpu must be True if gpus_per_worker is "
                             "set. ")
        if self.use_gpu and isinstance(self.gpus_per_worker,
                                  int) and self.gpus_per_worker < 1:
            raise ValueError(
                f"gpus_per_worker must be >= 1: Got {self.gpus_per_worker}.")
        self.gpus_per_worker = self.gpus_per_worker or int(self.use_gpu)


class Adapter(ABC):
    """Adapter for executing Ray calls for various types(e.g. static and elastic)
    Horovod jobs.
    """
    @abstractmethod
    def start(self,
              executable_cls: type = None,
              executable_args: Optional[List] = None,
              executable_kwargs: Optional[Dict] = None,
              extra_env_vars: Optional[Dict] = None):
        """Starts the Adapter

        Args:
            executable_cls (type): The class that will be created within
                an actor (BaseHorovodWorker). This will allow Horovod
                to establish its connections and set env vars.
            executable_args (List): Arguments to be passed into the
                worker class upon initialization.
            executable_kwargs (Dict): Keyword arguments to be passed into the
                worker class upon initialization.
            extra_env_vars (Dict): Environment variables to be set
                on the actors (worker processes) before initialization.
        """
        raise NotImplementedError("Method must be implemented in a subclass")

    @abstractmethod
    def execute(self, fn: Callable[["executable_cls"], Any],
                callbacks: Optional[List[Callable]] = None) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            fn: Target function to be invoked on every object.
            callbacks: List of callables. Each callback must either
                be a callable function or a class that implements __call__.
                Every callback will be invoked on every value logged
                by the rank 0 worker.

        Returns:
            Deserialized return values from the target function.
        """
        raise NotImplementedError("Method must be implemented in a subclass")

    @abstractmethod
    def run(self,
        fn: Callable[[Any], Any],
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
        callbacks: Optional[List[Callable]] = None) -> List[Any]:
        """Executes the provided function on all workers.

        Args:
            fn: Target function that can be executed with arbitrary
                args and keyword arguments.
            args: List of arguments to be passed into the target function.
            kwargs: Dictionary of keyword arguments to be
                passed into the target function.
            callbacks: List of callables. Each callback must either
                be a callable function or a class that implements __call__.
                Every callback will be invoked on every value logged
                by the rank 0 worker.

        Returns:
            Deserialized return values from the target function.
        """
        raise NotImplementedError("Method must be implemented in a subclass")

    @abstractmethod
    def run_remote(self,
                   fn: Callable[[Any], Any],
                   args: Optional[List] = None,
                   kwargs: Optional[Dict] = None,
                   callbacks: Optional[List[Callable]] = None):

        """Executes the provided function on all workers.

        Args:
            fn: Target function that can be executed with arbitrary
                args and keyword arguments.
            args: List of arguments to be passed into the target function.
            kwargs: Dictionary of keyword arguments to be
                passed into the target function.

        Returns:
            list: List of ObjectRefs that you can run `ray.get` on to
                retrieve values.
        """
        raise NotImplementedError("Method must be implemented in a subclass")

    @abstractmethod
    def execute_single(self,
                       fn: Callable[["executable_cls"], Any]) -> List[Any]:
        """Executes the provided function on the rank 0 worker (chief).

        Args:
            fn: Target function to be invoked on the chief object.

        Returns:
            Deserialized return values from the target function.
        """
        raise NotImplementedError("Method must be implemented in a subclass")

    @abstractmethod
    def shutdown(self):
        """Destroys the adapter."""
        raise NotImplementedError("Method must be implemented in a subclass")
