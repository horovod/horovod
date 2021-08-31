# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from queue import Queue, Empty
from threading import Thread, Event


class BaseDataLoader(object):
    def __len__(self):
        """
        Length of the batches to be loaded.
        """
        raise NotImplementedError()

    def _iterate(self):
        """
        Interface for the implimentation of iterate batches
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        Starting iteration and get batchs
        """
        for batch in self._iterate():
            yield self._process_batch(batch)

    def _process_batch(self, batch):
        """
        Hook to modify batch before output. Will be override by trainer to reshape the data
        as needed. Please do not override it.
        """
        return batch


class AsyncDataLoaderMixin(object):
    """
    Async Mixin on top of implementation of BaseDataLoader. It contains a seperate thread
    which reads batch from self._iterate() and push them in the queue. The self.__iter__() function
    will pop the batch from the queue.
    If async_loader_queue_size is set to 0, the data loader will not work in async mode.
    For example:
        class PytorchAsyncDataLoader(AsyncDataLoaderMixin, PytorchDataLoader):
    """

    def __init__(self, async_loader_queue_size=64, debug_data_loader=False, *args, **kwargs):
        """
        initialize the async data loader. Need to add this in the __init__() of the implementation
        """
        self.async_loader_queue_size = async_loader_queue_size
        self.debug_data_loader = debug_data_loader
        super().__init__(*args, **kwargs)

        print(f"Apply the AsyncDataLoaderMixin on top of the data loader, async_loader_queue_size={async_loader_queue_size}. ")

        if self.async_loader_queue_size > 0:
            self.finished_event = Event()
            self.queue = Queue(self.async_loader_queue_size)
            self.thread = Thread(target=self._async_worker)
            self.thread.daemon = True
            self.started = False

    def close_async_loader(self):
        """
        Close the async data loader.
        """
        print(f"close_async_loader[{self.async_loader_queue_size}], Closing the AsyncDataLoaderMixin.")
        if self.async_loader_queue_size > 0 and self.started:
            self.finished_event.set()
            c = 0
            while True:
                try:
                    # Drain buffer
                    self.queue.get_nowait()
                    if self.debug_data_loader:
                        print(f"close_async_loader[{self.async_loader_queue_size}], discarded batch #{c} from Queue.")

                    c += 1

                    # Force break out if hanging. We assume hanging if already pop items more than twice of the size of the queue.
                    if c > max(2 * self.async_loader_queue_size, 200):
                        print(f"close_async_loader: ERROR!!! Force break out after {c} times get_nowait.")
                        break
                except Empty:
                    break
            if self.debug_data_loader:
                print(f"close_async_loader[{self.async_loader_queue_size}], joining...")

            self.thread.join()

        print(f"close_async_loader[{self.async_loader_queue_size}] is closed.")

    def _async_worker(self):
        """
        Start worker thread to load data asynchronously.
        User need to implement self._iterate() to read the data.
        """
        try:
            c = 0
            while not self.finished_event.is_set():
                for batch in self._iterate():
                    if self.finished_event.is_set():
                        break
                    self.queue.put(batch)

                    if self.debug_data_loader:
                        print(f"_async_worker[{self.async_loader_queue_size}], push batch #{c}.")
                        c += 1

                if self.debug_data_loader:
                    print(f"_async_worker[{self.async_loader_queue_size}], finish reading at #{c}, reset debugging counter, append None to queue.")
                    c = 0

                self.queue.put(None)
        except Exception as ex:
            self.queue.put(ex)
            self.queue.put(None)
        finally:
            self.queue.put(None)

        print(f"_async_worker[{self.async_loader_queue_size}], stoped")

    def __iter__(self):
        """
        Override the __iter__() to iterate data asynchronously to produce batchs.
        Will procude batchs from the queue which were generated by self._iterate().
        """

        print("Start generating batches from async data loader.")
        if self.async_loader_queue_size > 0:
            if not self.started:
                self.started = True
                self.thread.start()

            c = 0
            while True:
                batch = self.queue.get()

                if self.debug_data_loader:
                    print(f"__iter__[{self.async_loader_queue_size}], get batch #{c}.")
                    c += 1

                if batch is None:
                    if self.debug_data_loader:
                        print(f"__iter__[{self.async_loader_queue_size}], get None from queue at #{c}.")
                    break
                if isinstance(batch, Exception):
                    raise batch

                yield self._process_batch(batch)
        else:
            for batch in self._iterate():
                yield self._process_batch(batch)
