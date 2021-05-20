import math
import torch
from queue import Queue, Empty
from threading import Thread, Event
from petastorm import make_batch_reader
from petastorm.pytorch import BatchedDataLoader


class BaseDataLoader(object):
    def __len__(self):
        # If we cannot infer the number of iteratios we return 0
        return 0

    @property
    def shape(self):
        # TODO: deprecate, change to annotation
        return self.annotation


class BaseAsyncDataLoader(BaseDataLoader):
    def __init__(self, maxsize=5):
        self.maxsize = maxsize
        if self.maxsize > 0:
            self.finished_event = Event()
            self.q = Queue(self.maxsize)
            self.t = Thread(target=self._worker)
            self.t.daemon = True
            self.started = False

    def __del__(self):
        self.close()

    def close(self):
        if self.maxsize > 0 and self.started:
            self.finished_event.set()
            try:
                # Free buffer to allow worker to retry
                self.q.get_nowait()
            except Empty:
                pass
            self.t.join()

    def _worker(self):
        try:
            while not self.finished_event.is_set():
                for b in self._iterate():
                    if self.finished_event.is_set():
                        break
                    self.q.put(b)
                self.q.put(None)
        except Exception as ex:
            self.q.put(ex)
            self.q.put(None)
        finally:
            self.q.put(None)

    def __iter__(self):
        if self.maxsize > 0:
            if not self.started:
                self.started = True
                self.t.start()
            while True:
                b = self.q.get()
                if b is None:
                    break
                if isinstance(b, Exception):
                    raise b
                yield b
        else:
            for b in self._iterate():
                yield b


class PetastormReaderWrapper(BaseDataLoader):
    def __init__(self, reader):
        self.reader = reader
        self.total_size = None
        self.annotation = None

    @property
    def schema(self):
        return self.reader.schema

    def __len__(self):
        return self.total_size

    def __iter__(self):
        if self.reader.last_row_consumed:
            self.reader.reset()

        total_size = 0
        for b in self.reader:
            b = b._asdict()
            # b = NamedDataset(b, annotation=self.annotation)
            total_size += len(b)

            yield b
        self.total_size = total_size


class PetastormDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffling_queue_capacity):
        if not isinstance(dataset, PetastormReaderWrapper):
            dataset = PetastormReaderWrapper(dataset)
        self.reader = dataset.reader
        self.annotation = dataset.annotation
        self.batch_size = batch_size
        self.shuffling_queue_capacity = shuffling_queue_capacity
        print(f"Initializing petastorm dataloader with batch_size {batch_size}"
              f" and shuffling_queue_capacity {shuffling_queue_capacity}")

    def __len__(self):
        return len(self.reader)

    def __iter__(self):
        if self.reader.last_row_consumed:
            print(f"Resetting Petastorm reader for {self.reader.dataset.paths}")
            self.reader.reset()

        data_loader = BatchedDataLoader(
            self.reader,
            batch_size=self.batch_size,
            shuffling_queue_capacity=self.shuffling_queue_capacity,
        )

        for batch in data_loader:
            # batch = NamedDataset(batch, annotation=self.annotation)
            yield batch


class PetastormAsyncDataLoader(BaseAsyncDataLoader):
    def __init__(self, dataset, batch_size, shuffling_queue_capacity, q_size=64):
        super().__init__(q_size)

        if not isinstance(dataset, PetastormReaderWrapper):
            dataset = PetastormReaderWrapper(dataset)
        self.reader = dataset.reader
        self.annotation = dataset.annotation
        self.batch_size = batch_size
        self.shuffling_queue_capacity = shuffling_queue_capacity

    def __len__(self):
        return len(self.reader)

    def _iterate(self):
        if self.reader.last_row_consumed:
            print(f"Resetting Petastorm reader for {self.reader.dataset.paths}")
            self.reader.reset()

        data_loader = BatchedDataLoader(
            self.reader,
            batch_size=self.batch_size,
            shuffling_queue_capacity=self.shuffling_queue_capacity,
        )

        for batch in data_loader:
            # batch = NamedDataset(batch, annotation=self.annotation)
            yield batch
