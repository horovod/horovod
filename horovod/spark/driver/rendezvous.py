
from horovod.spark.driver.driver_service import SparkDriverService
from horovod.run.http.http_server import RendezvousServer


class SparkRendezvousServer(RendezvousServer):
    def __init__(self, driver, verbose):
        super(SparkRendezvousServer, self).__init__(verbose)

        self._driver = driver

    def init(self, host_alloc_plan):
        super(SparkRendezvousServer, self).init(host_alloc_plan)
        print('new host alloc plan: {}'.format(['{}:{} -> {}'.format(slot_info.hostname, slot_info.local_rank, slot_info.rank) for slot_info in host_alloc_plan]))
        ranks_to_indices = {}
        host_indices = self._driver.task_host_hash_indices()
        for slot_info in host_alloc_plan:
            ranks_to_indices[slot_info.rank] = host_indices[slot_info.hostname][slot_info.local_rank]
        print('new ranks to indices: {}'.format(ranks_to_indices))
        self._driver.set_ranks_to_indices(ranks_to_indices)
