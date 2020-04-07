from horovod.spark.driver.driver_service import SparkDriverService
from horovod.run.elastic.discovery import HostDiscovery


class SparkDriverHostDiscovery(HostDiscovery):
    def __init__(self, driver):
        """
        :param driver: Spark driver service
        :type driver: SparkDriverService
        """
        self._driver = driver
        super(SparkDriverHostDiscovery, self).__init__()

    def find_available_hosts_and_slots(self):
        host_hash_indices = self._driver.task_host_hash_indices()
        slots = dict([(host, len(indices))
                      for host, indices in host_hash_indices.items()
                      if len(indices)])
        return slots
