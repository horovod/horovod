import sys
import os

from horovod.spark.util import codec, secret
from horovod.spark.driver.driver_service import BasicClient
from horovod.spark.util.network import NoValidAddressesFound


def main(service_name, addresses):
    """
    :param service_name:
    :param addresses:     # addresses = [(ip, port)]
    :return:
    """

    key = codec.loads_base64(os.environ[secret.HOROVOD_SECRET_KEY])
    valid_interfaces = {}
    try:
        service = BasicClient(service_name, addresses, key)
        print("CLIENT LAUNCH SUCCESSFUL.")
        valid_interfaces = service.addresses()

    except NoValidAddressesFound as e:
        print("CLIENT LAUNCH SUCCESSFUL.")

    print("SUCCESSFUL INTERFACE ADDRESSES {addresses} EOM.".format(
        addresses=codec.dumps_base64(valid_interfaces)))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s <service_name> <list of addresses>' % sys.argv[0])
        sys.exit(1)
    main(codec.loads_base64(sys.argv[1]), codec.loads_base64(sys.argv[2]))
