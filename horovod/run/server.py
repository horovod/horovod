import sys
import os

from horovod.spark.util import codec, secret
from horovod.spark.driver.driver_service import BasicService


def main(service_name):
    key = codec.loads_base64(os.environ[secret.HOROVOD_SECRET_KEY])
    try:
        service = BasicService(service_name, key)
        print('SERVER LAUNCH SUCCESSFUL on '
              '{service_name}.'.format(
            service_name=codec.dumps_base64(service_name)))
        print('PORT IS: {port} EOM'.format(
            port=codec.dumps_base64(service.get_port())))
    except Exception as e:
        print('SERVER LAUNCH FAILED.')
        print(e.message)
        exit(1)

    while True:
        pass


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <driver addresses>' % sys.argv[0])
        sys.exit(1)
    main(codec.loads_base64(sys.argv[1]))
