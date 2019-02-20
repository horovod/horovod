import sys

from horovod.spark.util import codec

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: %s <pickled function obj path> '
              '<pickled arg obj path>' % sys.argv[0])
        sys.exit(1)

    fn, arg = codec.loads_base64(sys.argv[1])
    results = fn(*arg)

    output_format = 'RESULT: {result} EOM'
    print('FUNCTION SUCCESSFULLY EXECUTED.')
    print(output_format.format(result=codec.dumps_base64(results)))
