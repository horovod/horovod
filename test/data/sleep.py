import os
import sys
import time

duration = int(sys.argv[1])
filename = sys.argv[2]
filename_tmp = filename + '.tmp'


def write(s):
    with open(filename_tmp, 'w') as f:
        f.write(str(s))

    # Atomic rename to prevent race conditions from reader
    os.rename(filename_tmp, filename)


write(os.getpid())
time.sleep(duration)
