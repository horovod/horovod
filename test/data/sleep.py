import os
import sys
import time

duration = int(sys.argv[1])
filename = sys.argv[2]


def write(s):
    with open(filename, 'w') as f:
        f.write(str(s))


write(os.getpid())
time.sleep(duration)
