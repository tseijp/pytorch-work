import sys
import time
for i in range(0, 3):
    sys.stdout.write("hello ! [Count {0:04d}]".format(i))
    time.sleep(1)
    with open('log.txt', mode='w') as f:
        f.write('hello')
