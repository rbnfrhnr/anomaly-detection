import os
import sys
import utils

if __name__ == '__main__':
    for i in range(0, 100):
        os.system('python3 vae_rnn.py ' + sys.argv[1])
