import os
import sys
import utils

if __name__ == '__main__':
    for i in range(1, 31):
        os.system('python3 vae_rnn.py ' + sys.argv[1])
