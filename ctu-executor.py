import os
import sys
import utils
from glob import glob

if __name__ == '__main__':

    files_to_exec = glob('./config/ctu-13/medium/**/*.yaml')
    for i in range(1, 31):
        for file in files_to_exec:
            os.system('python3 vae_rnn.py ' + file)
