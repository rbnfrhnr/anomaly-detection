import os
import sys
import utils
from glob import glob
import yaml
from itertools import product

CONFIG_LOCATION = './config/ucr/run-templates/window-size-dset-139'

if __name__ == '__main__':
    config_files = glob(CONFIG_LOCATION + '/*.yaml')

    for i in range(1, 11):
        for config_file in config_files:
            os.system('python3 vae_rnn.py ' + config_file)
            print("done with ", config_file, " iteration ", i)
