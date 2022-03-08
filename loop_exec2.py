import os
import sys
import utils

if __name__ == '__main__':
    bih_sets = [
        './config/mit-bih/803/apu/augmentation-noise-bih-803.yaml'
        , './config/mit-bih/803/apu/augmentation-reverse-bih-803.yaml'
        , './config/mit-bih/803/apu/augmentation-time-shift-bih-803.yaml'
        , './config/mit-bih/803/apu/no-augmentation-bih-803.yaml'
    ]
    for i in range(0, 100):
        for bih in bih_sets:
            os.system('python3 vae_rnn.py ' + bih)
