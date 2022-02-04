import os
import sys
import utils

if __name__ == '__main__':
    for i in range(0, 100):
        group_suffix = sys.argv[1] if len(sys.argv) >= 2 else 'test'
        augment_type = sys.argv[2] if len(sys.argv) >= 3 else None
        print("starting augment type: " + str(augment_type))
        os.system('python3 vae_rnn.py ' + str(group_suffix) + ' ' + str(augment_type))
