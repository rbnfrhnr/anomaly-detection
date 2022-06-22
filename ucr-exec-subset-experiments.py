import os
import sys
from glob import glob
from itertools import product

import yaml

import utils

if __name__ == '__main__':
    template = sys.argv[1]
    cfg = utils.read_cfg(template)
    subsets = [59, 114, 173, 6, 121, 193, 53, 35, 197, 70, 119, 22, 221,
               33, 62, 102, 123, 83, 249, 54, 236, 138, 229, 28]
    data_location = cfg['data']['location']
    data_files = [glob(data_location + '/*' + str(subset) + '_*')[0] for subset in subsets]
    base_augmentations = ['']
    # base_augmentations = ['', 'tenth-data', 'fifth-data', 'third-data', 'half-data', 'two-third-data',
    #                       'three-quarter-data', 'nine-tenth-data']
    base_aug_to_name = {'': '100', 'tenth-data': '10', 'fifth-data': '20', 'third-data': '33', 'half-data': '50',
                        'two-third-data': '66', 'three-quarter-data': '75', 'nine-tenth-data': '90'}
    augmentations = ['None', 'noise', 'reverse', 'time-shift', 'rvae-generate']
    augmentations_to_name = {'None': 'no-augmentation',
                             'noise': 'noise', 'reverse': 'reverse', 'time-shift': 'time-shift',
                             'rvae-generate': 'rvae-generate'}

    for i in range(1, 11):
        for comb in product(base_augmentations, augmentations, data_files):
            base_augmentation, augmentation, file = comb
            file_name = file.split('/')[-1]
            set_nr = file_name.split('_')[0]
            cfg['experiment-name'] = 'smoothing-' + augmentations_to_name[augmentation] + '-' + base_aug_to_name[
                base_augmentation]
            cfg['preprocessing']['augmentations'] = [augmentation]
            cfg['preprocessing']['base-augmentations'] = [base_augmentation]
            cfg['data']['train-sets'] = [set_nr]

            run_template_location = '/'.join(template.split('/')[0:-1]) + '/smoothing/run-templates/' + \
                                    augmentations_to_name[augmentation]
            runt_template_name = set_nr + '-' + cfg['experiment-name'] + '.yaml'
            run_file_name = run_template_location + '/' + runt_template_name
            os.makedirs(os.path.dirname(run_file_name), exist_ok=True)

            with open(run_file_name, 'w+') as yaml_file:
                yaml.dump(cfg, yaml_file, default_flow_style=False)
            print('running ', runt_template_name, 'iteration', i)

            os.system('python3 vae_rnn.py ' + run_file_name)

# for i in range(1, 101):
#     for file in data_files:
#         file_name = file.split('/')[-1]
#         set_nr = file_name.split('_')[0]
#         cfg['data']['train-sets'] = [set_nr]
#         cfg['experiment-name'] = expriment_name_template + '-' + set_nr
#
#         run_template_location = '/'.join(template.split('/')[0:-1]) + '/run-templates'
#         runt_template_name = template.split('/')[-1].split('.')[0] + '-' + set_nr + '.yaml'
#         run_file_name = run_template_location + '/' + runt_template_name
#         os.makedirs(os.path.dirname(run_file_name), exist_ok=True)
#         with open(run_file_name, 'w+') as yaml_file:
#             yaml.dump(cfg, yaml_file, default_flow_style=False)
#
#         os.system('python3 vae_rnn.py ' + run_file_name)
