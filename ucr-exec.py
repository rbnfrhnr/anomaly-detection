import os
import sys
import utils
from glob import glob
import yaml

if __name__ == '__main__':
    template = sys.argv[1]
    cfg = utils.read_cfg(template)
    data_location = cfg['data']['location']
    expriment_name_template = cfg['experiment-name']
    data_files = glob(data_location + '/*_*')

    for file in data_files:
        file_name = file.split('/')[-1]
        set_nr = file_name.split('_')[0]
        cfg['data']['train-sets'] = [set_nr]
        cfg['experiment-name'] = expriment_name_template + '-' + set_nr

        run_template_location = '/'.join(template.split('/')[0:-1]) + '/run-templates'
        runt_template_name = template.split('/')[-1].split('.')[0] + '-' + set_nr + '.yaml'
        run_file_name = run_template_location + '/' + runt_template_name
        os.makedirs(os.path.dirname(run_file_name), exist_ok=True)
        with open(run_file_name, 'w+') as yaml_file:
            yaml.dump(cfg, yaml_file, default_flow_style=False)

        os.system('python3 vae_rnn.py ' + run_file_name)
