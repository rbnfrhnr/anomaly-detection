from logging import StreamHandler
import pandas as pd
import wandb


class WandbHandler(StreamHandler):

    def __init__(self, **config):
        StreamHandler.__init__(self)
        wandb_group_name = config['experiment-name']
        wandb_project = config['logging']['wandb']['project']
        wandb_entity = config['logging']['wandb']['entity']
        wandb.init(project=wandb_project, entity=wandb_entity,
                   group=str(wandb_group_name))

    def emit(self, record):
        if hasattr(record, 'data'):
            info = getattr(record, 'data')
            context = info['context']
            entity = info['entity']
            item = info['item']
            # needs to be pandas dataframe
            data = info['data']
            table_name = context + '-' + entity + '-' + item
            wandb.log(
                {table_name: wandb.Table(
                    data=data.values.tolist(),
                    columns=data.columns.tolist())})
        if hasattr(record, 'summary'):
            info = getattr(record, 'summary')
            context = info['context']
            entity = info['entity']
            item = info['item']
            data = info['data']
            name = context + '-' + entity + '-' + item
            wandb.run.summary[name] = data
