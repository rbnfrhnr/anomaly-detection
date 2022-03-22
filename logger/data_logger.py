from logging import StreamHandler

import pandas as pd

record_cols = ['identifier', 'context', 'entity', 'item', 'value']


class DataHandler(StreamHandler):

    def __init__(self, log_dir):
        StreamHandler.__init__(self)
        self.log_dir = log_dir
        self.summary = pd.DataFrame(columns=record_cols)

    def emit(self, record):
        if hasattr(record, 'data'):
            info = getattr(record, 'data')
            context = info['context']
            entity = info['entity']
            item = info['item']
            # needs to be pandas dataframe
            data = info['data']
            data.to_csv(self.log_dir + '/' + context + '-' + entity + '-' + item + '.csv')
        if hasattr(record, 'summary'):
            info = getattr(record, 'summary')
            context = info['context']
            entity = info['entity']
            item = info['item']
            data = info['data']
            name = context + '-' + entity + '-' + item
            rec = pd.DataFrame(columns=record_cols, data=[[name, context, entity, item, data]])
            self.summary = pd.concat([self.summary, rec])
            print(rec.to_records())
            self.summary.to_csv(self.log_dir + '/summary.csv')
