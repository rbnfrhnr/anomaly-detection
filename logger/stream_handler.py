from logging import StreamHandler

import pandas as pd

record_cols = ['identifier', 'context', 'entity', 'item', 'value']


class CustomStreamHandler(StreamHandler):

    def __init__(self, log_dir):
        StreamHandler.__init__(self)
        self.log_dir = log_dir
        self.summary = pd.DataFrame(columns=record_cols)

    def emit(self, record):
        if hasattr(record, 'data') or hasattr(record, 'summary'):
            return
        super().emit(record)
