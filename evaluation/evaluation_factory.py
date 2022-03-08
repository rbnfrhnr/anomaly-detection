from evaluation.default_evaluator import test_eval
from evaluation.mit_bih_evaluator import eval

def get_evaluator(dataset):
    if dataset == 'mit-bih':
        return eval
    else:
        return test_eval
