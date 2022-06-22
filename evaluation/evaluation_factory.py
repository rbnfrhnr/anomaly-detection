from evaluation.default_evaluator import test_eval
from evaluation.mit_bih_evaluator import eval
from evaluation.ucr_evaluator import evaluate, evaluate2

def get_evaluator(dataset):
    if dataset == 'mit-bih':
        return eval
    if dataset == 'ucr':
        return evaluate2
    else:
        return test_eval
