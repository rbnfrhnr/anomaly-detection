import argparse
import torch
from datetime import datetime
from pathlib import Path
import re
home = str(Path.home())


def get_config():
    parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--model', default='DNN', type=str)
    # model_arg.add_argument('--model', default='RNN', type=str)
    model_arg.add_argument('--activation', default='relu', type=str)
    #DNN
    model_arg.add_argument('--dims', default=[23, 10, 5], type=list)
    # model_arg.add_argument('--dims', default=[95, 30, 10], type=list)
    #RNN
    model_arg.add_argument('--input-size', default=15, type=int)
    model_arg.add_argument('--hidden-size', default=30, type=int)
    model_arg.add_argument('--latent-size', default=10, type=int)
    model_arg.add_argument('--dropout', default=0.2, type= float)
    model_arg.add_argument('--num-layer', default=1, type=int)
    model_arg.add_argument('--bidirectional', action='store_true')

    data_arg = parser.add_argument_group('Data')
    # data_arg.add_argument('--log-dir', default='CTU-13-Dataset/1, type=str, help='directory of training/testing data (default: datasets)')
    data_arg.add_argument('--dataset', default='CTU', type=str)
    data_arg.add_argument('--preprocess', default='small', type=str)
    data_arg.add_argument('--period-len-ctu', default=60, type=int)
    data_arg.add_argument('--sample-rate-ctu', default=0.9, type=float)
    data_arg.add_argument('--sample-rate-kisti', default=0.15, type=float)
    data_arg.add_argument('--normal-select-rate', default=0.5, type=float)
    data_arg.add_argument('--period-len-kisti', default=20, type=int)
    data_arg.add_argument('--total-seq', default=3, type=int)
    data_arg.add_argument('--onlyconnect', action='store_true')
    data_arg.add_argument('--only-shared', action='store_true')
    data_arg.add_argument('--background', action='store_true')
    data_arg.add_argument('--overlap', action='store_true')
    data_arg.add_argument('--sampling-data', action='store_true')
    data_arg.add_argument('--rm-ntp', action='store_true')
    data_arg.add_argument('--all-scenario', action='store_true')
    data_arg.add_argument('--trainset', default=["3","4","5","7","10","11","12","13"], type=list)
    data_arg.add_argument('--testset', default=["1","2","6","8","9"], type=list)
    data_arg.add_argument('--scenario', default='all', type=str)
 
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--zeek', action='store_true')
    train_arg.add_argument('--KLD-W', action='store_true')
    train_arg.add_argument('--normal-select', action='store_true')
    train_arg.add_argument('--only-infer', action='store_true')
    train_arg.add_argument('--valid-distribution', action='store_true')
    train_arg.add_argument('--device', default=0, type=int)
    train_arg.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
    train_arg.add_argument('--epochs', default=500, type=int, help='number of total epochs (default: 200)')
    train_arg.add_argument('--x0', default=500, type=int)
    train_arg.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.05)')
    train_arg.add_argument('--lamb', default=1, type=float)
    train_arg.add_argument('--log-frequency', default=50, type=int)
    train_arg.add_argument('--eval-frequency', default=50, type=int)
    train_arg.add_argument('--evaluation', default='no-threshold', type=str)
    # train_arg.add_argument('--evaluation', default='threshold', type=str)
    # train_arg.add_argument('--valid-frequency', default=7000, type=int)
    train_arg.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M%S"), type=str)
    train_arg.add_argument('--load-model', default=None, type=str)
    train_arg.add_argument('--log-dir', default='saved/runs/', type=str)
    train_arg.add_argument('--threshold-fixed', default=0.3, type=float)
    train_arg.add_argument('--memo', default=None, type=str)

    args = parser.parse_args()

    # if args.dims[-1] != args.n_categories:
        # parser.error('dimension should be same with category number')
    # if args.dims[0] != args.input_size:
    #     parser.error('dimension should be same with feature number')
    if args.preprocess == 'small':
        # args.dims = [15, 10, 5]
        if args.rm_ntp:
            # args.dims = [22, 10, 5]
            args.input_size = 22
        else:
            # args.dims = [23, 10, 5]
            # args.dims = [48, 512, 512, 1024, 100]
            args.dims = [23, 512, 512, 1024, 100]
            args.input_size = 48
            # args.input_size = 23
    elif args.preprocess =="large":
        args.log_frequency = 500
        args.dims = [94, 30, 10]
        args.input_size = 96
        
    if args.model == 'DNN':
        args.latent_size = args.dims[-1]
        # args.epochs = 250
        # args.period_len_kisti = 15
    if args.all_scenario:
        # args.trainset = ["3","4","5","7","10","11","12","13", "1","2","6","8","9"]
        args.trainset = ["1","2","9"]
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if not args.memo:
        config_list = [args.dataset, args.model, args.preprocess, args.hidden_size, args.input_size, args.lr, args.bidirectional, \
                        args.onlyconnect, args.period_len_ctu, args.period_len_kisti, args.num_layer, args.KLD_W, args.x0, \
                            args.lamb, args.rm_ntp, args.sampling_data, args.normal_select, args.evaluation, args.total_seq, \
                                args.only_shared, args.valid_distribution]
    else:
        config_list = [args.dataset, args.model, args.preprocess, args.hidden_size, args.input_size, args.lr, \
                        args.bidirectional, args.onlyconnect, args.period_len_ctu, args.period_len_kisti, args.num_layer, args.KLD_W, \
                             args.x0, args.lamb, args.rm_ntp, args.sampling_data, args.normal_select, args.evaluation, args.total_seq, \
                                 args.only_shared, args.valid_distribution, args.memo]
    args.config = '_'.join(list(map(str, config_list))).replace("/", ".")
    return args
