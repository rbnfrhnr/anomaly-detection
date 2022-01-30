import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import random
import math
import torch
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, f1_score
from model import *
from preprocessing import *
from dataset import *
from config import *
from loss import *
import pandas as pd
from dataloader import *
import scipy
from scipy.stats import norm

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def recon_distribution(y):
    param = norm.fit(y)
    return param

def train(args):
    # writer = SummaryWriter(args.log_dir + args.timestamp + '_' + args.config)
    print("device", args.device)
    # criterion = nn.MSELoss()
    current_loss = 0
    if args.model == "RNN":
        model = RNN_VAE(args.input_size, args.hidden_size, args.num_layer, args.batch_size, \
                        args.seq_len, args.latent_size).to(args.device)
    elif args.model == 'DNN':
        model = DNN_VAE(args.dims, args.activation).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.model == "RNN":
        if "pre_vae_3456710111213_lstm.csv" in os.listdir('../ctu-13/CTU-13-Dataset'):
            dis_df = pd.read_csv('../ctu-13/CTU-13-Dataset/pre_vae_3456710111213_lstm.csv', index_col=[0])
            print("load train dataframe")
        else:
            df = make_conn_dataframe(["3","4","5","7","10","11","12","13"])
            # dis_df = preprocess_stat_lstm_2(df)
            dis_df = preprocess_CTU(df)
            dis_df.to_csv('../ctu-13/CTU-13-Dataset/pre_vae_3456710111213_lstm.csv')
            print("saved train preprocessed file")
        
        # if "pre_vae_12689_lstm.csv" in os.listdir('../ctu-13/CTU-13-Dataset'):
        #     test_df = pd.read_csv('../ctu-13/CTU-13-Dataset/pre_vae_12689_lstm.csv', index_col=[0])
        #     print("load test dataframe")
        # else:
        #     test_df = make_conn_dataframe(["1","2","6","8","9"])
        #     test_df = preprocess_stat_lstm_2(test_df)
        #     test_df.to_csv('../ctu-13/CTU-13-Dataset/pre_vae_12689_lstm.csv')
        #     print("saved test preprocessed file") 
    else:
        if "pre_vae_3456710111213.csv" in os.listdir('../ctu-13/CTU-13-Dataset'):
            dis_df = pd.read_csv('../ctu-13/CTU-13-Dataset/pre_vae_3456710111213.csv', index_col=[0])
            print("load train dataframe")
        else:
            df = make_conn_dataframe(["3","4","5","7","10","11","12","13"])
            # dis_df = preprocess_stat(df)
            dis_df = preprocess_CTU(df)
            dis_df.to_csv('../ctu-13/CTU-13-Dataset/pre_vae_3456710111213.csv')
            print("saved train preprocessed file")
        
        if "pre_vae_12689.csv" in os.listdir('../ctu-13/CTU-13-Dataset'):
            test_df = pd.read_csv('../ctu-13/CTU-13-Dataset/pre_vae_12689.csv', index_col=[0])
            print("load test dataframe")
        else:
            test_df = make_conn_dataframe(["1","2","6","8","9"])
            # test_df = preprocess_stat(test_df)
            test_df = preprocess_CTU(df)
            test_df.to_csv('../ctu-13/CTU-13-Dataset/pre_vae_12689.csv')
            print("saved test preprocessed file") 

    label_dict = {"normal":0, "botnet":1}
    
    dis_df["label_num"] = dis_df["class"].apply(lambda x : label_dict[x])
    test_df["label_num"] = test_df["class"].apply(lambda x : label_dict[x])
    print("train/valid set :", Counter(dis_df["class"]))
    # print("test set :", Counter(test_df["class"]))
    same_column = list(set(test_df.columns).intersection(set(dis_df.columns)))
    # print("same_column", same_column)
    # same_column = list(dis_df.columns)
    norm_df = dis_df[dis_df["label_num"]==0]
    traindataset = NetworkDataset_ae(norm_df, "train",  args.model, same_column)
    # traindataset = NetworkDataset_ae(args.seq_len, norm_df, "train",  args.model, same_column)
    # traindataset = NetworkDataset_ae(norm_df, "train",  args.model, same_column, args.preprocess, args.total_seq, args.overlap, 5, args.dataset)

    if args.model == 'RNN':
        trainloader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("made train data loader")
    
    validdataset = NetworkDataset_ae(args.seq_len, norm_df, "valid", args.model, same_column)
    if args.model == 'RNN':
        validloader = LSTM_VAE_dataloader(validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        validloader = DataLoader(validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    mix_validdataset = NetworkDataset_ae(args.seq_len, dis_df, "valid", args.model, same_column)
    if args.model == 'RNN':
        mix_validloader = LSTM_VAE_dataloader(mix_validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        mix_validloader = DataLoader(mix_validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("made valid data loader")
    testdataset = NetworkDataset_ae(args.seq_len, test_df, "test", args.model, same_column)
    if args.model == 'RNN':
        testloader = LSTM_VAE_dataloader(testdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        testloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("made test data loader")
    for e in range(args.epochs):
        current_loss = 0
        for i, data in enumerate(trainloader):
            model.train()
            model.zero_grad()
            if args.model == 'DNN':
                feature, _ = data
                feature = feature.to(args.device)
                mu, logvar, recon = model(feature)
            else:
                encode_x, _ = data
                encode_x = encode_x.to(args.device)
                # feature = feature.to(args.device)
                mu, logvar, recon, feature = model(encode_x)
            loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if i % args.log_frequency== 0:
                print(f'progress : {i/len(trainloader)*100 :.0f}%, batch loss : {loss.item():.3f}')
            # if i % args.valid_frequency == 0 and i // args.valid_frequency > 0:
        print("validation start")
        val_total_loss = 0
        model.eval()
        for k, data in enumerate(validloader):
            model.zero_grad()
            if args.model == 'DNN':
                feature, _ = data
                feature = feature.to(args.device)
                mu, logvar, recon = model(feature)
            else:
                encode_x, _ = data
                encode_x = encode_x.to(args.device)
                mu, logvar, recon, feature = model(encode_x)
            val_loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model)
            val_loss = val_loss.mean()
            val_total_loss += val_loss.item()
            if k % args.log_frequency == 0:
                print(f'validation progress : {k/len(validloader)*100:.0f}%, validation loss : {val_loss.item():.3f}')
        print(f'epoch: {e} / loss :{current_loss / (i+1):.3f}')
        
        label_lst = []
        test_label_lst = []
        recon_lst_val = []
        recon_lst_test = []
        for k, data in enumerate(mix_validloader):
            model.eval()
            if args.model == 'DNN':
                feature, label = data
                feature = feature.to(args.device)
                mu, logvar, recon = model(feature)
                label = label.to(args.device)
            else:
                encode_x, label = data
                encode_x = encode_x.to(args.device)
                mu, logvar, recon, feature = model(encode_x)
            val_loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model)
            if args.model == 'DNN':
                recon_lst_val.extend(val_loss.cpu().detach().numpy())
                label_lst.extend(label.squeeze().cpu().detach().numpy())
            else:
                bce = bce.view(1,-1)
                recon_lst_val.extend(bce.cpu().detach().numpy())
                label_lst.extend(label)
        ## distribution
        if args.model == 'RNN':
            flatten_l = [l for ls in label_lst for l in ls]
            print(flatten_l)
            flatten_recon = [rec for recs in recon_lst_val for rec in recs]
            print(len(flatten_recon), len(flatten_l))
        
        error_df = pd.DataFrame({'Reconstruction_error': recon_lst_val, 'True_class': label_lst})
        label_lst = np.asarray(label_lst)    
        recon_lst = np.asarray(recon_lst_val)
        botnet_index = (label_lst == 1).nonzero()[0]
        print("the number of botnet_index in validation set :", len(botnet_index))
        abnorm_param = recon_distribution(recon_lst[botnet_index])
        norm_param = recon_distribution(recon_lst[~botnet_index])
        # abnorm_err = recon_lst[botnet_index].mean()
        # norm_err = recon_lst[~botnet_index].mean()
        print("Mean of Abnorm error :", abnorm_param[0])
        print("Mean of Norm error :", norm_param[0])
        pred_y = list()
        for e in error_df.Reconstruction_error.values:
            if norm.pdf(e,loc=abnorm_param[0],scale=abnorm_param[1]) > norm.pdf(e,loc=norm_param[0],scale=norm_param[1]):
                pred_y.append(1)
            else:
                pred_y.append(0)
        # pred_y = [1 if e > fixed_threshold else 0 for e in error_df.Reconstruction_error.values]
        # pred_y = [1 if e > current_loss/(i+1)*1.5 else 0 for e in error_df.Reconstruction_error.values]
        abnorm_num = len((np.asarray(label_lst)==1).nonzero()[0])
        abnorm_predict = np.asarray(pred_y)[(np.asarray(label_lst)==1).nonzero()[0]]
        abnorm_cor = (abnorm_predict == 1).sum()
        abnorm_pre = (np.asarray(pred_y)==1).sum()
        valid_recall = abnorm_cor / abnorm_num
        valid_precision = abnorm_cor / abnorm_pre
        print("valid recall : ", valid_recall)
        print("valid precision : ", valid_precision)
        auc = roc_auc_score(error_df.True_class, pred_y)
        f1 = f1_score(error_df.True_class, pred_y)
        # print(f"threshold :{current_loss/(i+1)*1.5} | valid AUC :{auc} | f1 score :{f1}")
        print(f"valid AUC :{auc} | f1 score :{f1}")
        conf_matrix = confusion_matrix(error_df.True_class, pred_y)
        print(f"epooch : {e} \n valid confusion matrix\n", conf_matrix)

        for w, data in enumerate(testloader):
            if args.model == 'DNN':
                feature, label = data
                feature = feature.to(args.device)
                mu, logvar, recon = model(feature)
            else:
                encode_x, decode_x, label = data
                encode_x = encode_x.to(args.device)
                decode_x = decode_x.to(args.device)
                mu, logvar, recon, feature = model(encode_x, decode_x)
            label = label.to(args.device)
            mu, logvar, recon = model(feature)
            testloss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model)
            if args.model == 'DNN':
                recon_lst_test.extend(testloss.cpu().detach().numpy())
                test_label_lst.extend(label.squeeze().cpu().detach().numpy())
            else:
                bce = bce.view(1,-1)
                recon_lst_test.extend(bce.cpu().detach().numpy())
                test_label_lst.extend(label) 
        error_df = pd.DataFrame({'Reconstruction_error': recon_lst_test, 'True_class': test_label_lst})
        label_lst = np.asarray(test_label_lst)    
        recon_lst = np.asarray(recon_lst_test)
        botnet_index = (label_lst == 1).nonzero()[0]
        print("the number of botnet_index test set :", len(botnet_index))
        abnorm_param = recon_distribution(recon_lst[botnet_index])
        norm_param = recon_distribution(recon_lst[~botnet_index])

        pred_y = list()
        for e in error_df.Reconstruction_error.values:
            if norm.pdf(e,loc=abnorm_param[0],scale=abnorm_param[1]) > norm.pdf(e,loc=norm_param[0],scale=norm_param[1]):
                pred_y.append(1)
            else:
                pred_y.append(0)
        # pred_y = [1 if e > fixed_threshold else 0 for e in error_df.Reconstruction_error.values]
        # pred_y = [1 if e > current_loss/(i+1)*1.5 else 0 for e in error_df.Reconstruction_error.values]
        abnorm_num = len((np.asarray(test_label_lst)==1).nonzero()[0])
        abnorm_predict = np.asarray(pred_y)[(np.asarray(test_label_lst)==1).nonzero()[0]]
        abnorm_cor = (abnorm_predict == 1).sum()
        abnorm_pre = (np.asarray(pred_y)==1).sum()
        test_recall = abnorm_cor / abnorm_num
        test_precision = abnorm_cor / abnorm_pre
        print("test recall : ", test_recall)
        print("test precision : ", test_precision)
        auc = roc_auc_score(error_df.True_class, pred_y)
        f1 = f1_score(error_df.True_class, pred_y)
        # print(f"threshold :{current_loss/(i+1)*1.5} | test AUC :{auc} | f1 score :{f1}")
        print(f"test AUC :{auc} | f1 score :{f1}")
        conf_matrix = confusion_matrix(error_df.True_class, pred_y)
        print(f"epooch : {e} \n test confusion matrix\n", conf_matrix)
    
    
if __name__ == "__main__":
    train(get_config())