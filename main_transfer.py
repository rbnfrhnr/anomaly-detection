import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import random
import math
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, f1_score, roc_curve
from sklearn import metrics
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence
from model import *
from preprocessing import *
from dataset import *
from config import *
from loss import *
from dataloader import *
import scipy
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import norm
import warnings
from utils import *

warnings.filterwarnings('ignore')

def Load_Dataset(args):
    #source domain
    trainset_name = ''.join(args.trainset)
    testset_name = ''.join(args.testset)
    # if f"pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv" in os.listdir('./data/CTU-13-Dataset'):
    #     dis_df = pd.read_csv(f'./data/CTU-13-Dataset/pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv',\
    #                         index_col=[0])
    # else:  
    #     df = make_conn_dataframe(args.trainset)
    #     dis_df = preprocess_stat_2_CTU(df, args.period_len_ctu, args.rm_ntp)
    #     dis_df.to_csv(f'./data/CTU-13-Dataset/pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv')
    if f"preprocessed_129_{args.background}_{args.period_len_ctu}.csv" in os.listdir('./data/CTU-13-Dataset'):
        dis_df = pd.read_csv(f'./data/CTU-13-Dataset/preprocessed_129_{args.background}_{args.period_len_ctu}.csv',index_col=[0])
    else:
        df = pd.read_csv('./data/CTU-13-Dataset/ctu_129_bro_result.csv', index_col=[0])
        dis_df = preprocess_stat_3(df, args.period_len_ctu, args.rm_ntp)
        dis_df.to_csv(f'./data/CTU-13-Dataset/preprocessed_129_{args.background}_{args.period_len_ctu}.csv')

    norm_sample = dis_df[dis_df["class"]=='normal'].sample(frac=args.sample_rate_ctu, replace=False, random_state=1)
    norm_sample_ips = norm_sample["id.orig_h"].unique()
    remove_idx = dis_df.index[dis_df["id.orig_h"].isin(norm_sample_ips)].tolist()
    dis_df = dis_df.drop(remove_idx)
    if args.all_scenario:
        # total_df = dis_df
        # dis_df, test_df = train_test_split(total_df, test_size=0.3, random_state=1234)
        test_df = dis_df[:int(0.3*len(dis_df))]
        dis_df = dis_df[int(0.3*len(dis_df)):]
        
    label_dict = {"normal":0, "botnet":1}
    # label_dict = {"normal":0, "botnet":1, "background":0}
    dis_df["label_num"] = dis_df["class"].apply(lambda x : label_dict[x])

    print("Source Domain train/valid set :", Counter(dis_df["class"]))

    same_column = list(dis_df.columns)
    norm_df = dis_df[dis_df["label_num"]==0]
    traindataset = NetworkDataset_ae(norm_df, "train",  args.model, same_column, args.preprocess, args.total_seq, \
                                        args.overlap, 5, 'transfer')
    print("ctu trainset", len(traindataset))
    if args.model == 'RNN':
        sr_norm_trainloader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        sr_norm_trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    abnorm_df = dis_df[dis_df["label_num"]==1]
    traindataset = NetworkDataset_ae(abnorm_df, "train", args.model, same_column, args.preprocess, args.total_seq, args.overlap, 5, 'transfer')
    if args.model == 'RNN':
        sr_abnorm_trainloader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        sr_abnorm_trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("made train data loader")
    
    validdataset = NetworkDataset_ae(norm_df, "valid", args.model, same_column, args.preprocess, args.total_seq, args.overlap, 5, 'transfer')
    print("ctu validset", len(validdataset))
    if args.model == 'RNN':
        sr_norm_validloader = LSTM_VAE_dataloader(validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        sr_norm_validloader = DataLoader(validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    validdataset = NetworkDataset_ae(abnorm_df, "valid", args.model, same_column, args.preprocess, args.total_seq, args.overlap, 5, 'transfer')
    if args.model == 'RNN':
        sr_abnorm_validloader= LSTM_VAE_dataloader(validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        sr_abnorm_validloader = DataLoader(validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
    mix_validdataset = NetworkDataset_ae(dis_df, "valid", args.model, same_column, args.preprocess, args.total_seq, args.overlap, 5, 'transfer')
    if args.model == 'RNN':
        mix_validloader = LSTM_VAE_dataloader(mix_validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        mix_validloader = DataLoader(mix_validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("made valid data loader")

    eval_abnorm_rate = Counter(dis_df["class"])["botnet"] / len(dis_df)
    # if f"pre_vae_{testset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv" in os.listdir('./data/CTU-13-Dataset'):
    #     test_df = pd.read_csv(f'./data/CTU-13-Dataset/pre_vae_{testset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv', \
    #                         index_col=[0])
    #     print("load test dataframe")
    # else:
    #     test_df = make_conn_dataframe(args.testset)
    #     test_df = preprocess_stat_2_CTU(test_df, args.period_len_ctu, args.rm_ntp)
    #     test_df.to_csv(f'./data/CTU-13-Dataset/pre_vae_{testset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv')
    #     print("saved test preprocessed file") 

    print("test set :", Counter(test_df["class"]))
    label_dict = {"normal":0, "botnet":1}
    test_df["label_num"] = test_df["class"].apply(lambda x : label_dict[x])
    test_abnorm_rate = Counter(test_df["class"])["botnet"] / len(test_df)
    testdataset = NetworkDataset_ae(test_df, "test", args.model, same_column, args.preprocess, args.total_seq, \
                                        args.overlap, 5, 'transfer')
    if args.model == 'RNN':
        testloader = LSTM_VAE_dataloader(testdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        testloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("made test data loader")

    #target domain
    if args.sampling_data:
        if args.only_shared:
            if f"pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}_shared.csv" in os.listdir('./data/KISTI'):
                target_df = pd.read_csv(f'./data/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}_shared.csv',index_col=[0])
            else:
                df = pd.read_csv('data/KISTI/kisti_logs_20190714_6_shared.csv')
                target_df = preprocess_stat_KISTI(df, args.period_len_kisti, args.rm_ntp)
                target_df.to_csv(f'./data/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}_shared.csv')
        else:
            # if f"pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv" in os.listdir('./data/KISTI'):
            #     target_df = pd.read_csv(f'./data/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv',index_col=[0])
            # else:
            #     df = pd.read_csv('data/KISTI/kisti_logs_20190714_6.csv')
            #     target_df = preprocess_stat_KISTI(df, args.period_len_kisti, args.rm_ntp)
            #     target_df.to_csv(f'./data/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv')
            if f"preprocessed_kisti_{args.period_len_kisti}.csv" in os.listdir('./data/KISTI'):
                target_df = pd.read_csv(f'./data/KISTI/preprocessed_kisti_{args.period_len_kisti}.csv', index_col=[0])
            else:
                df = pd.read_csv('./data/KISTI/kisti_bro_result.csv')
                target_df = preprocess_stat_3(df, args.period_len_kisti, args.rm_ntp)
                target_df.to_csv(f'./data/KISTI/preprocessed_kisti_{args.period_len_kisti}.csv')
        print("saved target preprocessed file")
    
    norm_sample = target_df[target_df["class"]=='norm'].sample(frac=args.sample_rate_kisti, replace=False, random_state=1)
    norm_sample_ips = norm_sample["id.orig_h"].unique()
    remove_idx = target_df.index[target_df["id.orig_h"].isin(norm_sample_ips)].tolist()
    target_df = target_df.drop(remove_idx)

    label_dict = {"norm":0, "abnorm":1}
    target_df["label_num"] = target_df["class"].apply(lambda x : label_dict[x])
    print("kisti total dataset :", Counter(target_df["class"]))
    print("kisti length of df", len(target_df))
    
    train_df = target_df[:int(0.7*len(target_df))]
    test_df = target_df[int(0.7*len(target_df)):]

    print("kisti test dataset :", Counter(test_df["class"]))
    
    target_norm_df = train_df[train_df["label_num"]==0]
    kisti_abnorm_rate = Counter(train_df["class"])["abnorm"] / len(train_df)

    same_column = list(train_df.columns)
    if args.normal_select:
        traindataset = NetworkDataset_ae(train_df, "train",  args.model, same_column, args.preprocess, args.total_seq, \
                                            args.overlap, 5, 'transfer')
    else:
        traindataset = NetworkDataset_ae(target_norm_df, "train",  args.model, same_column, args.preprocess, args.total_seq, \
                                            args.overlap, 5, 'transfer')
    print("kisti trainset", len(traindataset))
    if args.model == 'RNN':
        if args.normal_select:
            kisti_trainloader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size-int(args.batch_size * args.normal_select_rate), \
                                                    shuffle=True, num_workers=2)
        else:
            kisti_trainloader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        kisti_trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("made train target data loader")
    
    # validdataset = NetworkDataset_ae(target_norm_df, "valid", args.model, same_column, args.preprocess, \
    #                                     args.total_seq, args.overlap, 5, 'transfer')
    validdataset = NetworkDataset_ae(train_df, "valid", args.model, same_column, args.preprocess, args.total_seq,\
                                        args.overlap, 5, "transfer")
    print("kisti valid", len(validdataset))
    if args.model == 'RNN':
        kisti_validloader = LSTM_VAE_dataloader(validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        kisti_validloader = DataLoader(validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("made valid target data loader")
    print("the number of target domain data", len(target_df))

    # inference_kisti = NetworkDataset_ae(target_df, "test",  args.model, same_column, args.preprocess, args.total_seq, args.overlap)
    inference_kisti = NetworkDataset_ae(test_df, "test",  args.model, same_column, args.preprocess, \
                                        args.total_seq, args.overlap, 5, 'transfer')
    print("# of inference dataset :", len(inference_kisti))
    if args.model == 'DNN':
        inference_loader = DataLoader(inference_kisti, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        inference_loader = LSTM_VAE_dataloader(inference_kisti, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return sr_norm_trainloader, sr_abnorm_trainloader, sr_norm_validloader, sr_abnorm_validloader, mix_validloader, testloader, \
                    kisti_trainloader, kisti_validloader, target_df, inference_loader,\
                        eval_abnorm_rate, test_abnorm_rate, kisti_abnorm_rate

def inference(args, model, inference_loader, target_df, writer, e, model_dir, test_abnorm_rate, dists_info):
    '''get z, decoder and calculate reconstruction error'''
    if args.only_infer:
        save_dir = args.log_dir + args.load_model
    else:
        save_dir = args.log_dir + args.timestamp + '_' + args.config + '_transfer'
    # inference part   
    model.to(args.device)
    model.eval()
    abnorm_scores_lst = []
    label_lst = []
    # method 
    if isinstance(e, int):
        z_d = torch.load(f"{model_dir}/z_{args.device}.pt")
    else:
        z_d = torch.load(f"{model_dir}/best_z_{args.device}.pt")
    print("load z")
    for d in inference_loader:
        if args.model == 'DNN':
            feature, label = d
        else:
            feature, label, _ = d
        feature = feature.to(args.device)
        if args.model == 'DNN':
            if feature.size()[0] != args.batch_size:
                break
            recon = model.decoder(z_d)
            abnorm_score = F.binary_cross_entropy(recon, feature, reduction='none').sum(1)
            abnorm_scores_lst.extend(abnorm_score.cpu().detach().numpy())
            label_lst.extend(label.squeeze().cpu().detach().numpy())
        else:
            padded, lens = pad_packed_sequence(feature, batch_first=True)
            padded = padded.to(args.device)
            decoded_input = padded[:,1:,:] #to make rnn decoder input
            m = nn.ConstantPad2d((0, 0, 1, 0), 0.0)
            inputs_decoder = m(decoded_input)
            inputs_decoder = pack_padded_sequence(inputs_decoder, lens, batch_first=True)
            if padded.size()[0] != args.batch_size:
                break
            recon = model.decoder(inputs_decoder, z_d)
            recon_score = inference_loss(recon, padded, lens, args.device)
            recon_score = recon_score.cpu().detach().numpy()
            recon_score_selected = recon_score.nonzero() # to flatten bce loss, discard masked value
            abnorm_score = recon_score[recon_score_selected]
            abnorm_scores_lst.extend(abnorm_score)
            label_lst.extend([l for l in label]) 
    if args.model == 'RNN':
        flatten_l = [l for ls in label_lst for l in ls]
        label_lst = flatten_l

    error_df = pd.DataFrame({'Reconstruction_error': abnorm_scores_lst, 'True_class':label_lst})
    if args.evaluation == 'threshold':
        asl = np.asarray(abnorm_scores_lst)
        sort_idx = np.argsort(asl)
        lens = len(asl)
        abnorm_idx = sort_idx[-int(lens*test_abnorm_rate):]
        pred_y = np.zeros_like(asl)
        pred_y[abnorm_idx] = 1
    else:
        label_lst = np.asarray(label_lst)    
        # print("here", len(label_lst))
        recon_lst = np.asarray(abnorm_scores_lst)
        abnorm_index = (label_lst == 1).nonzero()[0]
        print("the number of abnormal index in inference set :", len(abnorm_index))
        mask = np.ones(len(recon_lst), dtype=bool)
        mask[abnorm_index] = False
        if args.valid_distribution:
            abnorm_dist_n, abparams, norm_dist, params = dists_info
            abarg = abparams[:-2]
            abloc = abparams[-2]
            abscale = abparams[-1]
            best_dist_abnorm = getattr(st, abnorm_dist_n)
            narg = params[:-2]
            nloc = params[-2]
            nscale = params[-1]
            best_dist_norm = getattr(st, norm_dist)
        else:
            abnorm_dist_n, params = best_fit_distribution(recon_lst[abnorm_index])
            abarg = params[:-2]
            abloc = params[-2]
            abscale = params[-1]
            best_dist_abnorm = getattr(st, abnorm_dist_n)
            norm_dist, params = best_fit_distribution(recon_lst[mask])
            narg = params[:-2]
            nloc = params[-2]
            nscale = params[-1]
            best_dist_norm = getattr(st, norm_dist)
        pred_y = list()
        pred_y_auc = list()
        for er in error_df.Reconstruction_error.values:
            if best_dist_abnorm.pdf(er, loc=abloc, scale=abscale, *abarg) > best_dist_norm.pdf(er, loc=nloc, scale=nscale, *narg):
                pred_y.append(1)
                pred_y_auc.append(1-best_dist_norm.pdf(er, loc=nloc, scale=nscale, *narg)/best_dist_abnorm.pdf(er, loc=abloc, scale=abscale, *abarg))
            else:
                pred_y.append(0)
                pred_y_auc.append(0)
    abnorm_num = len((np.asarray(label_lst)==1).nonzero()[0])
    abnorm_predict = np.asarray(pred_y)[(np.asarray(label_lst)==1).nonzero()[0]]
    abnorm_cor = (abnorm_predict == 1).sum()
    abnorm_pre = (np.asarray(pred_y)==1).sum()
    valid_recall = abnorm_cor / abnorm_num
    valid_precision = abnorm_cor / abnorm_pre
    print(f"epoch : {e}")
    print(f"KISTI recall : {valid_recall:.3f}")
    print(f"KISTI precision : {valid_precision:.3f}")
    if args.evaluation == 'threshold':
        auc = roc_auc_score(error_df.True_class, error_df.Reconstruction_error)
        ps, rs, _ = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    else:
        auc = roc_auc_score(error_df.True_class, pred_y_auc)
        ps, rs, _ = precision_recall_curve(error_df.True_class, pred_y_auc)
    f1 = f1_score(label_lst, pred_y)
    prauc = metrics.auc(rs, ps)
    print(f"KISTI AUROC :{auc:.4f} | f1 score :{f1:.4f} | PRAUC: {prauc:.4f}")
    conf_matrix = confusion_matrix(label_lst, pred_y)
    TN = conf_matrix[0][0]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]
    FP = conf_matrix[0][1]
    tpr = TP/(TP+FN)
    tnr = TN/(TN+FP)
    fpr = FP/(FP+TN)
    fnr = FN/(TP+FN)
    print(f"TPR:{tpr} | TNR:{tnr} | FPR:{fpr} | FNR:{fnr}")
    print(f"confusion matrix\n", conf_matrix)
    if isinstance(e, int):
        writer.add_scalar('test recall', valid_recall, e)
        writer.add_scalar('test precision', valid_precision, e)
        writer.add_scalar('test ROAUC', auc, e)
        writer.add_scalar('test PRAUC', prauc, e)
        writer.add_scalar('test f1', f1, e)
    else:
        result_dict = {'final recall': valid_recall,
                        'final precision': valid_precision,
                        'final AUROC': auc,
                        'final AUPRC': prauc,
                        'final f1': f1}
        if args.evaluation == 'threshold':
            fpr, tpr, _ = roc_curve(error_df.True_class, error_df.Reconstruction_error)
        else:
            fpr, tpr, _ = roc_curve(error_df.True_class, pred_y_auc)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.title('Receiver operating characteristic example')
        plt.savefig(f'{args.log_dir + args.timestamp}_{args.config}_transfer/{args.evaluation}_roc_curve.jpg')

        plt.figure()
        lw = 2
        plt.plot(rs, ps, color='darkorange', lw=lw, label='PR curve (area = %0.2f)' % prauc)
        plt.plot([0, 1], [test_abnorm_rate, test_abnorm_rate], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower right")
        plt.title('Receiver operating characteristic example')
        plt.savefig(f'{args.log_dir + args.timestamp}_{args.config}_transfer/{args.evaluation}_PR_curve.jpg')
        if args.only_infer:
            with open(args.log_dir + args.load_model + f'/best_model_{args.evaluation}_target_result.pkl', 'wb') as f:
                pickle.dump(result_dict, f) 
        else:
            with open(args.log_dir + args.timestamp + '_' + args.config  + f'_transfer/best_model_{args.evaluation}_target_result.pkl', 'wb') as f:
                pickle.dump(result_dict, f)

def Evaluation(args, mix_validloader, model, step, e, writer, eval_abnorm_rate, dists_info):
    '''Calculate AUC, f1 score'''
    if args.only_infer:
        save_dir = args.log_dir + args.load_model
    else:
        save_dir = args.log_dir + args.timestamp + '_' + args.config + '_transfer'
    label_lst = []
    recon_lst_val = []
    for _, data in enumerate(mix_validloader):
        if args.model == 'DNN':
            feature, label = data
            feature = feature.to(args.device)
            mu, logvar, recon,_ = model(feature)
            label = label.to(args.device)
            val_loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], \
                                            args.device, step, args.x0, args.KLD_W)
        else:
            encode_x, label, _ = data
            encode_x = encode_x.to(args.device)
            mu, logvar, recon, feature, lens,_ = model(encode_x)
            val_loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, args.device,\
                                                step, args.x0, args.KLD_W)
        if args.model == 'DNN':
            # recon_lst_val.extend(val_loss.cpu().detach().numpy())
            recon_lst_val.extend(bce.cpu().detach().numpy())
            if label.size()[0] == 1:
                label_lst.extend(label.cpu().detach().numpy()[0])
            else:
                label_lst.extend(label.squeeze().cpu().detach().numpy())
        else:
            bce_arr = bce.cpu().detach().numpy()
            bce_select = bce_arr.nonzero() # to flatten bce loss, discard masked value
            rec = bce_arr[bce_select] # flatten bce loss
            recon_lst_val.extend(rec) # flatten bce loss
            label_lst.extend([l for l in label])
    if args.model == 'RNN':
        flatten_l = [l for ls in label_lst for l in ls]
        label_lst = flatten_l
    
    if args.evaluation == 'threshold':
        asl = np.asarray(recon_lst_val)
        sort_idx = np.argsort(asl)
        lens = len(asl)
        abnorm_idx = sort_idx[-int(lens*eval_abnorm_rate):]
        pred_y = np.zeros_like(asl)
        pred_y[abnorm_idx] = 1
        error_df = pd.DataFrame({'Reconstruction_error': recon_lst_val, 'True_class': label_lst})
    else:
        error_df = pd.DataFrame({'Reconstruction_error': recon_lst_val, 'True_class':label_lst})
        label_lst = np.asarray(label_lst)    
        recon_lst = np.asarray(recon_lst_val)
        botnet_index = (label_lst == 1).nonzero()[0]
        print("the number of botnet_index in validation set :", len(botnet_index))
        mask = np.ones(len(recon_lst), dtype=bool)
        mask[botnet_index] = False
        if isinstance(e, int):
            if e % args.eval_frequency == 0:
                abnorm_dist_n, abparams = best_fit_distribution(recon_lst[botnet_index])
                abarg = abparams[:-2]
                abloc = abparams[-2]
                abscale = abparams[-1]
                best_dist_abnorm = getattr(st, abnorm_dist_n)
                norm_dist, params = best_fit_distribution(recon_lst[mask])
                best_dist_norm = getattr(st, norm_dist)
                params = best_dist_norm.fit(recon_lst[mask])
                narg = params[:-2]
                nloc = params[-2]
                nscale = params[-1]
            else:
                abnorm_dist_n, abparams, norm_dist, params = dists_info
                abarg = abparams[:-2]
                abloc = abparams[-2]
                abscale = abparams[-1]
                best_dist_abnorm = getattr(st, abnorm_dist_n)
                narg = params[:-2]
                nloc = params[-2]
                nscale = params[-1]
                best_dist_norm = getattr(st, norm_dist)
        else:
            abnorm_dist_n, abparams = best_fit_distribution(recon_lst[botnet_index])
            abarg = abparams[:-2]
            abloc = abparams[-2]
            abscale = abparams[-1]
            best_dist_abnorm = getattr(st, abnorm_dist_n)
            norm_dist, params = best_fit_distribution(recon_lst[mask])
            best_dist_norm = getattr(st, norm_dist)
            params = best_dist_norm.fit(recon_lst[mask])
            narg = params[:-2]
            nloc = params[-2]
            nscale = params[-1]
        pred_y = list()
        pred_y_auc = list()
        for er in error_df.Reconstruction_error.values:
            if best_dist_abnorm.pdf(er, loc=abloc, scale=abscale, *abarg) > best_dist_norm.pdf(er, loc=nloc, scale=nscale, *narg):
                pred_y.append(1)
                pred_y_auc.append(1-best_dist_norm.pdf(er, loc=nloc, scale=nscale, *narg)/best_dist_abnorm.pdf(er, loc=abloc, scale=abscale, *abarg))
            else:
                pred_y.append(0)
                pred_y_auc.append(0)

    abnorm_num = len((np.asarray(label_lst)==1).nonzero()[0])
    abnorm_predict = np.asarray(pred_y)[(np.asarray(label_lst)==1).nonzero()[0]]
    abnorm_cor = (abnorm_predict == 1).sum()
    abnorm_pre = (np.asarray(pred_y)==1).sum()
    valid_recall = abnorm_cor / abnorm_num
    valid_precision = abnorm_cor / abnorm_pre
    print(f"valid recall : {valid_recall:.3f}")
    print(f"valid precision : {valid_precision:.3f}")
    if args.evaluation == 'threshold':
        auc = roc_auc_score(error_df.True_class, error_df.Reconstruction_error)
        ps, rs, _ = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    else:
        auc = roc_auc_score(error_df.True_class, pred_y_auc)
        ps, rs, _ = precision_recall_curve(error_df.True_class, pred_y_auc)
    f1 = f1_score(error_df.True_class, pred_y)
    prauc = metrics.auc(rs, ps)
    if isinstance(e, int):
        writer.add_scalar('Source Valid recall', valid_recall, e)
        writer.add_scalar('Source Valid precision', valid_precision, e)
        writer.add_scalar('Source Valid ROAUC', auc, e)
        writer.add_scalar('Source Valid PRAUC', prauc, e)
        writer.add_scalar('Source Valid f1', f1, e)
    else:
        # if results:
        #     if 'normal' not in results.keys():
        #         results["normal"] = recon_lst[mask].tolist()
        #         results["abnorm"] = recon_lst[botnet_index].tolist()
        #     else:
        #         results["normal"].extend(recon_lst[mask].tolist())
        #         results["abnorm"].extend(recon_lst[botnet_index].tolist())
        if args.evaluation != 'threshold':
            botnet_values = recon_lst[botnet_index]
            normal_values = recon_lst[mask]
            with open(save_dir +'/transfer_abnorm_values.pkl', 'wb') as f:
                pickle.dump(botnet_values, f)
            with open(save_dir +'/transfer_normal_values.pkl', 'wb') as f:
                pickle.dump(normal_values, f)
        result_dict = {'final recall': valid_recall,
                        'final precision': valid_precision,
                        'final AUROC': auc,
                        'final AUPRC': prauc,
                        'final f1': f1}
        if args.only_infer:
            with open(args.log_dir + args.load_model + f'/best_model_{args.evaluation}_source_valid_result.pkl', 'wb') as f:
                pickle.dump(result_dict, f) 
        else:
            with open(save_dir + f'/best_model_{args.evaluation}_source_valid_result.pkl', 'wb') as f:
                pickle.dump(result_dict, f)
    print(f"valid AUC :{auc:.4f} | f1 score :{f1:.4f} | prauc: {prauc:.4f}")
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    if isinstance(e, int):
        print(f"epoch : {e+1} \n valid confusion matrix\n", conf_matrix)
    else:
        print(f"valid confusion matrix\n", conf_matrix)

    # for _, data in enumerate(testloader):
    #     if args.model == 'DNN':
    #         feature, label = data
    #         feature = feature.to(args.device)
    #         mu, logvar, recon, _ = model(feature)
    #         testloss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], \
    #                                     args.device, step, args.x0, args.KLD_W)
    #     else:
    #         encode_x, label = data
    #         encode_x = encode_x.to(args.device)
    #         mu, logvar, recon, feature, lens, _ = model(encode_x)
    #         testloss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, \
    #                                         args.device, step, args.x0, args.KLD_W)
    #     if args.model == 'DNN':
    #         # recon_lst_test.extend(testloss.cpu().detach().numpy())
    #         recon_lst_test.extend(bce.cpu().detach().numpy())
    #         test_label_lst.extend(label.squeeze().cpu().detach().numpy())
    #     else:
    #         bce_arr = bce.cpu().detach().numpy()
    #         bce_select = bce_arr.nonzero()
    #         rec = bce_arr[bce_select] # flatten bce loss
    #         recon_lst_test.extend(rec) # flatten bce loss
    #         test_label_lst.extend([l for l in label])
            
    # if args.model == 'RNN':
    #     flatten_l = [l for ls in test_label_lst for l in ls]
    #     test_label_lst = flatten_l
    # if args.evaluation == 'threshold':
    #     asl = np.asarray(recon_lst_test)
    #     sort_idx = np.argsort(asl)
    #     lens = len(asl)
    #     abnorm_idx = sort_idx[-int(lens*test_abnorm_rate):]
    #     pred_y = np.zeros_like(asl)
    #     pred_y[abnorm_idx] = 1
    #     error_df = pd.DataFrame({'Reconstruction_error': recon_lst_test, 'True_class': test_label_lst})
    # else:
    #     error_df = pd.DataFrame({'Reconstruction_error': recon_lst_test, 'True_class': test_label_lst})
    #     label_lst = np.asarray(test_label_lst)    
    #     recon_lst = np.asarray(recon_lst_test)
    #     botnet_index = (label_lst == 1).nonzero()[0]
    #     print("the number of botnet_index test set :", len(botnet_index))
    #     mask = np.ones(len(recon_lst), dtype=bool)
    #     mask[botnet_index] = False
    #     abnorm_dist_n, abparams = best_fit_distribution(recon_lst[botnet_index])
    #     abarg = abparams[:-2]
    #     abloc = abparams[-2]
    #     abscale = abparams[-1]
    #     best_dist_abnorm = getattr(st, abnorm_dist_n)
    #     norm_dist, params = best_fit_distribution(recon_lst[mask])
    #     best_dist_norm = getattr(st, norm_dist)
    #     params = best_dist_norm.fit(recon_lst[mask])
    #     narg = params[:-2]
    #     nloc = params[-2]
    #     nscale = params[-1]
    #     pred_y = list()
    #     pred_y_auc = list()
    #     for er in error_df.Reconstruction_error.values:
    #         if best_dist_abnorm.pdf(er, loc=abloc, scale=abscale, *abarg) > best_dist_norm.pdf(er, loc=nloc, scale=nscale, *narg):
    #             pred_y.append(1)
    #             pred_y_auc.append(1-best_dist_norm.pdf(er, loc=nloc, scale=nscale, *narg)/best_dist_abnorm.pdf(er, loc=abloc, scale=abscale, *abarg))
    #         else:
    #             pred_y.append(0)
    #             pred_y_auc.append(0)
    # abnorm_num = len((np.asarray(test_label_lst)==1).nonzero()[0])
    # abnorm_predict = np.asarray(pred_y)[(np.asarray(test_label_lst)==1).nonzero()[0]]
    # abnorm_cor = (abnorm_predict == 1).sum()
    # abnorm_pre = (np.asarray(pred_y)==1).sum()
    # test_recall = abnorm_cor / abnorm_num
    # test_precision = abnorm_cor / abnorm_pre
    # print(f"test recall : {test_recall:.3f}")
    # print(f"test precision : {test_precision:.3f}")
    # if args.evaluation == 'threshold':
    #     auc = roc_auc_score(error_df.True_class, error_df.Reconstruction_error)
    #     ps, rs, _ = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    # else:
    #     auc = roc_auc_score(error_df.True_class, pred_y_auc)
    #     ps, rs, _ = precision_recall_curve(error_df.True_class, pred_y_auc)
    # f1 = f1_score(error_df.True_class, pred_y)
    # prauc = metrics.auc(rs, ps)
    # if isinstance(e, int):
    #     writer.add_scalar('Source test recall', test_recall, e)
    #     writer.add_scalar('Source test precision', test_precision, e)
    #     writer.add_scalar('Source test ROAUC', auc, e)
    #     writer.add_scalar('Source test PRAUC', prauc, e)
    #     writer.add_scalar('Source test f1', f1, e)
    # else:
    #     result_dict = {'final recall': test_recall,
    #                     'final precision': test_precision,
    #                     'final AUROC': auc,
    #                     'final AUPRC': prauc,
    #                     'final f1': f1}
    #     if args.only_infer:
    #         with open(args.log_dir + args.load_model + f'/best_model_{args.evaluation}_source_test_result.pkl', 'wb') as f:
    #             pickle.dump(result_dict, f) 
    #     else:
    #         with open(args.log_dir + args.timestamp + '_' + args.config  + f'_transfer/best_model_{args.evaluation}_source_test_result.pkl', 'wb') as f:
    #             pickle.dump(result_dict, f)
    # print(f"test AUC :{auc:.4f} | f1 score :{f1:.4f} | PRAUC :{prauc:.4f}")
    # conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    # if isinstance(e, int):
    #     print(f"epoch : {e+1} \n test confusion matrix\n", conf_matrix)
    # else:
    #     print(f"test confusion matrix\n", conf_matrix)
    if args.evaluation == 'threshold':
        return auc
    else:
        return (abnorm_dist_n, abparams, norm_dist, params), auc

def train(args):
    print("device", args.device)
    sr_norm_trainloader, sr_abnorm_trainloader, sr_norm_validloader, sr_abnorm_validloader,\
                     mix_validloader, testloader, kisti_trainloader, kisti_validloader, target_df, inference_loader,\
                         eval_abnorm_rate, test_abnorm_rate, kisti_abnorm_rate = Load_Dataset(args)
    if args.model == "RNN":
        model = RNN_VAE(args.input_size, args.hidden_size, args.num_layer, args.batch_size, \
                        args.latent_size, args.bidirectional).to(args.device)
    elif args.model == 'DNN':
        model = DNN_VAE(args.dims, args.activation).to(args.device)
    if args.only_infer:
        print("only inference")
        model_dir = args.log_dir + args.load_model 
        writer = SummaryWriter(args.log_dir + args.load_model)
        model.load_state_dict(torch.load(model_dir + '/best_model.pt'))
        dists = None
        dists = Evaluation(args, mix_validloader, model, 0, 'final', writer, eval_abnorm_rate, dists)
        inference(args, model, inference_loader, target_df, writer, 'final', model_dir, kisti_abnorm_rate, dists)
    else:
        model_dir = args.log_dir + args.timestamp + '_' + args.config + '_transfer'
        writer = SummaryWriter(model_dir)
        ###
        # classifier = DANN_Classifier(args.input_size).to(args.device)
        # cl_optimizer = torch.optim.SGD([
        #                                 {'params': classifier.parameters()},
        #                                 {'params': model.parameters()}
        #                                 ], lr=args.lr)
        # bce_loss = nn.BCELoss()
        ###
        print("source normal train loader", len(sr_norm_trainloader))
        print("target train loader", len(kisti_trainloader))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        step = 0 
        update_num = 0
        valid_update_num = 0
        best_auc = 0
        for e in range(args.epochs):
            print(f"epoch : {e}")
            if e == 0:
                data_source_iter = iter(sr_norm_trainloader)
                data_target_iter = iter(kisti_trainloader)
            if len(sr_norm_trainloader) >= len(kisti_trainloader):
                len_dataloader = len(kisti_trainloader)
                data_target_iter = iter(kisti_trainloader)
            else:
                len_dataloader = len(sr_norm_trainloader)
                data_source_iter = iter(sr_norm_trainloader)
            auc_loss = 0 
            i = 0
            while i < len_dataloader:
                if len(sr_norm_trainloader) >= len(kisti_trainloader):
                    if e * len_dataloader + i >= len(sr_norm_trainloader) * (update_num + 1):
                        data_source_iter = iter(sr_norm_trainloader)
                        update_num += 1 
                        print("update source iteration")
                data = data_source_iter.next()
                model.train()
                model.zero_grad()
                if args.model == 'DNN':
                    feature, _ = data
                    feature = feature.to(args.device)
                    mu, logvar, recon, _ = model(feature)
                    loss, sr_bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], \
                                            args.device, step, args.x0, args.KLD_W)
                    ###
                    # result = classifier(recon)
                    # result = result.squeeze()
                    # target = torch.ones(feature.shape[0]).to(args.device)
                    # classifier_output = bce_loss(result, target)
                    ###
                else:
                    encode_x, _, _ = data
                    encode_x = encode_x.to(args.device)
                    mu, logvar, recon, feature, lens, _ = model(encode_x)
                    loss, sr_bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, \
                                                args.device, step, args.x0, args.KLD_W)
                    ###
                    # result = classifier(recon)
                    # result = result.squeeze()
                    # target = torch.ones(result.shape).to(args.device)
                    # classifier_output = bce_loss(result, target)
                    ###
                for _, d in enumerate(sr_abnorm_trainloader):
                    if args.model == "DNN":
                        f, _ = d
                    else:
                        f, _, _ = d
                    f = f.to(args.device)
                    if args.model == 'DNN':
                        mu_ab, logvar_ab, recon_ab, _ = model(f)
                        _, absr_bce, _ = vae_loss(recon_ab, f, mu_ab, logvar_ab, args.model, [],\
                                                    args.device, step, args.x0, args.KLD_W)
                        for l in sr_bce:
                            auc_loss += torch.sum(absr_bce - l.view(1, -1)) # absr [batch] / l : [1]
                    else:
                        mu_ab, logvar_ab, recon_ab, feature_ab, lens_ab, _ = model(f)
                        _, absr_bce, _ = vae_loss(recon_ab, feature_ab, mu_ab, logvar_ab, args.model, lens_ab, \
                                                args.device, step, args.x0, args.KLD_W)
                        for l in sr_bce:
                            auc_loss += torch.sum(absr_bce.mean(1) - l.mean())
                auc_loss = 1 / args.batch_size / (len(sr_abnorm_trainloader) * args.batch_size) * auc_loss
                if args.model == 'RNN':
                    if auc_loss > 0:
                        lamb = 1
                    else:
                        lamb = args.lamb
                else:
                    lamb = args.lamb
                sr_recon_loss = sr_bce.mean() - lamb * auc_loss
                source_loss = sr_recon_loss + 100 * kld.mean()
                source_loss.backward(retain_graph=True)
                ###
                # classifier_output.backward(retain_graph=True)
                # cl_optimizer.step()
                ###
                optimizer.step()
                writer.add_scalar('Batch Loss_source', source_loss.item(), i+e*len_dataloader)
                if args.model == 'DNN':
                    writer.add_scalar('Recon Loss_source', sr_bce.mean().item(), i+e*len_dataloader)
                else:
                    writer.add_scalar('Recon Loss_source', sr_bce.mean().item(), i+e*len_dataloader)
                writer.add_scalar('KLD', kld.mean().item(), i+e*len_dataloader)
                if len(sr_norm_trainloader) < len(kisti_trainloader):
                    if e * len_dataloader + i >= len(kisti_trainloader) * (update_num + 1):
                        data_target_iter = iter(kisti_trainloader)
                        update_num += 1
                        print("update target iteration")
                target_data = data_target_iter.next()
                
                if args.model == 'DNN':
                    feature, _ = target_data
                else:feature, _, _ = target_data
                feature = feature.to(args.device)
                model.zero_grad()
                if args.normal_select and e > 48:
                    if i == 0 :
                        pass
                    else:
                        if args.model == 'DNN':
                            num = feature.size()[0]
                            idx = np.random.choice(num, int(num/2), replace=False)
                            indices = torch.tensor(idx).to(args.device)
                            selected = torch.index_select(feature, 0, indices)
                            feature = torch.cat((selected,next_data), 0)
                        else:
                            # padded, lens = pad_packed_sequence(feature, batch_first=True)
                            selected, lens = pad_packed_sequence(feature, batch_first=True)
                            lens = lens.to(args.device)
                            # padded = padded.to(args.device)
                            selected = selected.to(args.device)
                            # num = padded.size()[0]
                            # idx = np.random.choice(num, int(num/3), replace=False)
                            # indices = torch.tensor(idx).to(args.device)
                            # selected = torch.index_select(padded, 0, indices)
                            if selected.size()[1] != next_data.size()[1]:
                                if selected.size()[1] > next_data.size()[1]:
                                    pad_n = selected.size()[1] - next_data.size()[1]
                                    m = nn.ConstantPad2d((0, 0, 0, pad_n), 0.0)
                                    next_data = m(next_data)
                                else:
                                    pad_n = next_data.size()[1] - selected.size()[1]
                                    m = nn.ConstantPad2d((0, 0, 0, pad_n), 0.0)
                                    selected = m(selected)
                            # selected_lens = torch.index_select(lens, 0, indices)
                            total_input = torch.cat((selected, next_data),0)
                            # total_lens = torch.cat((selected_lens, next_lens),0)
                            total_lens = torch.cat((lens, next_lens),0)
                            sorted_lens, sorted_lens_idx = torch.sort(total_lens, descending=True)
                            feature = total_input[sorted_lens_idx]
                            feature = pack_padded_sequence(feature, sorted_lens, batch_first=True)
                if args.model == 'DNN':
                    mu, logvar, recon, z_d = model(feature)
                    loss, target_bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], \
                                            args.device, step, args.x0, args.KLD_W)
                    ###
                    # result = classifier(recon)
                    # result = result.squeeze()
                    # target = torch.zeros(feature.shape[0]).to(args.device)
                    # classifier_output_target = bce_loss(result, target)
                    ###
                else:
                    pads, _ = pad_packed_sequence(feature, batch_first=True)
                    mu, logvar, recon, feature, lens, z_d = model(feature)
                    loss, target_bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, \
                                                args.device, step, args.x0, args.KLD_W)
                    ###
                    # result = classifier(recon)
                    # result = result.squeeze()
                    # target = torch.zeros(result.shape).to(args.device)
                    # classifier_output_target = bce_loss(result, target)
                    ###
                target_loss = loss.mean()
                target_loss.backward()
                optimizer.step()
                if args.normal_select and e > 48:
                    print("normal select")
                    if args.model == 'DNN':
                        norm_ix = torch.argsort(target_bce)
                    else:
                        norm_ix = torch.argsort(target_bce.mean(1))
                    norm_ix_select = norm_ix[:int(args.batch_size * args.normal_select_rate)]
                    if args.model == 'DNN':
                        next_data = feature[norm_ix_select]
                    else: 
                        next_data = pads[norm_ix_select].to(args.device)
                        next_lens = lens[norm_ix_select].to(args.device)
                ###
                # target_loss.backward(retain_graph=True)
                # classifier_output_target.backward(retain_graph=True)
                # cl_optimizer.step()
                ###

                # log
                if (e * len_dataloader + i) % args.log_frequency== 0:
                    print(f'progress : {i/len_dataloader*100:.0f}%, batch source total loss : {source_loss.item():.3f}, batch source recon loss : {sr_bce.mean().item():.3f}, {sr_recon_loss.item():.3f}')
                    print(f'auc loss:{auc_loss.item():.3f}')
                    print(f'batch target recon loss : {target_bce.mean().item():.3f}, batch target total loss : {target_loss.item():.3f}')
                    
                writer.add_scalar('Batch Loss_target', target_loss.item(), i+e*len_dataloader)
                if args.model == 'DNN':
                    writer.add_scalar('Recon Loss_target', sr_bce.mean().item(), i+e*len_dataloader)
                else:
                    writer.add_scalar('Recon Loss_target', sr_bce.mean().item(), i+e*len_dataloader)
                writer.add_scalar('KLD_target', kld.mean().item(), i+e*len_dataloader)
                if args.normal_select:
                    if e < 48:
                        if z_d.shape[0] == (args.batch_size-int(args.batch_size * args.normal_select_rate)):
                            torch.save(z_d, f"{model_dir}/z_{args.device}.pt")
                            z_saved = z_d
                            print("saved z 1")
                    else:
                        if z_d.shape[0] == (args.batch_size):
                            torch.save(z_d, f"{model_dir}/z_{args.device}.pt")
                            z_saved = z_d
                            print("saved z 1")
                else:
                    if z_d.shape[0] == (args.batch_size):
                        torch.save(z_d, f"{model_dir}/z_{args.device}.pt")
                        z_saved = z_d
                        print("saved z 1")


                step += 1
                i += 1

            # print("validation start")
            # w = 0
            # val_auc_loss = 0 
            # if e == 0:
            #     norm_loader = iter(sr_norm_validloader)
            #     valid_target_loader = iter(kisti_validloader)
            # if len(sr_norm_validloader) >= len(kisti_validloader):
            #     len_loader = len(kisti_validloader)
            #     valid_target_loader = iter(kisti_validloader)
            # else:
            #     len_loader = len(sr_norm_validloader)
            #     norm_loader = iter(sr_norm_validloader)
            # model.eval()
            # while w < len_loader:
            #     if len(sr_norm_validloader) >= len(kisti_validloader):
            #         if e * len_loader + w >= len(sr_norm_validloader) * (valid_update_num + 1):
            #             norm_loader = iter(sr_norm_validloader)
            #             valid_update_num += 1 
            #             print("update source iteration")
            #     data = norm_loader.next()
            #     model.zero_grad()
            #     if args.model == 'DNN':
            #         feature, _ = data
            #         feature = feature.to(args.device)
            #         mu, logvar, recon,_ = model(feature)
            #         val_loss, valid_bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], args.device, step, args.x0, args.KLD_W)
            #     else:
            #         encode_x, _, _ = data
            #         encode_x = encode_x.to(args.device)
            #         mu, logvar, recon, feature, lens, _ = model(encode_x)
            #         val_loss, valid_bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, args.device, step, args.x0, args.KLD_W)
            #     for _, d in enumerate(sr_abnorm_validloader):
            #         if args.model == "DNN":
            #             f, _ = d
            #         else:
            #             f, _, _ = d
            #         f = f.to(args.device)
            #         if args.model == 'DNN':
            #             mu_ab, logvar_ab, recon_ab, _ = model(f)
            #             _, absr_bce, _ = vae_loss(recon_ab, f, mu_ab, logvar_ab, args.model, [],\
            #                                         args.device, step, args.x0, args.KLD_W)
            #             for l in valid_bce:
            #                 val_auc_loss += torch.sum(absr_bce - l.view(1, -1))
            #         else:
            #             mu_ab, logvar_ab, recon_ab, feature_ab, lens_ab, _ = model(f)
            #             _, absr_bce, _ = vae_loss(recon_ab, feature_ab, mu_ab, logvar_ab, args.model, lens_ab, \
            #                                     args.device, step, args.x0, args.KLD_W)
            #             for l in valid_bce:
            #                 val_auc_loss += torch.sum(absr_bce.mean(1) - l.mean()) # calculate the average loss of total sequence, l : [seq] / absr_bce : [batch, seq]
            #     val_auc_loss = 1 / args.batch_size / (len(sr_abnorm_validloader) * args.batch_size) * val_auc_loss
            #     sr_recon_loss = valid_bce.mean() - args.lamb * val_auc_loss
            #     val_loss = sr_recon_loss + kld.mean()
                
            #     writer.add_scalar('Valid Batch Loss_source', val_loss.item(), w+e*len_loader)
            #     if args.model == 'DNN':
            #         writer.add_scalar('Valid Recon Loss_source', valid_bce.mean().item(), w+e*len_loader)
            #     else:
            #         writer.add_scalar('Valid Recon Loss_source', valid_bce.mean(1).mean().item(), w+e*len_loader)
            #     writer.add_scalar('Valid KLD_source', kld.mean().item(), w+e*len_loader)

            #     if len(sr_norm_validloader) <= len(kisti_validloader):
            #         if e * len_loader + w >= len(kisti_validloader) * (valid_update_num + 1):
            #             valid_target_loader = iter(kisti_validloader)
            #             valid_update_num += 1
            #             print("update target iteration")
            #     target_data = valid_target_loader.next()
            #     if args.model == 'DNN':
            #         feature, _ = target_data
            #         feature = feature.to(args.device)
            #         mu, logvar, recon, _ = model(feature)
            #         loss, sr_bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], \
            #                                 args.device, step, args.x0, args.KLD_W)
            #     else:
            #         encode_x, _, _ = target_data
            #         encode_x = encode_x.to(args.device)
            #         mu, logvar, recon, feature, lens, _ = model(encode_x)
            #         loss, sr_bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, \
            #                                     args.device, step, args.x0, args.KLD_W)
            #     loss = loss.mean()
                
            #     if w % args.log_frequency== 0:
            #         print(f'progress : {w/len_loader*100:.0f}%, valid source total loss : {val_loss.item():.3f}')
            #         print(f'valid source recon loss : {valid_bce.mean().item():.3f}') 
            #         print(f'valid target recon loss : {sr_bce.mean().item():.3f}, valid target loss : {loss.item():.3f}')
                
            #     writer.add_scalar('Valid batch Loss_target', loss.item(), w+e*len_loader)
            #     if args.model == 'DNN':
            #         writer.add_scalar('Valid Recon Loss_target', sr_bce.mean().item(), w+e*len_loader)
            #     else:
            #         writer.add_scalar('Valid Recon Loss_target', sr_bce.mean().item(), w+e*len_loader)
            #     writer.add_scalar('Valid KLD_target', kld.mean().item(), w+e*len_loader)
            #     w += 1
            
            if e % args.eval_frequency == 0:
                model.eval()
                print("evaluation start!")
                if e == 0:
                    dists = None
                if args.evaluation == 'threshold':
                    # results = Evaluation(args, mix_validloader, model, 10000, e, writer, eval_abnorm_rate, None)
                    results = Evaluation(args, kisti_validloader, model, 10000, e, writer, eval_abnorm_rate, None)
                else:
                    # dists, results = Evaluation(args, mix_validloader, model, 10000, e, writer, eval_abnorm_rate, dists)
                    dists, results = Evaluation(args, kisti_validloader, model, 10000, e, writer, eval_abnorm_rate, dists)
                if results > best_auc:
                    torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '_transfer/'  + 'best_model.pt')
                    best_auc = results
                    torch.save(z_saved, f"{model_dir}/best_z_{args.device}.pt")
                    print("saved z")
                    print("saved the best model!")
                print("inference start!")
                if e != 0:
                    inference(args, model, inference_loader, target_df, writer, e, model_dir, kisti_abnorm_rate, dists)

        ## final inference
        model.load_state_dict(torch.load(args.log_dir + args.timestamp + '_' + args.config + '_transfer/'  + 'best_model.pt'))
        print("The best model loaded")
        if args.evaluation == 'threshold':
            # results = Evaluation(args, mix_validloader, model, 10000, 'final', writer, eval_abnorm_rate,  dists)
            results = Evaluation(args, kisti_validloader, model, 10000, 'final', writer, eval_abnorm_rate,  dists)
        else:
            dists = None
            # dists, results = Evaluation(args, mix_validloader, model, 10000, 'final', writer, eval_abnorm_rate, dists)
            dists, results = Evaluation(args, kisti_validloader, model, 10000, 'final', writer, eval_abnorm_rate, dists)
        if args.evaluation == 'threshold':
            inference(args, model, inference_loader, target_df, writer, 'final', model_dir, kisti_abnorm_rate, None)
        else:
            inference(args, model, inference_loader, target_df, writer, 'final', model_dir, kisti_abnorm_rate, dists)
    
if __name__ == "__main__":
    train(get_config())