from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import random
import math
import torch
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, f1_score
from sklearn import metrics
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence
from model import *
from preprocessing import *
from dataset import *
from config import *
from loss import *
from dataloader import *
import scipy
from scipy.stats import norm
from sklearn.model_selection import train_test_split

def Load_Dataset(args):
    # # #source domain
    trainset_name = ''.join(args.trainset)
    testset_name = ''.join(args.testset)
    '../ctu-13/CTU-13-Dataset/pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv'
    if f"pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv" in os.listdir('../ctu-13/CTU-13-Dataset'):
        dis_df = pd.read_csv(f'../ctu-13/CTU-13-Dataset/pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv',\
                            index_col=[0])
    else:  
        df = make_conn_dataframe(args.trainset, args.scenario)
        dis_df = preprocess_stat_2_CTU(df, args.period_len_ctu, args.rm_ntp)
        dis_df.to_csv(f'../ctu-13/CTU-13-Dataset/pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv')

    if args.all_scenario:
        total_df = dis_df
        dis_df, test_df = train_test_split(total_df, test_size=0.3, random_state=1234)

    label_dict = {"normal":0, "botnet":1}
    dis_df["label_num"] = dis_df["class"].apply(lambda x : label_dict[x])
    print("total length", len(dis_df))
    print("Source Domain train/valid set :", Counter(dis_df["class"]))
    eval_abnorm_rate = Counter(dis_df["class"])["botnet"] / len(dis_df)


    same_column = list(dis_df.columns)
    traindataset_cv = NetworkDataset_ae(dis_df, "train",  args.model, same_column, args.preprocess, args.total_seq, args.overlap, 5, args.dataset) # cross validation dataset

    input_X = dis_df.filter(same_column, axis=1)
    # print("column", same_column)
    input_X = input_X.drop(["label_num", "class", "id.orig_h", "time_chunk"], axis=1)
    input_y = dis_df["label_num"]
    input_X = input_X.values
    input_y = input_y.values[: np.newaxis]

    trainx, validx, trainy, validy = train_test_split(np.array(input_X), np.array(input_y), test_size=0.2, random_state=1234)
    if not args.all_scenario:
        if f"pre_vae_{testset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}_{args.scenario}.csv" in os.listdir('../ctu-13/CTU-13-Dataset'):
            test_df = pd.read_csv(f'../ctu-13/CTU-13-Dataset/pre_vae_{testset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}_{args.scenario}.csv', \
                                index_col=[0])
            print("load test dataframe")
        else:
            test_df = make_conn_dataframe(args.testset, args.scenario)
            test_df = preprocess_stat_2_CTU(test_df, args.period_len_ctu, args.rm_ntp)
            test_df.to_csv(f'../ctu-13/CTU-13-Dataset/pre_vae_{testset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}_{args.scenario}.csv')
            print("saved test preprocessed file") 

    print("test set :", Counter(test_df["class"]))
    label_dict = {"normal":0, "botnet":1}
    test_df["label_num"] = test_df["class"].apply(lambda x : label_dict[x])
    
    input_X = test_df.filter(same_column, axis=1)
    # print("column", same_column)
    input_X = input_X.drop(["label_num", "class", "id.orig_h", "time_chunk"], axis=1)
    input_y = test_df["label_num"]
    input_X = input_X.values
    input_y = input_y.values[: np.newaxis]

    testx = input_X
    testy = input_y

    #target domain
    # if args.sampling_data:
    #     if args.only_shared:
    #         if f"pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}_shared.csv" in os.listdir('../ctu-13/KISTI'):
    #             target_df = pd.read_csv(f'../ctu-13/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}_shared.csv',index_col=[0])
    #         else:
    #             df = pd.read_csv('data/KISTI/kisti_logs_20190714_6_shared.csv')
    #             target_df = preprocess_stat_KISTI(df, args.period_len_kisti, args.rm_ntp)
    #             target_df.to_csv(f'../ctu-13/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}_shared.csv')
    #     else:
    #         if f"pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv" in os.listdir('../ctu-13/KISTI'):
    #             target_df = pd.read_csv(f'../ctu-13/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv',index_col=[0])
    #         else:
    #             df = pd.read_csv('data/KISTI/kisti_logs_20190714_6.csv')
    #             target_df = preprocess_stat_KISTI(df, args.period_len_kisti, args.rm_ntp)
    #             target_df.to_csv(f'../ctu-13/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv')
     #if f"pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv" in os.listdir('../ctu-13/KISTI'):
     #    target_df = pd.read_csv(f'../ctu-13/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv',index_col=[0])
     #else:
     #    if args.sampling_data:
     #        df = pd.read_csv('data/KISTI/kisti_logs_20190714_6.csv')
     #    else:
     #        df = pd.read_csv('data/KISTI/kisti_logs.csv')
     #    target_df = preprocess_stat_KISTI(df, args.period_len_kisti, args.rm_ntp)
     #    target_df.to_csv(f'../ctu-13/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv')
     #    print("saved target preprocessed file")
    
    # label_dict = {"norm":0, "abnorm":1}
    # target_df["label_num"] = target_df["class"].apply(lambda x : label_dict[x])
    # print("kisti dataset :", Counter(target_df["class"]))
    # print("kisti length of df", len(target_df))
    # same_column = list(target_df.columns)
    # input_X = target_df.filter(same_column, axis=1)
    # input_X = input_X.drop(["label_num", "class", "id.orig_h", "time_chunk"], axis=1)
    # input_y = target_df["label_num"]
    # input_X = input_X.values
    # input_y = input_y.values[: np.newaxis]

    # trainx, validx, trainy, validy = train_test_split(np.array(input_X), np.array(input_y), test_size=0.2, random_state=1234)
    # print("# of train", len(trainx))
    # print("# of valid", len(validx))
    # return trainx, validx, trainy, validy
    # return trainx, input_X, trainy, input_y
    return trainx, validx, trainy, validy, testx, testy

if __name__ == "__main__":
    args = get_config()
    # trainx, validx, trainy, validy = Load_Dataset(args)
    trainx, validx, trainy, validy, testx, testy = Load_Dataset(args)
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    clf.fit(trainx, trainy)

    predicted_y = clf.predict(validx)
    abnorm_num = len((validy==1).nonzero()[0])
    abnorm_predict = predicted_y[(validy == 1).nonzero()[0]]
    abnorm_cor = (abnorm_predict == 1).sum()
    abnorm_pre = (np.asarray(predicted_y)==1).sum()
    valid_recall = abnorm_cor / abnorm_num
    valid_precision = abnorm_cor / abnorm_pre

    pred_y_prob = clf.predict_proba(validx)
    print(f"KISTI recall : {valid_recall:.3f}")
    print(f"KISTI precision : {valid_precision:.3f}")

    auc = roc_auc_score(validy, pred_y_prob[:,1])
    f1 = f1_score(validy, predicted_y)
    ps, rs, _ = precision_recall_curve(validy, pred_y_prob[:,1])
    prauc = metrics.auc(rs, ps)
    print(f"KISTI AUROC :{auc:.4f} | f1 score :{f1:.4f} | PRAUC: {prauc:.4f}")
    conf_matrix = confusion_matrix(validy, predicted_y)
    print(f"confusion matrix\n", conf_matrix)

    
    predicted_y = clf.predict(testx)
    abnorm_num = len((testy==1).nonzero()[0])
    abnorm_predict = predicted_y[(testy == 1).nonzero()[0]]
    abnorm_cor = (abnorm_predict == 1).sum()
    abnorm_pre = (np.asarray(predicted_y)==1).sum()
    valid_recall = abnorm_cor / abnorm_num
    valid_precision = abnorm_cor / abnorm_pre

    pred_y_prob = clf.predict_proba(testx)
    print(f"KISTI recall : {valid_recall:.3f}")
    print(f"KISTI precision : {valid_precision:.3f}")

    auc = roc_auc_score(testy, pred_y_prob[:,1])
    f1 = f1_score(testy, predicted_y)
    ps, rs, _ = precision_recall_curve(testy, pred_y_prob[:,1])
    prauc = metrics.auc(rs, ps)
    print(f"KISTI AUROC :{auc:.4f} | f1 score :{f1:.4f} | PRAUC: {prauc:.4f}")
    conf_matrix = confusion_matrix(testy, predicted_y)
    print(f"confusion matrix\n", conf_matrix)

    print("feature importance", clf.feature_importances_)
    

#    predicted_y = clf.predict(testx)
#    abnorm_num = len((testy==1).nonzero()[0])
#    abnorm_predict = predicted_y[(testy == 1).nonzero()[0]]
#    abnorm_cor = (abnorm_predict == 1).sum()
#    abnorm_pre = (np.asarray(predicted_y)==1).sum()
#    valid_recall = abnorm_cor / abnorm_num
#    valid_precision = abnorm_cor / abnorm_pre
#
#    pred_y_prob = clf.predict_proba(testx)
#    print(f"KISTI test recall : {valid_recall:.3f}")
#    print(f"KISTI test precision : {valid_precision:.3f}")
#
#    # print(validy.shape)
#    # print(pred_y_prob.shape)
#    auc = roc_auc_score(testy, pred_y_prob[:,1])
#    f1 = f1_score(testy, predicted_y)
#    ps, rs, _ = precision_recall_curve(testy, pred_y_prob[:,1])
#    prauc = metrics.auc(rs, ps)
#    print(f"KISTI test AUROC :{auc:.4f} | f1 score :{f1:.4f} | PRAUC: {prauc:.4f}")
#    conf_matrix = confusion_matrix(testy, predicted_y)
#    print(f"test confusion matrix\n", conf_matrix)
