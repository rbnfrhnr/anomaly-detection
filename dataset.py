from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import torch
import pandas as pd

class NetworkDataset(Dataset):
    def __init__(self, seq_len, df, train, model):
        self.seq_len = seq_len 
        self.train = train
        self.model = model 
        self.df = df
        self.make_dataset_from_df()
        self.X, self.y = self.split_valid_test()
        
    def make_dataset_from_df(self):
        input_X = self.df.filter(['Proto_icmp', 'Proto_rtcp', 'Proto_rtp', 'Proto_tcp', 'Proto_udp',
                            'Proto_udt', 'service_', 'service_dns', 'service_http', 'service_ntp',
                            'TotBytes', 'SrcBytes', 'TotPkts', 'Dur', 'sTos', 'dTos', 'min', 'hour',
                            'sum_totalbytes', 'sum_cnt'], axis=1).values
        input_y = self.df["label_num"].values
        if self.model =="RNN":
            self.X = []
            self.y = []
            for i in range(len(input_X) - self.seq_len -1):
                self.X.append(input_X[i:i+self.seq_len])
                self.y.append([input_y[i+self.seq_len+1]])
        elif self.model =="DNN":
            self.X = input_X[:, np.newaxis]
            self.y = input_y[:, np.newaxis]
            
    def split_valid_test(self):
        trainx, testx, trainy, testy = train_test_split(np.array(self.X), np.array(self.y), test_size=0.2, random_state=1234)
        trainx, validx, trainy, validy = train_test_split(np.array(trainx), np.array(trainy), test_size=0.2, random_state=1234) 
        if self.train == 'train':
            return trainx, trainy
        elif self.train == 'valid':
            return validx, validy
        else:
            return testx, testy

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.X[idx]), torch.LongTensor(self.y[idx]))
            
    def __len__(self):
        return len(self.X)
    
class NetworkDataset_ae(NetworkDataset):
    def __init__(self, df, train, model, same_column, preprocess, total_seq, overlap, n_split, dataset):
        self.train = train
        self.df = df
        self.preprocess = preprocess
        self.model = model
        # self.scenario = scenario
        self.dataset = dataset
        if model == 'RNN' and self.dataset == 'CTU' and train == 'train':
            self.cv_split(n_split)
        self.column = same_column
        self.overlap = overlap
        self.total_seq = total_seq
        self.make_dataset_from_df()
        if not self.dataset == 'CTU':
            if not train == 'test':
                self.X, self.y = self.split_valid_test()
            else:
                self.X, self.y = np.asarray(self.X), np.asarray(self.y)
        self.X, self.y = np.asarray(self.X), np.asarray(self.y)
        
    def cv_split(self, n_split):
        kf = KFold(n_splits=n_split)
        self.kf_train_indices = dict()
        self.kf_valid_indices = dict()
        for cv_idx, (train_index, valid_index) in enumerate(kf.split(self.df.values)):
            normal_idx = (self.df.iloc[train_index]["label_num"].values == 0).nonzero()[0] # only normal data
            new_train_index = train_index[normal_idx]
            self.kf_train_indices[cv_idx] = new_train_index
            self.kf_valid_indices[cv_idx] = valid_index

    def make_dataset_from_df(self):
        # if self.train == 'test' and not self.scenario == 'all':
            # self.df = self.df[self.df["scenario"].isin(list(self.scenario))]
        if self.model == 'DNN':
            input_X = self.df.filter(self.column, axis=1)
            input_X = input_X.drop(["label_num", "class", "id.orig_h", "time_chunk"], axis=1)
            input_y = self.df["label_num"]
            input_X = input_X.values
            input_y = input_y.values
            self.X = input_X[:]
            self.y = input_y[:, np.newaxis]
        else:
            if self.dataset == 'CTU' and self.train =='train':
                self.X = []
                self.y = []
                self.cv_idx_train = dict()
                self.cv_idx_valid = dict()
                for i in range(5):
                    input_X = self.df.iloc[self.kf_train_indices[i]].filter(self.column, axis=1)
                    input_X = input_X.drop(["label_num",  "class", "id.orig_h"], axis=1)
                    input_y = self.df.iloc[self.kf_train_indices[i]]["label_num"]
                    if self.overlap:
                        self.X, self.y = self.RNN_step(1, input_X, input_y, self.X, self.y)
                    else:
                        self.X, self.y = self.RNN_step_2(3, input_X, input_y, self.X, self.y)
                    self.cv_idx_train[i] = len(self.X)
                    input_X = self.df.iloc[self.kf_valid_indices[i]].filter(self.column, axis=1)
                    input_X = input_X.drop(["label_num",  "class", "id.orig_h"], axis=1)
                    input_y = self.df.iloc[self.kf_valid_indices[i]]["label_num"]
                    if self.overlap:
                        self.X, self.y = self.RNN_step(1, input_X, input_y, self.X, self.y)
                    else:
                        self.X, self.y = self.RNN_step_2(3, input_X, input_y, self.X, self.y)
                    self.cv_idx_valid[i] = len(self.X)
            else:
                self.X = []
                self.y = []
                input_X = self.df.filter(self.column, axis=1)
                input_X = input_X.drop(["label_num",  "class", "id.orig_h"], axis=1)
                input_y = self.df["label_num"]
                if self.overlap:
                    self.X, self.y = self.RNN_step(1, input_X, input_y, self.X, self.y)
                else:
                    self.X, self.y = self.RNN_step_2(3, input_X, input_y, self.X, self.y)
    
    def RNN_step(self, step_size, input_X, input_y, X, y):
        time_intervals = input_X["time_chunk"].unique()
        input_X_fil = input_X.drop(["time_chunk"], axis=1)
        prev_chunk = dict()
        prev_chunk_y = dict()
        for i in range(1, self.total_seq+1):
            prev_chunk[i] = None
            prev_chunk_y[i] = None
        for t, inter in enumerate(time_intervals):
            if t < self.total_seq - 1:
                prev_chunk[t+1] = input_X_fil[input_X["time_chunk"]==inter].values 
                prev_chunk_y[t+1] = input_y[input_X["time_chunk"]==inter].values
                continue
            prev_chunk[self.total_seq] = input_X_fil[input_X["time_chunk"]==inter].values
            prev_chunk_y[self.total_seq] = input_y[input_X["time_chunk"]==inter].values
            prev_lsts = list()
            prev_y_lsts = list()
            for w in range(1, self.total_seq+1):
                prev_lsts.extend(prev_chunk[w].tolist())
                prev_y_lsts.extend(prev_chunk_y[w].tolist())
            X.append(prev_lsts)
            y.append(prev_y_lsts)
            for k in range(1, self.total_seq):
                prev_chunk[k] = prev_chunk[k+1]
                prev_chunk_y[k] = prev_chunk_y[k+1]
            if t == len(time_intervals) - self.total_seq:
                break
        return X, y
            # prev_chunk[self.total_seq] = present
            # prev_chunk_y[self.total_seq] = present_y
        
        # prev_2 = None
        # prev_1 = None
        # for t, inter in enumerate(time_intervals):
        #     if t < 2:
        #         continue
        #     if (prev_2 is None) or (prev_1 is None):
        #         prev_2 = input_X_fil[input_X["time_chunk"]==(inter-2)].values
        #         prev_2_y = input_y[input_X["time_chunk"]==(inter-2)].values
        #         prev_1 = input_X_fil[input_X["time_chunk"]==(inter-1)].values
        #         prev_1_y = input_y[input_X["time_chunk"]==(inter-1)].values
        #     present = input_X_fil[input_X["time_chunk"]==inter].values
        #     present_y = input_y[input_X["time_chunk"]==inter].values
        #     self.X.append(prev_2.tolist()+prev_1.tolist()+present.tolist())
        #     self.y.append(prev_2_y.tolist()+prev_1_y.tolist()+present_y.tolist())
        #     prev_2 = prev_1
        #     prev_1 = present
        #     prev_2_y = prev_1_y
        #     prev_1_y = present_y

    def RNN_step_2(self, step_size, input_X, input_y, X, y):
        time_intervals = input_X["time_chunk"].unique()
        input_X_fil = input_X.drop(["time_chunk"], axis=1)
        # for i in range(1, self.total_seq+1):
        #     prev_chunk[i] = None
        #     prev_chunk_y[i] = None

        for t, inter in enumerate(time_intervals):
            if t % step_size == 0:
                X_lst = list()
                y_lst = list()
            X_lst.extend(input_X_fil[input_X["time_chunk"]==inter].values.tolist())
            y_lst.extend(input_y[input_X["time_chunk"]==inter].values.tolist())
            if t % step_size == 2:
                X.append(X_lst)
                y.append(y_lst)
            elif t == len(time_intervals) - 1:
                X.append(X_lst)
                y.append(y_lst)
        return X, y

    def split_valid_test(self):
        # trainx, validx, trainy, validy = train_test_split(np.array(self.X), np.array(self.y), test_size=0.2, random_state=1234)
        trainx, validx, trainy, validy = train_test_split(np.array(self.X), np.array(self.y), test_size=0.2, random_state=0)
        if self.train == 'train':
            return trainx, trainy
        else:
            return validx, validy
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), self.y[idx]
            
    def __len__(self):
        return len(self.X)
    
if __name__ == "__main__":
    label_dict = {"norm":0, "abnorm":1}
    target_df = pd.read_csv('./KISTI/pre_vae_RNN_small_False_30.csv',index_col=[0])
    target_df["label_num"] = target_df["class"].apply(lambda x : label_dict[x])
    same_column = list(target_df.columns)
    traindataset = NetworkDataset_ae(target_df, "train",  'RNN', same_column, 'small', 3)
    print(len(traindataset))
    
    