import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import random
import math
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_recall_curve
from model import *
from preprocessing import *
from dataset import *
from config import *

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def validation(model, device, seq_len, n_hidden,loader):
    print("validation start")
    botnet_num = 0
    botnet_cor = 0 
    botnet_pre = 0 
    model.eval()
    predicted_lst = list()
    label_tensor_lst = list()
    correct = 0
    total = 0 
    for _, (feature_tensor, label_tensor) in enumerate(loader):
        feature_tensor = feature_tensor.to(device)
        label_tensor = label_tensor.to(device)
        label_tensor_lst.extend(label_tensor.squeeze().cpu().detach().numpy())
        output = model(feature_tensor)
        _, top_i = output.topk(1)
        correct += (top_i == label_tensor).sum().item()
        total += label_tensor.size(0)
        predicted_lst.extend(top_i.squeeze().cpu().detach().numpy())
        if len((label_tensor.squeeze().cpu().detach().numpy()==2).nonzero()[0]):
            botnet_num += len((label_tensor.squeeze().cpu().detach().numpy()==2).nonzero()[0])
            botnet_predict = top_i.cpu().detach().numpy()[(label_tensor.squeeze().cpu().detach().numpy()==2).nonzero()[0]]
            botnet_cor += (botnet_predict == 2).sum()
        if (top_i == 2).sum().item() > 0:
            botnet_pre += (top_i == 2).sum().item()
    c = confusion_matrix(label_tensor_lst, predicted_lst, labels=[0,1,2])
    acc = correct / total * 100
    return botnet_cor, botnet_num, botnet_pre, c, acc

def train(args):
    writer = SummaryWriter(args.log_dir + args.timestamp + '_' + args.config)
    print("device", args.device)
    args.seq_len = 60
    criterion = nn.NLLLoss()
    # criterion = nn.NLLLoss(weight=torch.FloatTensor([1, args.weight, args.weight]).to(args.device))
    current_loss = 0
    if args.model == "RNN":
        model = RNN(args.input_size, args.hidden_size, args.n_categories, args.seq_len)
    elif args.model == "DNN":
        model = DNN(args.dims, args.activation)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if "preprocessed.csv" in os.listdir('../ctu-13/CTU-13-Dataset/1'):
        dis_df = pd.read_csv('../ctu-13/CTU-13-Dataset/1/preprocessed.csv')
        print("load dataframe")
    else:
        df = make_conn_dataframe(["1"])
        # dis_df = preprocess(df)
        dis_df = preprocess_stat_CTU(df)
        
        dis_df.to_csv('../ctu-13/CTU-13-Dataset/1/preprocessed.csv')
        print("saved preprocessed file")
    
    label_dict = {"background":0, "normal":1, "botnet":2}
    dis_df["label_num"] = dis_df["class"].apply(lambda x : label_dict[x])
    traindataset = NetworkDataset(args.seq_len, dis_df, "train", args.model)
    trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True)
    print("made train data loader")
    validdataset = NetworkDataset(args.seq_len, dis_df, "valid", args.model)
    validloader = DataLoader(validdataset, batch_size=args.batch_size, shuffle=True)
    print("made valid data loader")
    testdataset = NetworkDataset(args.seq_len, dis_df, "test", args.model)
    testloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=True)
    print("made valid data loader")
    start = time.time()
    recall_lst = []
    precision_lst = []
    max_r_p = 0
    for e in range(args.epochs):
        current_loss = 0
        iter = 0
        correct = 0
        total = 0
        tr_botnet_num = 0
        tr_botnet_cor = 0 
        tr_botnet_pre = 0
        valid_num = 0
        for _, (feature_tensor, label_tensor) in enumerate(trainloader):
            model = model.to(args.device)
            feature_tensor = feature_tensor.to(args.device)
            label_tensor = label_tensor.to(args.device)
            label_tensor = label_tensor.squeeze()
            model.train()
            iter += 1
            model.zero_grad()
            output = model(feature_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if iter % args.log_frequency == 0:
                _, top_i = output.topk(1)
                correct += (top_i == label_tensor.unsqueeze(1)).sum().item()
                total += args.batch_size
                accuracy = correct / total * 100
                print(f'epoch: {e} | {iter*args.batch_size}/{len(trainloader)*args.batch_size} {iter/len(trainloader)*100:.3f}%',\
                        f'({timeSince(start)}) loss : {loss.item():.4f} / accuracy : {accuracy:.2f}%')
                if len((label_tensor.cpu().detach().numpy()==2).nonzero()[0]):
                    tr_botnet_num += len((label_tensor.cpu().detach().numpy()==2).nonzero()[0])
                    botnet_pre = top_i.cpu().detach().numpy()[(label_tensor.cpu().detach().numpy()==2).nonzero()[0]]
                    tr_botnet_cor += (botnet_pre == 2).sum()
                if (top_i == 2).sum().item() > 0:
                    tr_botnet_pre += (top_i == 2).sum().item()
                if tr_botnet_num != 0:
                    tr_recall = tr_botnet_cor / tr_botnet_num
                else:
                    tr_recall = 0 
                if tr_botnet_pre != 0:
                    tr_precision = tr_botnet_cor / tr_botnet_pre
                else:
                    tr_precision = 0
                print(f'recall: {tr_recall:.3f} / precision: {tr_precision:.3f}') 
                c = confusion_matrix(label_tensor.cpu().detach().numpy(), top_i.cpu().detach().numpy(), labels=[0,1,2])
                print("train confusion_matrix\n", c)
                
                writer.add_scalar('Batch Loss', loss.item(), iter*(e+1))
                writer.add_scalar('Cumulative Recall', tr_recall, iter*(e+1))
                writer.add_scalar('Cumulative Precision', tr_precision, iter*(e+1))
                writer.add_scalar('Cumulative Accuracy', accuracy, iter*(e+1))

            # if iter % args.valid_frequency == 0 and iter // args.valid_frequency > 0:
        valid_num += 1
        botnet_cor, botnet_num, botnet_pre, c, acc = validation(model, args.device, args.seq_len, \
                                                    args.hidden_size, validloader)
        if botnet_num !=0:
            recall = botnet_cor / botnet_num
        else:
            recall = 0
        if botnet_pre != 0:
            precision = botnet_cor / botnet_pre
        else:
            precision = 0
        recall_lst.append(recall)
        precision_lst.append(precision)
        print("valid acc", acc)
        print("valid recall", recall)
        print("valid precision", precision )
        print("valid confusion matrix\n", c)
        
        if recall*precision > max_r_p :
            torch.save(model.state_dict(), args.log_dir + args.timestamp + '_' + args.config + '/model_best.pt')
            print("Model saved")
        writer.add_scalar('Valid Recall', recall, valid_num*(e+1))
        writer.add_scalar('Valid Precision', precision, valid_num*(e+1))
        writer.add_scalar('Valied Accuracy', acc, valid_num*(e+1))
                
        print(f'epoch: {e} / loss :{current_loss / (iter+1)}')
        print(f"epoch: {e} / recall: {recall_lst}")
        print(f"epoch: {e} / precision: {precision_lst}")
    
    print("start test")
    best_model = torch.load(args.log_dir + args.timestamp + '_' + args.config + '/model_best.pt')
    model.load_state_dict(best_model)
    botnet_cor, botnet_num, botnet_pre, c, acc = validation(model, args.device, args.seq_len, \
                                                            args.hidden_size, testloader)
    if botnet_num !=0:
        recall = botnet_cor / botnet_num
    else:
        recall = 0
    if botnet_pre != 0:
        precision = botnet_cor / botnet_pre
    else:
        precision = 0
    print("test acc", acc)
    print("test recall", recall)
    print("test precision", precision)
    print("test confusion matrix\n", c)
    
        
if __name__ == "__main__":
    train(get_config())