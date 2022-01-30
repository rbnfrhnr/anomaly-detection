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
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, f1_score, roc_curve
from sklearn import metrics
from model import *
from preprocessing import *
from dataset import *
from config import *
from loss import *
from dataloader import *
import scipy
from scipy.stats import norm
import pickle
import scipy.stats as st
import warnings
from utils import *
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')


def Load_Dataset(args):
    if args.dataset == 'CTU':
        trainset_name = ''.join(args.trainset)
        testset_name = ''.join(args.testset)
        if args.zeek:
            try:
                #  f"preprocessed_129_{args.background}_{args.period_len_ctu}.csv" in os.listdir('../ctu-13/CTU-13-Dataset'):
                dis_df = pd.read_csv(
                    f'../ctu-13/CTU-13-Dataset/preprocessed_129_{args.background}_{args.period_len_ctu}.csv',
                    index_col=[0])
            except:
                print("no dataset from Zeek")
        else:
            # for Zeek
            # df = pd.read_csv('../ctu-13/CTU-13-Dataset/ctu_129_bro_result.csv', index_col=[0])
            # dis_df = preprocess_stat_3(df, args.period_len_ctu, args.rm_ntp)
            # dis_df.to_csv(f'../ctu-13/CTU-13-Dataset/preprocessed_129_{args.background}_{args.period_len_ctu}.csv')
            if f"pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv" in os.listdir(
                    '../ctu-13/CTU-13-Dataset'):
                dis_df = pd.read_csv(
                    f'../ctu-13/CTU-13-Dataset/pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv', \
                    index_col=[0])
            else:
                print("make dataset")
                df = make_conn_dataframe(args.trainset, args.scenario)
                dis_df = preprocess_stat_2_CTU(df, args.period_len_ctu, args.rm_ntp)
                # dis_df = preprocess_stat_CTU(df)
                dis_df.to_csv(
                    f'../ctu-13/CTU-13-Dataset/pre_vae_{trainset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}.csv')
        # Zeek
        norm_sample = dis_df[dis_df["class"] == 'normal'].sample(frac=args.sample_rate_ctu, replace=False,
                                                                 random_state=1)
        norm_sample_ips = norm_sample["id.orig_h"].unique()
        remove_idx = dis_df.index[dis_df["id.orig_h"].isin(norm_sample_ips)].tolist()
        dis_df = dis_df.drop(remove_idx)
        if args.all_scenario:
            total_df = dis_df
            dis_df, test_df = train_test_split(total_df, test_size=0.3, random_state=1234)

        label_dict = {"normal": 0, "botnet": 1, 'background': 0}
        # label_dict = {"normal":0, "botnet":1, "background":0}
        dis_df["label_num"] = dis_df["class"].apply(lambda x: label_dict[x])
        print("total length", len(dis_df))
        # print(dis_df.columns)
        # print(dis_df.head())
        print("Source Domain train/valid set :", Counter(dis_df["class"]))
        eval_abnorm_rate = Counter(dis_df["class"])["botnet"] / len(dis_df)

        same_column = list(dis_df.columns)
        traindataset_cv = NetworkDataset_ae(dis_df, "train", args.model, same_column, args.preprocess, args.total_seq, \
                                            args.overlap, 5, args.dataset)  # cross validation dataset

        if not args.all_scenario:
            if f"pre_vae_{testset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}_{args.scenario}.csv" in os.listdir(
                    '../ctu-13/CTU-13-Dataset'):
                test_df = pd.read_csv(
                    f'../ctu-13/CTU-13-Dataset/pre_vae_{testset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}_{args.scenario}.csv', \
                    index_col=[0])
                print("load test dataframe")
            else:
                test_df = make_conn_dataframe(args.testset, args.scenario)
                # test_df = preprocess_stat_CTU(test_df)
                test_df = preprocess_stat_2_CTU(test_df, args.period_len_ctu, args.rm_ntp)
                test_df.to_csv(
                    f'../ctu-13/CTU-13-Dataset/pre_vae_{testset_name}_{args.preprocess}_{args.onlyconnect}_{args.period_len_ctu}_{args.rm_ntp}_{args.scenario}.csv')
                print("saved test preprocessed file")

        print("Source domain test set :", Counter(test_df["class"]))
        test_df["label_num"] = test_df["class"].apply(lambda x: label_dict[x])
        test_abnorm_rate = Counter(test_df["class"])["botnet"] / len(test_df)
        print(test_df.columns)
        testdataset = NetworkDataset_ae(test_df, "test", args.model, same_column, args.preprocess, args.total_seq,
                                        args.overlap, 5, args.dataset)

        print("test set count", len(testdataset))
        if args.model == 'RNN':
            testloader = LSTM_VAE_dataloader(testdataset, batch_size=args.batch_size, num_workers=0)
        else:
            testloader = DataLoader(testdataset, batch_size=args.batch_size, num_workers=2)
        print("made test data loader")
        return testloader, eval_abnorm_rate, test_abnorm_rate, traindataset_cv
    else:
        if args.sampling_data:
            if args.only_shared:
                if f"pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}_shared.csv" in os.listdir(
                        './data/KISTI'):
                    target_df = pd.read_csv(
                        f'./data/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}_shared.csv',
                        index_col=[0])
                else:
                    df = pd.read_csv('data/KISTI/kisti_logs_20190714_6_shared.csv')
                    target_df = preprocess_stat_KISTI(df, args.period_len_kisti, args.rm_ntp)
                    target_df.to_csv(
                        f'./data/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}_shared.csv')
            else:
                if f"preprocessed_kisti_{args.period_len_kisti}.csv" in os.listdir('./data/KISTI'):
                    target_df = pd.read_csv(f'./data/KISTI/preprocessed_kisti_{args.period_len_kisti}.csv',
                                            index_col=[0])
                else:
                    df = pd.read_csv('./data/KISTI/kisti_bro_result.csv')
                    target_df = preprocess_stat_3(df, args.period_len_kisti, args.rm_ntp)
                    target_df.to_csv(f'./data/KISTI/preprocessed_kisti_{args.period_len_kisti}.csv')
                # if f"pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv" in os.listdir('./data/KISTI'):
                #     target_df = pd.read_csv(f'./data/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv',index_col=[0])
                # else:
                #     df = pd.read_csv('data/KISTI/kisti_logs_20190714_6.csv')
                #     target_df = preprocess_stat_KISTI(df, args.period_len_kisti, args.rm_ntp)
                #     target_df.to_csv(f'./data/KISTI/pre_vae_{args.preprocess}_{args.onlyconnect}_{args.period_len_kisti}_{args.rm_ntp}_{args.sampling_data}.csv')
            print("saved target preprocessed file")

        norm_sample = target_df[target_df["class"] == 'norm'].sample(frac=args.sample_rate_kisti, replace=False,
                                                                     random_state=1)
        norm_sample_ips = norm_sample["id.orig_h"].unique()
        remove_idx = target_df.index[target_df["id.orig_h"].isin(norm_sample_ips)].tolist()
        target_df = target_df.drop(remove_idx)

        label_dict = {"norm": 0, "abnorm": 1}
        target_df["label_num"] = target_df["class"].apply(lambda x: label_dict[x])
        kisti_abnorm_rate = Counter(target_df["class"])["abnorm"] / len(target_df)
        print("KISTI set :", Counter(target_df["class"]))
        train_df = target_df[:int(len(target_df) * 0.7)]
        test_df = target_df[int(len(target_df) * 0.7):]

        norm_df = train_df[train_df["label_num"] == 0]
        same_column = list(target_df.columns)
        ## need to be changed
        traindataset = NetworkDataset_ae(norm_df, "train", args.model, same_column, args.preprocess, args.total_seq,
                                         args.overlap, 5, args.dataset)
        print("kisti train", len(traindataset))
        if args.model == 'RNN':
            kisti_trainloader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size, num_workers=2,
                                                    shuffle=True)
        else:
            kisti_trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        print("made train target data loader")

        validdataset = NetworkDataset_ae(train_df, "valid", args.model, same_column, args.preprocess, args.total_seq,
                                         args.overlap, 5, args.dataset)
        if args.model == 'RNN':
            kisti_mixed_validloader = LSTM_VAE_dataloader(validdataset, batch_size=args.batch_size, num_workers=2,
                                                          shuffle=True)
        else:
            kisti_mixed_validloader = DataLoader(validdataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        print("made valid target data loader")
        print("the number of target domain data", len(target_df))

        inference_kisti = NetworkDataset_ae(test_df, "test", args.model, same_column, args.preprocess, args.total_seq,
                                            args.overlap, 5, args.dataset)
        print("# of inference dataset :", len(inference_kisti))
        if args.model == 'DNN':
            inference_loader = DataLoader(inference_kisti, batch_size=args.batch_size, shuffle=True, num_workers=2)
        else:
            inference_loader = LSTM_VAE_dataloader(inference_kisti, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=2)

        return kisti_trainloader, kisti_mixed_validloader, inference_loader, kisti_abnorm_rate


def Evaluation(args, mix_validloader, model, step, e, writer, eval_abnorm_rate, dists_info, results):
    if args.only_infer:
        save_dir = args.log_dir + args.load_model
    else:
        save_dir = args.log_dir + args.timestamp + '_' + args.config
    label_lst = []
    recon_lst_val = []
    for k, data in enumerate(mix_validloader):
        if args.model == 'DNN':
            feature, label = data
            feature = feature.to(args.device)
            mu, logvar, recon, _ = model(feature)
            label = label.to(args.device)
            val_loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], \
                                          args.device, step, args.x0, args.KLD_W)
        else:
            encode_x, label, _ = data
            encode_x = encode_x.to(args.device)
            mu, logvar, recon, feature, lens, _ = model(encode_x)
            val_loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, args.device, \
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
            bce_select = bce_arr.nonzero()  # to flatten bce loss, discard masked value
            rec = bce_arr[bce_select]  # flatten bce loss
            recon_lst_val.extend(rec)  # flatten bce loss
            label_lst.extend([l for l in label])

    if args.model == 'RNN':
        flatten_l = [l for ls in label_lst for l in ls]
        label_lst = flatten_l
    if args.evaluation == 'threshold':
        asl = np.asarray(recon_lst_val)
        sort_idx = np.argsort(asl)
        lens = len(asl)
        abnorm_idx = sort_idx[-int(lens * (eval_abnorm_rate)):]
        pred_y = np.zeros_like(asl)
        pred_y[abnorm_idx] = 1
        error_df = pd.DataFrame({'Reconstruction_error': recon_lst_val, 'True_class': label_lst})
    else:
        error_df = pd.DataFrame({'Reconstruction_error': recon_lst_val, 'True_class': label_lst})
        label_lst = np.asarray(label_lst)
        recon_lst = np.asarray(recon_lst_val)
        botnet_index = (label_lst == 1).nonzero()[0]
        print("the number of botnet_index in validation set :", len(botnet_index))
        mask = np.ones(len(recon_lst), dtype=bool)
        mask[botnet_index] = False
        # abnorm_param = recon_distribution(recon_lst[botnet_index])
        # norm_param = recon_distribution(recon_lst[mask])
        # print(f"Mean of Abnorm error : {abnorm_param[0]:.3f}")
        # print(f"Mean of Norm error : {norm_param[0]:.3f}")
        # print(f"std of Abnorm error :{abnorm_param[1]:.3f}")
        # print(f"std of Norm error {norm_param[1]:.3f}")
        # if isinstance(e, int):
        #     writer.add_scalars(f'Valid Mean of error distribution', \
        #                         {'mean of abnorm error':abnorm_param[0],
        #                             'mean of norm error':norm_param[0]}, e)
        #     writer.add_scalars(f'Valid std of error distribution', \
        #                         {'std of abnorm error':abnorm_param[1],
        #                             'std of norm error':norm_param[1]}, e)
        if isinstance(e, int):
            if e % args.eval_frequency == 0:
                abnorm_dist_n, abparams = best_fit_distribution(recon_lst[botnet_index])
                abarg = abparams[:-2]
                abloc = abparams[-2]
                abscale = abparams[-1]
                best_dist_abnorm = getattr(st, abnorm_dist_n)
                print("abnorm pdf", abnorm_dist_n)
                norm_dist, params = best_fit_distribution(recon_lst[mask])
                print("normal pdf", norm_dist)
                best_dist_norm = getattr(st, norm_dist)
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
            narg = params[:-2]
            nloc = params[-2]
            nscale = params[-1]

            ## cv
            if results:
                if 'normal' not in results.keys():
                    results["normal"] = recon_lst[mask].tolist()
                    results["abnorm"] = recon_lst[botnet_index].tolist()
                else:
                    results["normal"].extend(recon_lst[mask].tolist())
                    results["abnorm"].extend(recon_lst[botnet_index].tolist())

        pred_y = list()
        pred_y_auc = list()
        # for er in error_df.Reconstruction_error.values:
        #     if norm.pdf(er,loc=abnorm_param[0],scale=abnorm_param[1]) > norm.pdf(er,loc=norm_param[0],scale=norm_param[1]):
        #         pred_y.append(1)
        #         pred_y_auc.append(1-(norm.pdf(er,loc=norm_param[0],scale=norm_param[1]) / norm.pdf(er,loc=abnorm_param[0],scale=abnorm_param[1])))
        #     else:
        #         pred_y.append(0)
        #         pred_y_auc.append(0)
        for er in error_df.Reconstruction_error.values:
            if best_dist_abnorm.pdf(er, loc=abloc, scale=abscale, *abarg) > best_dist_norm.pdf(er, loc=nloc,
                                                                                               scale=nscale, *narg):
                pred_y.append(1)
                pred_y_auc.append(
                    1 - best_dist_norm.pdf(er, loc=nloc, scale=nscale, *narg) / best_dist_abnorm.pdf(er, loc=abloc,
                                                                                                     scale=abscale,
                                                                                                     *abarg))
            else:
                pred_y.append(0)
                pred_y_auc.append(0)

    abnorm_num = len((np.asarray(label_lst) == 1).nonzero()[0])
    abnorm_predict = np.asarray(pred_y)[(np.asarray(label_lst) == 1).nonzero()[0]]
    abnorm_cor = (abnorm_predict == 1).sum()
    abnorm_pre = (np.asarray(pred_y) == 1).sum()
    valid_recall = abnorm_cor / abnorm_num
    valid_precision = abnorm_cor / abnorm_pre
    if args.evaluation == 'threshold':
        if 1 < error_df.True_class.unique().shape[0]:
            auc = roc_auc_score(error_df.True_class, error_df.Reconstruction_error)
        else:
            auc = 0
        ps, rs, _ = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    else:
        if 1 < error_df.True_class.unique().shape[0]:
            auc = roc_auc_score(error_df.True_class, pred_y_auc)
        else:
            auc =0
        ps, rs, _ = precision_recall_curve(error_df.True_class, pred_y_auc)
    f1 = f1_score(error_df.True_class, pred_y)
    prauc = metrics.auc(rs, ps)
    # print(f"valid recall : {valid_recall:.3f}")
    # print(f"valid precision : {valid_precision:.3f}")
    # print(f"valid AUROC :{auc:.4f} | f1 score :{f1:.4f} | PRAUC: {prauc:.4f}")

    if results is None or args.dataset == 'KISTI':
        results = dict()
        results['recall'] = valid_recall
        results['precision'] = valid_precision
        results['auroc'] = auc
        results['f1'] = f1
        results['prauc'] = prauc
    else:
        results['recall'] += valid_recall
        results['precision'] += valid_precision
        results['auroc'] += auc
        results['f1'] += f1
        results['prauc'] += prauc
    if isinstance(e, int):
        # print(f"valid recall : {results['recall']/5:.3f}")
        # print(f"valid precision : {results['precision']/5:.3f}")
        # print(f"valid AUROC :{results['auroc']/5:.4f} | f1 score :{results['f1']/5:.4f} | PRAUC: {results['prauc']/5:.4f}")
        writer.add_scalar('Valid recall', valid_recall, e)
        writer.add_scalar('Valid precision', valid_precision, e)
        writer.add_scalar('Valid ROAUC', auc, e)
        writer.add_scalar('Valid PRAUC', prauc, e)
        writer.add_scalar('Valid f1', f1, e)
    else:
        if args.evaluation != 'threshold':
            botnet_values = recon_lst[botnet_index]
            normal_values = recon_lst[mask]
            with open(save_dir + f'/{args.dataset}_valid_botnet_values.pkl', 'wb') as f:
                pickle.dump(botnet_values, f)
            with open(save_dir + f'/{args.dataset}_valid_normal_values.pkl', 'wb') as f:
                pickle.dump(normal_values, f)
            print('saved reconstruction values')
        # result_dict = {'final recall': valid_recall,
        #                 'final precision': valid_precision,
        #                 'final AUROC': auc,
        #                 'final AUPRC': prauc,
        #                 'final f1': f1}
        fpr, tpr, _ = roc_curve(error_df.True_class, error_df.Reconstruction_error)
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
        plt.savefig(save_dir + f'/{args.evaluation}_valid_roc_curve.jpg')

        plt.figure()
        lw = 2
        plt.plot(rs, ps, color='darkorange', lw=lw, label='PR curve (area = %0.2f)' % prauc)
        plt.plot([0, 1], [eval_abnorm_rate, eval_abnorm_rate], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower right")
        plt.title('Receiver operating characteristic example')
        plt.savefig(save_dir + f'/{args.evaluation}_valid_PR_curve.jpg')
        # with open(save_dir + f'/best_model_{args.evaluation}_valid_result.pkl', 'wb') as f:
        #     pickle.dump(result_dict, f)
    # print(f"threshold :{current_loss/(i+1)*1.5} | valid AUC :{auc} | f1 score :{f1}")
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    print(f"epoch : {e} \n valid confusion matrix\n", conf_matrix)
    if args.evaluation == 'threshold':
        return results
    else:
        return (abnorm_dist_n, abparams, norm_dist, params), results


def inference(args, model, inference_loader, writer, e, step, test_abnorm_rate, dists_info):
    if args.only_infer:
        save_dir = args.log_dir + args.load_model
    else:
        save_dir = args.log_dir + args.timestamp + '_' + args.config
    test_label_lst = []
    recon_lst_test = []
    order_lst = []
    for w, data in enumerate(inference_loader):
        if args.model == 'DNN':
            feature, label = data
            feature = feature.to(args.device)
            mu, logvar, recon, _ = model(feature)
            testloss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], \
                                          args.device, step, args.x0, args.KLD_W)
        else:
            encode_x, label, order = data
            encode_x = encode_x.to(args.device)
            mu, logvar, recon, feature, lens, _ = model(encode_x)
            testloss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, \
                                          args.device, step, args.x0, args.KLD_W)
        if args.model == 'DNN':
            # recon_lst_test.extend(testloss.cpu().detach().numpy())
            recon_lst_test.extend(bce.cpu().detach().numpy())
            test_label_lst.extend(label.squeeze().cpu().detach().numpy())
        else:
            bce_arr = bce.cpu().detach().numpy()
            bce_select = bce_arr.nonzero()
            rec = bce_arr[bce_select]  # flatten bce loss
            recon_lst_test.extend(rec)  # flatten bce loss
            test_label_lst.extend([l for l in label])
            order_lst.append(order)
    if args.model == 'RNN':
        flatten_l = [l for ls in test_label_lst for l in ls]
        test_label_lst = flatten_l
    if args.evaluation == 'threshold':
        asl = np.asarray(recon_lst_test)
        sort_idx = np.argsort(asl)
        lens = len(asl)
        abnorm_idx = sort_idx[-int(lens * (test_abnorm_rate)):]
        pred_y = np.zeros_like(asl)
        pred_y[abnorm_idx] = 1
        error_df = pd.DataFrame({'Reconstruction_error': recon_lst_test, 'True_class': test_label_lst})
    else:
        error_df = pd.DataFrame({'Reconstruction_error': recon_lst_test, 'True_class': test_label_lst})
        label_lst = np.asarray(test_label_lst)
        recon_lst = np.asarray(recon_lst_test)
        botnet_index = (label_lst == 1).nonzero()[0]
        print("the number of botnet_index test set :", len(botnet_index))
        mask = np.ones(len(recon_lst), dtype=bool)
        mask[botnet_index] = False
        # abnorm_param = recon_distribution(recon_lst[botnet_index])
        # norm_param = recon_distribution(recon_lst[mask])
        # print(f"Mean of Abnorm error : {abnorm_param[0]:.3f}")
        # print(f"Mean of Norm error : {norm_param[0]:.3f}")
        # print(f"std of Abnorm error :{abnorm_param[1]:.3f}")
        # print(f"std of Norm error {norm_param[1]:.3f}")
        # if isinstance(e, int):
        #     writer.add_scalars('Test Mean of error distribution', \
        #                         {'mean of abnorm error':abnorm_param[0],
        #                             'mean of norm error':norm_param[0],}, e)
        #     writer.add_scalars('Test std of error distribution', \
        #                         {'std of abnorm error':abnorm_param[1],
        #                         'std of norm error':norm_param[1],}, e)
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
            abnorm_dist_n, params = best_fit_distribution(recon_lst[botnet_index])
            abarg = params[:-2]
            abloc = params[-2]
            abscale = params[-1]
            best_dist_abnorm = getattr(st, abnorm_dist_n)
            mask = np.ones(len(recon_lst), dtype=bool)
            mask[botnet_index] = False
            norm_dist, params = best_fit_distribution(recon_lst[mask])
            narg = params[:-2]
            nloc = params[-2]
            nscale = params[-1]
            best_dist_norm = getattr(st, norm_dist)
        pred_y = list()
        pred_y_auc = list()
        # for er in error_df.Reconstruction_error.values:
        #     if norm.pdf(er,loc=abnorm_param[0],scale=abnorm_param[1]) > norm.pdf(er,loc=norm_param[0],scale=norm_param[1]):
        #         pred_y.append(1)
        #         pred_y_auc.append(1-(norm.pdf(er,loc=norm_param[0],scale=norm_param[1]) / norm.pdf(er,loc=abnorm_param[0],scale=abnorm_param[1])))
        #     else:
        #         pred_y.append(0)
        #         pred_y_auc.append(0)
        for er in error_df.Reconstruction_error.values:
            if best_dist_abnorm.pdf(er, loc=abloc, scale=abscale, *abarg) > best_dist_norm.pdf(er, loc=nloc,
                                                                                               scale=nscale, *narg):
                pred_y.append(1)
                pred_y_auc.append(
                    1 - best_dist_norm.pdf(er, loc=nloc, scale=nscale, *narg) / best_dist_abnorm.pdf(er, loc=abloc,
                                                                                                     scale=abscale,
                                                                                                     *abarg))
            else:
                pred_y.append(0)
                pred_y_auc.append(0)
    abnorm_num = len((np.asarray(test_label_lst) == 1).nonzero()[0])
    abnorm_predict = np.asarray(pred_y)[(np.asarray(test_label_lst) == 1).nonzero()[0]]
    abnorm_cor = (abnorm_predict == 1).sum()
    abnorm_pre = (np.asarray(pred_y) == 1).sum()
    test_recall = abnorm_cor / abnorm_num
    test_precision = abnorm_cor / abnorm_pre
    print(f"test recall : {test_recall:.3f}")
    print(f"test precision : {test_precision:.3f}")
    if args.evaluation == 'threshold':
        auc = roc_auc_score(error_df.True_class, error_df.Reconstruction_error)
        ps, rs, _ = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    else:
        auc = roc_auc_score(error_df.True_class, pred_y_auc)
        ps, rs, _ = precision_recall_curve(error_df.True_class, pred_y_auc)
    f1 = f1_score(error_df.True_class, pred_y)
    prauc = metrics.auc(rs, ps)
    print(f"test AUROC :{auc:.4f} | f1 score :{f1:.4f} | PRAUC: {prauc:.4f}")
    if isinstance(e, int):
        writer.add_scalar('test recall', test_recall, e)
        writer.add_scalar('test precision', test_precision, e)
        writer.add_scalar('test ROAUC', auc, e)
        writer.add_scalar('test PRAUC', prauc, e)
        writer.add_scalar('test f1', f1, e)
    else:
        result_dict = {'final recall': test_recall,
                       'final precision': test_precision,
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
        plt.savefig(save_dir + f'/{args.evaluation}_test_roc_curve.jpg')

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
        plt.savefig(save_dir + f'/{args.evaluation}_test_PR_curve.jpg')
        with open(save_dir + f'/best_model_{args.evaluation}_test_result.pkl', 'wb') as f:
            pickle.dump(result_dict, f)
        if args.evaluation != 'threshold':
            recon_lst_dict = dict()
            recon_lst_dict["recon_lst"] = recon_lst
            recon_lst_dict["botnet_index"] = botnet_index
            recon_lst_dict["order"] = order_lst
            with open(save_dir + f'/{args.period_len_ctu}_{args.dataset}_test_recon_lst.pkl', 'wb') as f:
                pickle.dump(recon_lst_dict, f)
            botnet_values = recon_lst[botnet_index]
            normal_values = recon_lst[mask]
            with open(save_dir + f'/{args.period_len_ctu}_{args.dataset}_{args.scenario}_test_botnet_values.pkl',
                      'wb') as f:
                pickle.dump(botnet_values, f)
            with open(save_dir + f'/{args.period_len_ctu}_{args.dataset}_{args.scenario}_test_normal_values.pkl',
                      'wb') as f:
                pickle.dump(normal_values, f)
            print('saved reconstruction values')
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    print(f"epoch : {e} \n test confusion matrix\n", conf_matrix)
    TN = conf_matrix[0][0]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]
    FP = conf_matrix[0][1]
    tpr = TP / (TP + FN)
    tnr = TN / (TN + FP)
    fpr = FP / (FP + TN)
    fnr = FN / (TP + FN)
    print(f"TPR:{tpr} | TNR:{tnr} | FPR:{fpr} | FNR:{fnr}")


def rnn_generator(traindataset):
    for z in range(5):
        if z == 0:
            yield z, (list(range(traindataset.cv_idx_train[z])),
                      list(range(traindataset.cv_idx_train[z], traindataset.cv_idx_valid[z])))
        else:
            yield z, (list(range(traindataset.cv_idx_valid[z - 1], traindataset.cv_idx_train[z])),
                      list(range(traindataset.cv_idx_train[z], traindataset.cv_idx_valid[z])))


def train(args):
    print("device", args.device)
    print("dim", args.dims)
    if args.dataset == 'CTU':
        testloader, eval_abnorm_rate, test_abnorm_rate, traindataset = Load_Dataset(args)
    else:
        kisti_trainloader, kisti_mixed_validloader, inference_loader, kisti_abnorm_rate = Load_Dataset(args)
    if args.dataset == 'KISTI':
        train_loader = kisti_trainloader
        mixed_valid_loader = kisti_mixed_validloader
        testloader = inference_loader
        eval_abnorm_rate = kisti_abnorm_rate
        test_abnorm_rate = kisti_abnorm_rate
    if args.model == "RNN":
        model = RNN_VAE(args.input_size, args.hidden_size, args.num_layer, args.batch_size, \
                        args.latent_size, args.bidirectional).to(args.device)
    elif args.model == 'DNN':
        model = DNN_VAE(args.dims, args.activation).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.only_infer:
        save_dir = args.log_dir + args.load_model
        writer = SummaryWriter(args.log_dir + args.load_model)
        model.load_state_dict(torch.load(save_dir + '/best_model.pt'))
        # if args.dataset == 'CTU':
        #     if args.model == 'DNN':
        #         kf = KFold(n_splits=5, shuffle=True, random_state=1234)
        #         cv_iterable = enumerate(kf.split(traindataset.X))
        #     else:
        #         cv_iterable = rnn_generator(traindataset)
        #     results = None
        #     for cv_idx, (_, valid_index) in cv_iterable:
        #         valid_sampler = torch.utils.data.SubsetRandomSampler(valid_index)
        #         if args.model == 'RNN':
        #             mixed_valid_loader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size, \
        #                                                         num_workers=2, sampler=valid_sampler)
        #         else:
        #             mixed_valid_loader = DataLoader(traindataset, batch_size=args.batch_size, \
        #                                                 sampler=valid_sampler, num_workers=2)
        #         if args.evaluation == 'threshold':
        #             results = Evaluation(args, mixed_valid_loader, model, 10000, 'final', writer, eval_abnorm_rate, dists, results)
        #         else:
        #             dists = None
        #             dists, results = Evaluation(args, mixed_valid_loader, model, 10000, 'final', writer, eval_abnorm_rate, dists, results)
        #         if cv_idx == 4:
        #             results['recall'] /= 5
        #             results['precision'] /= 5
        #             results['auroc'] /= 5
        #             results['f1'] /= 5
        #             results['prauc'] /= 5
        #             print(f"valid recall : {results['recall']:.3f}")
        #             print(f"valid precision : {results['precision']:.3f}")
        #             print(f"valid AUROC :{results['auroc']:.4f} | f1 score :{results['f1']:.4f} | PRAUC: {results['prauc']:.4f}")
        #             with open(save_dir + f'/best_model_{args.evaluation}_valid_result.pkl', 'wb') as f:
        #                 pickle.dump(results, f)
        ##inference
        ## add
        if args.evaluation != 'threshold':
            with open(save_dir + '/valid_dists_info.pkl', 'rb') as f:
                dists = pickle.load(f)
            abnorm_dist_n, abparams, norm_dist, params = dists
            # abnorm_dist_n, abparams = best_fit_distribution(np.asarray(results["abnorm"]))
            # norm_dist, params = best_fit_distribution(np.asarray(results["normal"]))
            # dists = (abnorm_dist_n, abparams, norm_dist, params)
            print(f"abnorm : {abnorm_dist_n} | norm_dist : {norm_dist}")
            print("dists :", dists)
            # botnet_values = results["abnorm"]
            # normal_values = results["normal"]
            # with open(save_dir +f'/{args.period_len_ctu}_{args.dataset}_valid_botnet_values.pkl', 'wb') as f:
            #     pickle.dump(botnet_values, f)
            # with open(save_dir +f'/{args.period_len_ctu}_{args.dataset}_valid_normal_values.pkl', 'wb') as f:
            #     pickle.dump(normal_values, f)
            # print('saved reconstruction values')
        ###
        inference(args, model, testloader, writer, 'final', 10000, test_abnorm_rate, dists)
        # else:
        #     if args.evaluation == 'threshold':
        #         results = Evaluation(args, mixed_valid_loader, model, 10000, 'final', writer, eval_abnorm_rate, dists, results)
        #     else:
        #         dists, results = Evaluation(args, mixed_valid_loader, model, 10000, 'final', writer, eval_abnorm_rate, dists, results)
        #     save_dir = args.log_dir + args.timestamp + '_' + args.config
        #     with open(save_dir + f'/best_model_{args.evaluation}_valid_result.pkl', 'wb') as f:
        #         pickle.dump(results, f)
        #     ##inference
        #     inference(args, model, testloader, writer, 'final', 10000, test_abnorm_rate, dists)
        print("only inference")
    else:
        writer = SummaryWriter(args.log_dir + args.timestamp + '_' + args.config)
        current_loss = 0
        step = 0
        best_prauc = 0
        if args.dataset == 'CTU' and args.model == 'DNN':
            kf = KFold(n_splits=5, shuffle=True, random_state=1234)
        for e in range(args.epochs):
            print("epoch : ", e)
            current_loss = 0
            results = None
            if args.dataset == 'CTU':
                if args.model == 'DNN':
                    cv_iterable = enumerate(kf.split(traindataset.X))
                else:
                    cv_iterable = rnn_generator(traindataset)
                for cv_idx, (train_index, valid_index) in cv_iterable:
                    print(f"{cv_idx} fold")
                    # print(train_index)
                    # print(valid_index)
                    if args.model == 'DNN':
                        normal_idx = (traindataset.y[train_index] == 0).nonzero()[0]  # only normal data
                        train_index = train_index[normal_idx]
                    train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
                    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_index)
                    if args.model == 'RNN':
                        train_loader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size, \
                                                           num_workers=2, sampler=train_sampler)
                        mixed_valid_loader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size, \
                                                                 num_workers=2, sampler=valid_sampler)
                    else:
                        train_loader = DataLoader(traindataset, batch_size=args.batch_size, \
                                                  sampler=train_sampler, num_workers=2)
                        mixed_valid_loader = DataLoader(traindataset, batch_size=args.batch_size, \
                                                        sampler=valid_sampler, num_workers=2)
                    for i, data in enumerate(train_loader):
                        model.train()
                        model.zero_grad()
                        if args.model == 'DNN':
                            feature, _ = data
                            feature = feature.to(args.device)
                            mu, logvar, recon, _ = model(feature)
                            loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], \
                                                      args.device, step, args.x0, args.KLD_W)
                        else:
                            encode_x, _, _ = data
                            encode_x = encode_x.to(args.device)
                            mu, logvar, recon, feature, lens, _ = model(encode_x)
                            loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, \
                                                      args.device, step, args.x0, args.KLD_W)
                        loss = loss.mean()
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        current_loss += loss.item()
                        writer.add_scalar('Batch Loss', loss.item(), i + e * len(train_loader))
                        if args.model == 'DNN':
                            writer.add_scalar('Recon Loss', bce.mean().item(), i + e * len(train_loader))
                        else:
                            writer.add_scalar('Recon Loss', bce.sum(1).mean().item(), i + e * len(train_loader))
                        writer.add_scalar('KLD', kld.mean().item(), i + e * len(train_loader))
                        if i % args.log_frequency == 0:
                            print(f'progress : {i / len(train_loader) * 100 :.0f}%, batch loss : {loss.item():.3f}')
                        step += 1
                    model.eval()
                    if e % args.eval_frequency == 0:
                        if e == 0:
                            dists = None
                        if args.evaluation == 'threshold':
                            results = Evaluation(args, mixed_valid_loader, model, step, e, writer, eval_abnorm_rate,
                                                 dists, results)
                        else:
                            dists, results = Evaluation(args, mixed_valid_loader, model, step, e, writer,
                                                        eval_abnorm_rate, dists, results)
                        if cv_idx == 4:
                            print(f"valid recall : {results['recall'] / 5:.3f}")
                            print(f"valid precision : {results['precision'] / 5:.3f}")
                            print(
                                f"valid AUROC :{results['auroc'] / 5:.4f} | f1 score :{results['f1'] / 5:.4f} | PRAUC: {results['prauc'] / 5:.4f}")
                            if results['prauc'] / 5 > best_prauc:
                                best_prauc = results['prauc'] / 5
                                torch.save(model.state_dict(),
                                           args.log_dir + args.timestamp + '_' + args.config + '/best_model.pt')
                                print("saved the best model!")
                                with open(args.log_dir + args.timestamp + '_' + args.config + '/valid_dists_info.pkl',
                                          'wb') as f:
                                    pickle.dump(dists, f)
                                print("dist info saved")
                            inference(args, model, testloader, writer, e, step, test_abnorm_rate, dists)
            else:
                for i, data in enumerate(train_loader):
                    model.train()
                    model.zero_grad()
                    if args.model == 'DNN':
                        feature, _ = data
                        feature = feature.to(args.device)
                        mu, logvar, recon, _ = model(feature)
                        loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, [], \
                                                  args.device, step, args.x0, args.KLD_W)
                    else:
                        encode_x, _, _ = data
                        encode_x = encode_x.to(args.device)
                        mu, logvar, recon, feature, lens, _ = model(encode_x)
                        loss, bce, kld = vae_loss(recon, feature, mu, logvar, args.model, lens, \
                                                  args.device, step, args.x0, args.KLD_W)
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    current_loss += loss.item()
                    writer.add_scalar('Batch Loss', loss.item(), i + e * len(train_loader))
                    if args.model == 'DNN':
                        writer.add_scalar('Recon Loss', bce.mean().item(), i + e * len(train_loader))
                    else:
                        writer.add_scalar('Recon Loss', bce.sum(1).mean().item(), i + e * len(train_loader))
                    writer.add_scalar('KLD', kld.mean().item(), i + e * len(train_loader))
                    if i % args.log_frequency == 0:
                        print(f'progress : {i / len(train_loader) * 100 :.0f}%, batch loss : {loss.item():.3f}')
                    step += 1
                model.eval()
                if e % args.eval_frequency == 0:
                    if e == 0:
                        dists = None
                    if args.evaluation == 'threshold':
                        results = Evaluation(args, mixed_valid_loader, model, step, e, writer, eval_abnorm_rate, dists,
                                             results)
                    else:
                        dists, results = Evaluation(args, mixed_valid_loader, model, step, e, writer, eval_abnorm_rate,
                                                    dists, results)
                    print(f"valid recall : {results['recall']:.3f}")
                    print(f"valid precision : {results['precision']:.3f}")
                    print(
                        f"valid AUROC :{results['auroc']:.4f} | f1 score :{results['f1']:.4f} | PRAUC: {results['prauc']:.4f}")
                    if results['prauc'] > best_prauc:
                        best_prauc = results['prauc']
                        torch.save(model.state_dict(),
                                   args.log_dir + args.timestamp + '_' + args.config + '/best_model.pt')
                        print("saved the best model!")

                    inference(args, model, testloader, writer, e, step, test_abnorm_rate, dists)

        model.load_state_dict(torch.load(args.log_dir + args.timestamp + '_' + args.config + '/best_model.pt'))
        print("The best model loaded")
        results = None
        ## cross validation
        if args.dataset == 'CTU':
            if args.model == 'DNN':
                cv_iterable = enumerate(kf.split(traindataset.X))
            else:
                cv_iterable = rnn_generator(traindataset)
            for cv_idx, (_, valid_index) in cv_iterable:
                valid_sampler = torch.utils.data.SubsetRandomSampler(valid_index)
                if args.model == 'RNN':
                    mixed_valid_loader = LSTM_VAE_dataloader(traindataset, batch_size=args.batch_size, \
                                                             num_workers=2, sampler=valid_sampler)
                else:
                    mixed_valid_loader = DataLoader(traindataset, batch_size=args.batch_size, \
                                                    sampler=valid_sampler, num_workers=2)
                if args.evaluation == 'threshold':
                    results = Evaluation(args, mixed_valid_loader, model, step, 'final', writer, eval_abnorm_rate,
                                         dists, results)
                else:
                    dists, results = Evaluation(args, mixed_valid_loader, model, step, 'final', writer,
                                                eval_abnorm_rate, dists, results)
                if cv_idx == 4:
                    results['recall'] /= 5
                    results['precision'] /= 5
                    results['auroc'] /= 5
                    results['f1'] /= 5
                    results['prauc'] /= 5
                    print(f"valid recall : {results['recall']:.3f}")
                    print(f"valid precision : {results['precision']:.3f}")
                    print(
                        f"valid AUROC :{results['auroc']:.4f} | f1 score :{results['f1']:.4f} | PRAUC: {results['prauc']:.4f}")
                    if args.only_infer:
                        save_dir = args.log_dir + args.load_model
                    else:
                        save_dir = args.log_dir + args.timestamp + '_' + args.config
                    with open(save_dir + f'/best_model_{args.evaluation}_valid_result.pkl', 'wb') as f:
                        pickle.dump(results, f)
            ##inference
            ## add
            if args.evaluation != 'threshold':
                abnorm_dist_n, abparams = best_fit_distribution(np.asarray(results["abnorm"]))
                norm_dist, params = best_fit_distribution(np.asarray(results["normal"]))
                dists = (abnorm_dist_n, abparams, norm_dist, params)
                print(f"abnorm : {abnorm_dist_n} | norm_dist : {norm_dist}")
                botnet_values = results["abnorm"]
                normal_values = results["normal"]
                with open(save_dir + f'/{args.dataset}_valid_botnet_values.pkl', 'wb') as f:
                    pickle.dump(botnet_values, f)
                with open(save_dir + f'/{args.dataset}_valid_normal_values.pkl', 'wb') as f:
                    pickle.dump(normal_values, f)
                print('saved reconstruction values')
            ###
            inference(args, model, testloader, writer, 'final', step, test_abnorm_rate, dists)
        else:
            if args.evaluation == 'threshold':
                results = Evaluation(args, mixed_valid_loader, model, step, 'final', writer, eval_abnorm_rate, dists,
                                     results)
            else:
                dists, results = Evaluation(args, mixed_valid_loader, model, step, 'final', writer, eval_abnorm_rate,
                                            dists, results)
            if args.only_infer:
                save_dir = args.log_dir + args.load_model
            else:
                save_dir = args.log_dir + args.timestamp + '_' + args.config
            with open(save_dir + f'/best_model_{args.evaluation}_valid_result.pkl', 'wb') as f:
                pickle.dump(results, f)
            ##inference
            inference(args, model, testloader, writer, 'final', step, test_abnorm_rate, dists)


if __name__ == "__main__":
    train(get_config())
