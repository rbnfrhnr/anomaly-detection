import pandas as pd
import numpy as np
from collections import *
from itertools import islice
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from multiprocessing import Process, Pool, Manager
import multiprocessing as mp
import logging
import threading
import pickle
import os
import time
import csv
import glob
from sklearn.preprocessing import normalize


def make_conn_dataframe(conn_lst, scenario='all'):
    dfs = []
    for i, c in enumerate(conn_lst):
        if not scenario == 'all':
            if c != scenario:
                continue
        for file in os.listdir('../ctu-13/CTU-13-Dataset/' + c):
            if file.endswith(".binetflow"):
                if i == 0:
                    df = pd.read_csv('../ctu-13/CTU-13-Dataset/' + c + '/' + file, delimiter=",")
                    # df["scenario"] = c
                    dfs.append(df)
                else:
                    df = pd.read_csv('../ctu-13/CTU-13-Dataset/' + c + '/' + file, delimiter=",", skiprows=0)
                    # df["scenario"] = c
                    dfs.append(df)
    df = pd.concat(dfs)
    return df


def preprocess_CTU(df, **kwargs):
    '''only connection'''
    df = df.reset_index(drop=True)
    service_lst = ["http", "dns", "icmp", "ntp", "ssl", "smtp"]
    df["service"] = df.Label.apply(
        lambda x: list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(service_lst)))[0] \
            if len(list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(service_lst)))) > 0 else '')
    class_lst = ["background", "normal", "botnet"]
    df["class"] = df.Label.apply(
        lambda x: list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(class_lst)))[0] \
            if len(list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(class_lst)))) > 0 else '')
    df = df[(df["class"] == "normal") | (df["class"] == "botnet")]
    df["min"] = df["StartTime"].apply(lambda x: int(x.split(' ')[1].split(":")[1]))
    df["hour"] = df["StartTime"].apply(lambda x: int(x.split(' ')[1].split(":")[0]))
    df["time_bin"] = df.apply(lambda x: int(x["hour"]) * 60 + int(x["min"]), axis=1)
    df = df.dropna()
    df_final = df.filter(["Proto", "State", "service", "class", "Dur", "TotPkts", "TotBytes", "time_bin"], axis=1)
    df_fill_oh = pd.get_dummies(df_final, columns=['Proto', 'State', 'service'])
    normalize_column = ["Dur", "TotPkts", "TotBytes", "time_bin"]
    for i in normalize_column:
        df_fill_oh[i] = (df_fill_oh[i] - df_fill_oh[i].min()) / (df_fill_oh[i].max() - df_fill_oh[i].min())
    # dis_df["class"] = df_fill_oh["class"]
    return df_fill_oh


def preprocess_stat_CTU(df, **kwargs):
    '''large dataset
       stat & connection info'''
    df = df.reset_index(drop=True)
    service_lst = ["http", "dns", "icmp", "ntp", "ssl", "smtp"]
    df["service"] = df.Label.apply(
        lambda x: list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(service_lst)))[0] \
            if len(list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(service_lst)))) > 0 else '')
    class_lst = ["background", "normal", "botnet"]
    df["class"] = df.Label.apply(
        lambda x: list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(class_lst)))[0] \
            if len(list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(class_lst)))) > 0 else '')
    # df = df[(df["class"]=="normal") | (df["class"]=="botnet")]
    df["min"] = df["StartTime"].apply(lambda x: int(x.split(' ')[1].split(":")[1]))
    df["hour"] = df["StartTime"].apply(lambda x: int(x.split(' ')[1].split(":")[0]))
    df["time_bin"] = df.apply(lambda x: int(x["hour"]) * 60 + int(x["min"]), axis=1)
    tb_lst = df.time_bin.unique()
    tb_dict = dict()
    for i in tb_lst:
        tb_dict[i] = (i - df.time_bin.min()) // 3
    df["time_chunk"] = df["time_bin"].apply(lambda x: tb_dict[x])
    # df = port_service_rate(df)

    grouped = df.filter(['Dur', 'TotPkts', 'TotBytes', 'Proto', 'service', "time_chunk", "SrcAddr"], axis=1).groupby(
        ["time_chunk", "SrcAddr"])
    result = grouped["Dur", "TotPkts", 'TotBytes'].aggregate([np.mean, np.std])
    df_merged = result.reset_index(level=[0, 1])
    df_merged.columns = [' '.join(col).strip() for col in df_merged.columns.values]
    df_merged = df_merged.drop_duplicates()
    df_merged = df_merged.fillna(0)

    df = df.filter(["Proto", "State", "service", "class", "SrcAddr", "time_chunk"], axis=1)
    df_final = df.merge(df_merged)  # use stat & connection info
    df_final = df_final.dropna()
    # df_final = df_final.filter(["Proto","State","service","class","Dur mean","Dur std", "TotPkts mean", \
    # "TotPkts std", "TotBytes mean","TotBytes std"], axis=1)
    df_final = df_final.filter(["Proto", "service", "class", "Dur mean", "Dur std", "TotPkts mean", \
                                "TotPkts std", "TotBytes mean", "TotBytes std"], axis=1)
    # df_fill_oh = pd.get_dummies(df_final, columns=['Proto','State','service'])
    df_fill_oh = pd.get_dummies(df_final, columns=['Proto', 'service'])
    normalize_column = ["Dur mean", "Dur std", "TotPkts mean", \
                        "TotPkts std", "TotBytes mean", "TotBytes std"]
    for i in normalize_column:
        df_fill_oh[i] = (df_fill_oh[i] - df_fill_oh[i].min()) / (df_fill_oh[i].max() - df_fill_oh[i].min())
    # dis_df["class"] = df_fill_oh["class"]

    df_fill_oh = df_fill_oh.rename(
        columns={"SrcAddr": "id.orig_h", "Dur mean": "duration mean", "Dur std": "duration std", \
                 "TotPkts mean": "orig_pkts mean", "TotPkts std": "orig_pkts std", \
                 "TotBytes mean": "orig_ip_bytes mean", "TotBytes std": "orig_ip_bytes std", \
                 "Dport": "id.resp_p", "DstAddr": "id.resp_h", "Sport": "id.orig_p"})

    return df_fill_oh


def preprocess_stat_2_CTU(df, period_len=60, rm_ntp=False):
    '''medium dataset
       stat(1 min) 
       use port, ip addr'''
    df['Date'] = pd.to_datetime(df.StartTime)
    df = df.sort_values(by='Date')
    df = df.reset_index(drop=True)
    if rm_ntp:
        service_lst = ['dhcp', 'dns', 'http', 'smtp', 'ssh', 'ssl']
    else:
        service_lst = ['dhcp', 'dns', 'http', 'ntp', 'smtp', 'ssh', 'ssl']
    df["service"] = df.Label.apply(
        lambda x: list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(service_lst)))[0] \
            if len(list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(service_lst)))) > 0 else '')
    # class_lst = ["background","normal", "botnet"]
    class_lst = ["normal", "botnet"]
    df["class"] = df.Label.apply(
        lambda x: list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(class_lst)))[0] \
            if len(list(set(map(str.lower, x.split("=")[1].split("-"))).intersection(set(class_lst)))) > 0 else '')
    df = df[(df["class"] == "normal") | (df["class"] == "botnet")]
    min_day = df["StartTime"].apply(lambda x: int(x.split(' ')[0].split('/')[2])).min()
    df["day"] = df["StartTime"].apply(lambda x: int(x.split(' ')[0].split('/')[2]) - min_day)
    df["min"] = df["StartTime"].apply(lambda x: int(x.split(' ')[1].split(":")[1]))
    df["hour"] = df["StartTime"].apply(lambda x: int(x.split(' ')[1].split(":")[0]))
    df["sec"] = df["StartTime"].apply(lambda x: int(x.split(' ')[1].split(":")[2].split(".")[0]))
    # df["time_bin"] = df.apply(lambda x:int(x["hour"]) * 60 +int(x["min"]), axis=1)
    df["time_bin"] = df.apply(
        lambda x: int(x["day"]) * 24 * 60 * 60 + int(x["hour"]) * 60 * 60 + int(x["min"]) * 60 + int(x["sec"]), axis=1)
    tb_lst = df.time_bin.unique()
    tb_dict = dict()
    for i in tb_lst:
        tb_dict[i] = (i - df.time_bin.min()) // period_len
    df["time_chunk"] = df["time_bin"].apply(lambda x: tb_dict[x])
    df_class = df.filter(["time_chunk", "SrcAddr", "class", "scenario"], axis=1)
    # df_class = df.filter(["time_chunk", "SrcAddr", "class"], axis=1)
    df_filtered = df.filter(['Dur', 'TotPkts', 'TotBytes', 'Proto', 'service', "time_chunk", \
                             "SrcAddr", "State", "Dport", "Sport", "DstAddr"], axis=1)
    df_dport = df_filtered.groupby(["time_chunk", "SrcAddr"])["Dport"].nunique().reset_index(level=[0, 1])
    df_dstaddr = df_filtered.groupby(["time_chunk", "SrcAddr"])["DstAddr"].nunique().reset_index(level=[0, 1])
    df_sport = df_filtered.groupby(["time_chunk", "SrcAddr"])["Sport"].nunique().reset_index(level=[0, 1])

    grouped = df_filtered.groupby(["time_chunk", "SrcAddr"])
    df_ipcnt = pd.DataFrame(grouped["SrcAddr"].agg('count')).rename(columns={'SrcAddr': "sripcnt"}).reset_index()
    result = grouped["Dur", "TotPkts", 'TotBytes'].aggregate([np.mean, np.std])
    result_service = df_filtered.groupby(["time_chunk", "SrcAddr", "service"]).size().unstack(fill_value=0).reset_index(
        level=[0, 1])
    result_service = result_service.drop([""], axis=1)
    result_proto = df_filtered.groupby(["time_chunk", "SrcAddr", "Proto"]).size().unstack(fill_value=0).reset_index(
        level=[0, 1])
    result_state = df_filtered.groupby(["time_chunk", "SrcAddr", "State"]).size().unstack(fill_value=0).reset_index(
        level=[0, 1])

    df_merged = result.reset_index(level=[0, 1])
    df_merged.columns = [' '.join(col).strip() for col in df_merged.columns.values]
    df_merged = df_merged.drop_duplicates()
    df_merged = df_merged.fillna(0)
    df_merged = df_merged.merge(df_class.drop_duplicates(subset=["SrcAddr", "time_chunk"]), how='left')
    df_final = result_service.merge(df_merged)
    df_final = df_final.merge(result_proto)
    df_final = df_final.merge(df_dport)
    df_final = df_final.merge(df_dstaddr)
    df_final = df_final.merge(df_sport)
    df_final = df_final.merge(df_ipcnt)
    # df_final = df_final.merge(result_state)
    df_fill_oh = df_final.dropna()

    normalize_column = ["Dur mean", "Dur std", "TotPkts mean", \
                        "TotPkts std", "TotBytes mean", "TotBytes std"] \
                       + ["Dport", "DstAddr", "Sport", "sripcnt"]
    normalize_column = [x for x in normalize_column if str(x) != 'nan']
    for i in normalize_column:
        df_fill_oh[i] = (df_fill_oh[i] - df_fill_oh[i].min()) / (df_fill_oh[i].max() - df_fill_oh[i].min())

    normalize_column = service_lst
    for c in normalize_column:
        if c not in df_fill_oh.columns:
            df_fill_oh[c] = 0
    df_fill_oh.loc[:, normalize_column] = \
        df_fill_oh.loc[:, normalize_column].div(df_fill_oh.loc[:, normalize_column].sum(axis=1), axis=0)
    normalize_column = ['icmp', 'tcp', 'udp', 'arp', 'igmp', 'rtp']
    for c in normalize_column:
        if c not in df_fill_oh.columns:
            df_fill_oh[c] = 0
    df_fill_oh.loc[:, normalize_column] = \
        df_fill_oh.loc[:, normalize_column].div(df_fill_oh.loc[:, normalize_column].sum(axis=1), axis=0)
    df_fill_oh = df_fill_oh.fillna(0)
    df_fill_oh = df_fill_oh.rename(
        columns={"SrcAddr": "id.orig_h", "Dur mean": "duration mean", "Dur std": "duration std", \
                 "TotPkts mean": "orig_pkts mean", "TotPkts std": "orig_pkts std", \
                 "TotBytes mean": "orig_ip_bytes mean", "TotBytes std": "orig_ip_bytes std", \
                 "Dport": "id.resp_p", "DstAddr": "id.resp_h", "Sport": "id.orig_p"})
    df_fill_oh = df_fill_oh.sort_index(axis=1)

    df_fill_oh = df_fill_oh.drop(columns=['id.orig_h', 'time_chunk'])

    return df_fill_oh


def preprocess_stat_3(df, period_len, rm_ntp):
    #     df = df.loc[~df["service"].isin(['mysql', 'imap','ftp','ftp-data'])]
    df["orig_bytes"] = df["orig_bytes"].apply(lambda x: 0 if x == '-' else x)
    df["resp_bytes"] = df["orig_bytes"].apply(lambda x: 0 if x == '-' else x)
    df["orig_bytes"] = df["orig_bytes"].astype('int64')
    df["resp_bytes"] = df["resp_bytes"].astype('int64')
    if rm_ntp:
        df = df.loc[~df["service"].isin(['ntp'])]
    df = df.sort_values(by='ts')
    df = df.reset_index(drop=True)
    df["StartTime"] = pd.to_datetime(df.ts, unit='s')
    min_day = df["StartTime"].apply(lambda x: x.day).min()
    df["day"] = df["StartTime"].apply(lambda x: x.day - min_day)
    df["min"] = df["StartTime"].apply(lambda x: x.minute)
    df["sec"] = df["StartTime"].apply(lambda x: x.second)
    df["hour"] = df["StartTime"].apply(lambda x: x.hour)
    # df["time_bin"] = df.apply(lambda x:int(x["hour"]) * 60 * 60 +int(x["min"]) * 60 + int(x["sec"]), axis=1)
    df["time_bin"] = df.apply(
        lambda x: int(x["day"]) * 24 * 60 * 60 + int(x["hour"]) * 60 * 60 + int(x["min"]) * 60 + int(x["sec"]), axis=1)
    tb_lst = df.time_bin.unique()
    tb_dict = dict()
    for i in tb_lst:
        tb_dict[i] = (i - df.time_bin.min()) // period_len
    df["time_chunk"] = df["time_bin"].apply(lambda x: tb_dict[x])
    # df = port_service_rate(df)
    df_class = df.filter(["time_chunk", "id.orig_h", "class"], axis=1)
    df.duration = df.apply(lambda x: 0 if x.duration == '-' else x.duration, axis=1)
    df.duration = df.duration.astype(float)
    df_filtered = df.filter(['duration', 'orig_pkts', 'orig_ip_bytes', 'orig_bytes', 'resp_bytes', \
                             'missed_bytes', 'resp_pkts', 'resp_ip_bytes', 'proto', 'service', \
                             "time_chunk", 'id.orig_h', 'id.resp_p', 'id.orig_p', 'id.resp_h'], axis=1)

    df_dport = df_filtered.groupby(["time_chunk", "id.orig_h"])["id.resp_p"].nunique().reset_index(level=[0, 1])
    df_dstaddr = df_filtered.groupby(["time_chunk", "id.orig_h"])["id.resp_h"].nunique().reset_index(level=[0, 1])
    df_sport = df_filtered.groupby(["time_chunk", "id.orig_h"])["id.orig_p"].nunique().reset_index(level=[0, 1])

    grouped = df_filtered.groupby(["time_chunk", "id.orig_h"])
    df_ipcnt = pd.DataFrame(grouped["id.orig_h"].agg('count')).rename(columns={'id.orig_h': "sripcnt"}).reset_index()
    result = grouped['duration', 'orig_pkts', 'orig_ip_bytes', \
                     'missed_bytes', 'resp_pkts', 'resp_ip_bytes', 'orig_bytes', 'resp_bytes'].aggregate(
        [np.mean, np.std])
    result_service = df_filtered.groupby(["time_chunk", "id.orig_h", "service"]).size().unstack(
        fill_value=0).reset_index(level=[0, 1])
    result_service = result_service.drop(["-"], axis=1)
    result_proto = df_filtered.groupby(["time_chunk", "id.orig_h", "proto"]).size().unstack(fill_value=0).reset_index(
        level=[0, 1])
    # result_state = df_filtered.groupby(["time_chunk","SrcAddr","State"]).size().unstack(fill_value=0).reset_index(level=[0,1])

    df_merged = result.reset_index(level=[0, 1])
    df_merged.columns = [' '.join(col).strip() for col in df_merged.columns.values]
    df_merged = df_merged.drop_duplicates()
    df_merged = df_merged.fillna(0)
    df_merged = df_merged.merge(df_class.drop_duplicates(subset=["id.orig_h", "time_chunk"]), how='left')
    df_final = result_service.merge(df_merged)
    df_final = df_final.merge(result_proto)
    df_final = df_final.merge(df_dport)
    df_final = df_final.merge(df_dstaddr)
    df_final = df_final.merge(df_sport)
    df_final = df_final.merge(df_ipcnt)
    # df_final = df_final.merge(result_state)
    df_fill_oh = df_final.dropna()
    normalize_column = ["duration mean", "duration std", "orig_pkts mean", \
                        "orig_pkts std", "orig_ip_bytes mean", "orig_ip_bytes std", \
                        "orig_bytes mean", "orig_bytes std", "resp_bytes mean", "resp_bytes std", \
                        "missed_bytes mean", "missed_bytes std", \
                        "resp_pkts mean", "resp_pkts std", "resp_ip_bytes mean", "resp_ip_bytes std"] \
                       + ["id.resp_p", "id.resp_h", "id.orig_p", "sripcnt"]
    normalize_column = [x for x in normalize_column if str(x) != 'nan']
    for i in normalize_column:
        df_fill_oh[i] = (df_fill_oh[i] - df_fill_oh[i].min()) / (df_fill_oh[i].max() - df_fill_oh[i].min())

    normalize_column = list(df.service.unique())
    normalize_column.remove("-")
    normalize_column = [x for x in normalize_column if str(x) != 'nan']
    df_fill_oh.loc[:, normalize_column] = \
        df_fill_oh.loc[:, normalize_column].div(df_fill_oh.loc[:, normalize_column].sum(axis=1), axis=0)
    normalize_column = ['icmp', 'tcp', 'udp', 'arp', 'igmp', 'rtp']
    for c in normalize_column:
        if c not in df_fill_oh.columns:
            df_fill_oh[c] = 0
    df_fill_oh.loc[:, normalize_column] = \
        df_fill_oh.loc[:, normalize_column].div(df_fill_oh.loc[:, normalize_column].sum(axis=1), axis=0)
    df_fill_oh = df_fill_oh.fillna(0)
    df_fill_oh = df_fill_oh.sort_index(axis=1)
    return df_fill_oh


if __name__ == "__main__":
    df = make_conn_dataframe(["1", "2", "6", "8", "9"])
    # dis_df = preprocess_stat(df)
    dis_df = preprocess_stat_2_CTU(df, 60, False)
    dis_df.to_csv("test.csv")
