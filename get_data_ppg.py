import pandas as pd 
import glob
import pandas as pd
import numpy as np
import datetime
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import random
import os
import matplotlib.pyplot as plt


def create_input_sequences(input_data, tw, pred_len):
    """
    :param input_data:
    :param tw:
    :return:
    """
    data_seq = 0
    label_seq = 0
    L = len(input_data)
    # print('L',L,L-tw-pred_len+1)

    # input_size = input_data.shape[1]
    for i in range(L - tw - pred_len + 1):
        train_seq = input_data[i:i + tw].unsqueeze(0)
        train_label = input_data[i + tw + pred_len - 1:i + tw + pred_len].unsqueeze(0)

        if not torch.is_tensor(data_seq):
            data_seq = train_seq
            label_seq = train_label
        else:
            data_seq = torch.cat((data_seq, train_seq), 0)
            label_seq = torch.cat((label_seq, train_label), 0)

    return data_seq, label_seq.squeeze()

def get_HR_data():
        # Path of the folder containing the CSV files
    folder_path = './data/PPG'

    # Get a list of file names in the folder
    file_names = glob.glob(folder_path + '/*.csv')

    # Loop over the file names and read each CSV file into a DataFrame
    data_frames = []
    for file_name in file_names:
        df = pd.read_csv(file_name)
        print('Read file: ' + file_name)
        print(df.shape)
        data_frames.append(df)

    return data_frames


def get_supervise(df,time_step,pred_len):

    df = df.iloc[:, 0].values
    all_data = df
    all_data_normalized = torch.FloatTensor(all_data)
    all_data_normalized, all_label_normalized = create_input_sequences(all_data_normalized, time_step,pred_len)
    data_tensor = torch.cat([all_label_normalized.unsqueeze(1), all_data_normalized.squeeze()], dim=1)

    return data_tensor


# get each one's HR data and use get_supervise_orignal to get the supervised data, splite each one's data into train and test
def get_split_data_clients_all(time_step,pred_len,train_rate=0.8):

    client_data = get_HR_data()
    client_data_supervise = []

    for i in range(len(client_data)):
        print('client ',i,' is processing')
        print(client_data[i].shape)
        one_supervise = get_supervise(client_data[i],time_step,pred_len)
        client_data_supervise.append(one_supervise)


    client_data_supervise = [client_data_supervise[i] for i in range(len(client_data_supervise)) if torch.is_tensor(client_data_supervise[i])]
    train_data_client = [0]*len(client_data_supervise)

    test_data_client = [0]*len(client_data_supervise)


    dict_client = {}
    start_client = 0
    for i in range(len(client_data_supervise)):
        test_split = int(len(client_data_supervise[i])*train_rate)
        client_data_supervise[i] = client_data_supervise[i]/100
        train_data_client[i],test_data_client[i] = client_data_supervise[i][:test_split],client_data_supervise[i][test_split:]
        train_data_client[i] = train_data_client[i][:10*(len(train_data_client[i])//10)]
        dict_client[i] = [j for j in range(start_client,start_client+train_data_client[i].shape[0])]
        start_client = start_client+train_data_client[i].shape[0]
        print(len(dict_client[i]))

    train_data = torch.cat(train_data_client, dim=0)[:,1:]
    train_label = torch.cat(train_data_client, dim=0)[:,0]
    
    test_data, test_label = torch.cat(test_data_client, dim=0)[:,1:],torch.cat(test_data_client, dim=0)[:,0]
    print(train_data[:10])

    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)
    return train_data_client, test_data_client,dict_client

    return train_data_client, test_data_client,dict_client

if __name__ == '__main__':
    print('reading data...')
    get_split_data_clients_all(100,1,1,0.8)