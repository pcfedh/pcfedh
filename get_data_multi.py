import pandas as pd
import torch
import os

def optimize_get_supervise(data, time_step, pred_len):

    length = len(data) - time_step - pred_len
    if length <= 0:
        return None

    supervise_data = torch.zeros((length, time_step + pred_len))
    for i in range(length):
        supervise_data[i, :] = torch.tensor(data[i:(i+time_step+pred_len)].values)
    supervise_data = supervise_data[~(supervise_data == 0).any(axis=1)]
    if len(supervise_data) == 0:
        return None
    return supervise_data

def process_single_driver_data(file_path, time_step, pred_len, train_rate):

    data = pd.read_csv(file_path)
    filtered_data = data[~data['Drive'].isin([0,1, 2,8])]

    if filtered_data['Heart.Rate'].isnull().all():
        return None, None

    grouped = filtered_data.groupby('Drive')
    train_data_list = []
    test_data_list = []

    for _, group in grouped:
        hr_data = group['Heart.Rate'].dropna()
        if len(hr_data) < time_step + pred_len:
            continue
        
        supervise_data = optimize_get_supervise(hr_data, time_step, pred_len)
        if supervise_data is None:
            continue
        supervise_data = supervise_data/100

        test_split = int(len(supervise_data) * train_rate)
        train_data = supervise_data[:test_split]
        test_data = supervise_data[test_split:]
        if len(test_data) < 1:
            continue
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    if train_data_list and test_data_list:
        train_data_final = torch.cat(train_data_list, dim=0)
        test_data_final = torch.cat(test_data_list, dim=0)
        return train_data_final, test_data_final
    else:
        return None, None

def get_split_data_clients_all_multi(time_step, pred_len,train_rate=0.8):

    train_data_clients = []
    test_data_clients = []
    folder_path = './data/multi_HR'
    i=0
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        train_data, test_data = process_single_driver_data(file_path, time_step, pred_len, train_rate)

        if train_data is not None and test_data is not None:
            print(i,file_path)
            i+=1
            if len(train_data) < 100:
                continue
            train_data_clients.append(train_data)
            test_data_clients.append(test_data)

    return train_data_clients, test_data_clients


def get_split_data_clients_all_type(time_step, pred_len, folder_path, train_rate=0.8):
    train_data_clients = []
    test_data_clients = []
    folder_path = './data/multi_HR'
    i=0
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        train_data, test_data = process_single_driver_data_type(file_path, time_step, pred_len, train_rate)

        if train_data is not None and test_data is not None:
            print(i,file_path)
            i+=1
            if len(train_data) < 100:
                continue
            train_data_clients.append(train_data)
            test_data_clients.append(test_data)
    
    return train_data_clients, test_data_clients


