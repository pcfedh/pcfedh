import numpy as np
from get_data_ppg import get_split_data_clients_all
from get_data_multi import get_split_data_clients_all_multi
from clients import Client_FedHeart,Client_FedHeart_Predction
from torch.utils.data import TensorDataset


def create_clients_ppg_fedheart_pretrain(client_solver,cluster_num=4,pred_len=1):

    num_clients = 15
    
    time_step = 150
    train_rate = 0.8

    train_data_client, test_data_client,dict_client = get_split_data_clients_all(time_step,pred_len,train_rate)
    client_vec = []

    for i in range(num_clients):
      
        ds_client = TensorDataset(train_data_client[i][:,1:].unsqueeze(-1),train_data_client[i][:,0])
        tsds_client = TensorDataset(test_data_client[i][:,1:].unsqueeze(-1),test_data_client[i][:,0])

        num_samples = len(ds_client)
        tag = 'ppg data. total {} data points.'.format(
            num_samples)
        client = Client_FedHeart(ID=i, ds=ds_client, pcl_ds=ds_client, ts_ds=tsds_client, solver=client_solver, tag=tag)
        client_vec.append(client)
        
    param_dict = {
        'num_clusters': cluster_num,
        'num_clients': num_clients,
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag

    return client_vec , param_dict


def create_clients_ppg_fedheart(client_solver,cluster_num=2,pred_len=1):

    num_clients = 15
    time_step = 150
    train_rate = 0.8

    train_data_client, test_data_client,dict_client = get_split_data_clients_all(time_step,pred_len,train_rate)
    client_vec = []
    for i in range(num_clients):
      
        ds_client = TensorDataset(train_data_client[i][:,1:].unsqueeze(-1),train_data_client[i][:,0])
        tsds_client = TensorDataset(test_data_client[i][:,1:].unsqueeze(-1),test_data_client[i][:,0])
        num_samples = len(ds_client)
        tag = 'homony data. total {} data points.'.format(
            num_samples)
        client = Client_FedHeart_Predction(ID=i, ds=ds_client, pcl_ds=ds_client, ts_ds=tsds_client, solver=client_solver, tag=tag)
        client_vec.append(client)
        
    param_dict = {
        'num_clusters': cluster_num,
        'num_clients': num_clients,
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag


    return client_vec , param_dict


def create_clients_multi_fedheart_pretrain(client_solver,cluster_num=7,pred_len=1):

    
    pred_len = pred_len
    time_step = 150
    train_rate = 0.8

    train_data_client, test_data_client = get_split_data_clients_all_multi(time_step,pred_len,train_rate)
    num_clients = len(train_data_client)
    client_vec = []
    test_ds = []
    # mixture_vec = []
    for i in range(num_clients):
      
        ds_client = TensorDataset(train_data_client[i][:,:-pred_len].unsqueeze(-1),train_data_client[i][:,-pred_len:].squeeze(-1))
        tsds_client = TensorDataset(test_data_client[i][:,:-pred_len].unsqueeze(-1),test_data_client[i][:,-pred_len:].squeeze(-1))
        # test_ds.append(tsds_client)
        num_samples = len(ds_client)
        tag = 'multi data. total {} data points.'.format(
            num_samples)
        client = Client_FedHeart(ID=i, ds=ds_client, pcl_ds=ds_client, ts_ds=tsds_client, solver=client_solver, tag=tag)
        client_vec.append(client)
        
    param_dict = {
        'num_clusters': cluster_num,
        'num_clients': num_clients,
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag


    return client_vec , param_dict


def create_clients_multi_fedheart(client_solver,cluster_num=7,pred_len=1):

    
    pred_len = pred_len
    time_step = 150
    train_rate = 0.8
    train_data_client, test_data_client = get_split_data_clients_all_multi(time_step,pred_len)
    num_clients = len(train_data_client)
    # print(train_data_client[0].shape)
    client_vec = []
    test_ds = []
    # mixture_vec = []
    for i in range(num_clients):
        # print(train_data_client[i].shape,test_data_client[i].shape)
        ds_client = TensorDataset(train_data_client[i][:,:-pred_len].unsqueeze(-1),train_data_client[i][:,-pred_len:].squeeze(-1))
        tsds_client = TensorDataset(test_data_client[i][:,:-pred_len].unsqueeze(-1),test_data_client[i][:,-pred_len:].squeeze(-1))
        # test_ds.append(tsds_client)
        num_samples = len(ds_client)
        tag = 'multi data. total {} data points.'.format(
            num_samples)
        client = Client_FedHeart_Predction(ID=i, ds=ds_client, pcl_ds=ds_client, ts_ds=tsds_client, solver=client_solver, tag=tag)
        client_vec.append(client)
        
    param_dict = {
        'num_clusters': cluster_num,
        'num_clients': num_clients,
        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag


    return client_vec , param_dict


def create_clients_multi_all(client_solver,client_type,cluster_num=7,pred_len=1):


    time_step = 150
    train_rate = 0.8

    train_data_client, test_data_client = get_split_data_clients_all_multi(time_step,pred_len)
    num_clients = len(train_data_client)
    client_vec = []
    test_ds = []
    # mixture_vec = []
    for i in range(num_clients):
        print(i)
        ds_client = TensorDataset(train_data_client[i][:,:-pred_len].unsqueeze(-1),train_data_client[i][:,-pred_len:].squeeze(-1))
        tsds_client = TensorDataset(test_data_client[i][:,:-pred_len].unsqueeze(-1),test_data_client[i][:,-pred_len:].squeeze(-1))
        # test_ds.append(tsds_client)
        num_samples = len(ds_client)
        tag = 'multi data. total {} data points.'.format(
            num_samples)
        if client_type=='fedsoft':
            client = Client(ID=i, ds=ds_client, ts_ds=tsds_client, solver=client_solver, tag=tag)
        elif client_type=='fedavg':
            client = FedAvgClient(ID=i, ds=ds_client, ts_ds=tsds_client, solver=client_solver, tag=tag)
        elif client_type=='fedala':
            client = FedALAClient(ID=i, ds=ds_client, ts_ds=tsds_client, solver=client_solver, tag=tag)

        client_vec.append(client)
   
    param_dict = {
        'num_clusters': cluster_num,
        'num_clients': num_clients,

        'client_tags': {},
        'mixture': {}
    }
    for k, client in enumerate(client_vec):
        param_dict['client_tags']['client_' + str(client.ID)] = client.tag
        # param_dict['mixture']['client_' + str(client.ID)] = mixture_vec[k]

    return client_vec , param_dict

