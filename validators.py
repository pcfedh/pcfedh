from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import parameters_to_vector
from sklearn.metrics import mean_squared_error , mean_absolute_error,mean_absolute_percentage_error
import FLAG
import datetime
import torch.nn.functional as F
from tqdm import tqdm


def _eval_with_pooling(x):
    x = F.max_pool1d(
            x.transpose(1, 2),
            kernel_size = x.size(1),
        ).transpose(1, 2).squeeze(1)
    
    return x

class ValidatorConfig:
    def __init__(self, num_class=10, num_epochs=200, verbose=True, test_ds_batch_size=50, do_client_model_compare=False,
                 do_importance_estimation=True, do_client_eval=True, do_cluster_eval=True, client_eval_idx_vec=None):
        if client_eval_idx_vec is None:
            client_eval_idx_vec = [0, -1]
        self.num_class = num_class
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.test_ds_batch_size = test_ds_batch_size
        self.do_client_model_compare = do_client_model_compare
        self.do_importance_estimation = do_importance_estimation
        self.do_client_eval = do_client_eval
        self.do_cluster_eval = do_cluster_eval
        self.client_eval_idx_vec = client_eval_idx_vec


class ValidatorRegression:
    def __init__(self, test_ds_vec, config):
        self.config = config
        self.verbose = config.verbose
        self.test_loader_vec = []
        for ds in test_ds_vec:
            self.test_loader_vec.append(DataLoader(ds, batch_size=config.test_ds_batch_size))

    def validate(self, client_vec, cluster_vec, t):
        with torch.no_grad():
            validation_dict = {'round': t,
                               'client_model_div_mat': None,
                               'importance_estimation': {},
                               'client_eval': {},
                               'cluster_eval': {}, }
            info = '-' * 30 + ' Round {} '.format(t) + '-' * 30 + '\n'
            mse_loss = nn.MSELoss(reduction='sum')

            if self.config.do_client_model_compare:
                l2_mat = np.zeros((len(client_vec), len(client_vec)))
                for i in range(len(client_vec)):
                    for j in range(i + 1, len(client_vec)):
                        l2 = mse_loss(parameters_to_vector(client_vec[i].model.parameters()),
                                      parameters_to_vector(client_vec[j].model.parameters())).item()
                        l2_mat[i][j] = l2
                        l2_mat[j][i] = l2
                np.set_printoptions(precision=3)
                info += 'l2 mat = \n{}'.format(l2_mat)
                validation_dict['client_model_div_mat'] = l2_mat

            if self.config.do_importance_estimation:
                for client in client_vec:
                    # print(client.importance_estimated)
                    info += 'client {} importance estimation = {}\n'.format(client.ID, client.importance_estimated.tolist())
                    validation_dict['importance_estimation']['client_' + str(client.ID)] = client.importance_estimated.tolist()

            if self.config.do_client_eval:
                client_set = [client_vec[i] for i in self.config.client_eval_idx_vec]
                if t == self.config.num_epochs-1:
                    client_set = client_vec
                for i, client in enumerate(client_set):
                    client.model.eval()
                    info += '*' * 10 + ' Client {} '.format(i) + '*' * 10 + '\n'
                    validation_dict['client_eval']['client_' + str(client.ID)] = {}
                    loader = client.loader
                    loss = 0.

                    for x, y in loader:
                        x = x.to(FLAG.device)
                        y = y.to(FLAG.device)
                        out = client.model(x)
                        loss += mse_loss(y, out).item()
                        label_list += y.cpu().numpy().tolist()
                        preds_list += out.cpu().numpy().tolist()
                        input_list += x[:, -1].cpu().numpy().tolist()
                    average_loss = loss / len(loader) / self.config.test_ds_batch_size
                    label_list = [100*k for k in label_list]
                    preds_list = [100*k for k in preds_list]
                    input_list = [100*k for k in input_list]
                    mse = mean_squared_error(label_list, preds_list)
                    mae = mean_absolute_error(label_list, preds_list)
                    mape = mean_absolute_percentage_error(label_list, preds_list)
                    
                    mse_na = mean_squared_error(label_list, input_list)
                    mae_na  = mean_absolute_error(label_list, input_list)
                    mape_na  = mean_absolute_percentage_error(label_list, input_list)
                    info += 'Client {0:d} has average ' \
                            'mse = {2:.3f}, mae = {3:.3f}, mape = {4:.3f}\n mse_na = {5:.3f}, mae_na = {6:.3f}, mape_na = {7:.3f}\n'.format(i, j, mse, mae, mape, mse_na, mae_na, mape_na)
                    validation_dict['client_eval']['client_' + str(client.ID)]['average_mse'] = mse
                    validation_dict['client_eval']['client_' + str(client.ID)]['average_mae'] = mae
                    validation_dict['client_eval']['client_' + str(client.ID)]['average_mape'] = mape
                    validation_dict['client_eval']['client_' + str(client.ID)]['ds_num'] = client.num_samples
                    validation_dict['client_eval']['client_' + str(client.ID)]['tsds_num'] = client.tsds_num
                    info += '-' * 30 + ' Round {} '.format(t) + '-' * 30 + '\n'

            if self.verbose:
                print(info)
            # print(validation_dict)
            return validation_dict


class ValidatorRegressionPersonal:
    def __init__(self, config):
        self.config = config
        self.verbose = config.verbose

    def validate(self, client_vec, cluster_vec, t):
        with torch.no_grad():
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            validation_dict = {'round': t,
                               'client_model_div_mat': None,
                               'importance_estimation': {},
                               'client_eval': {},
                               'cluster_eval': {}, 
                               'validataion_time': formatted_time}
            info = '-' * 30 + ' Round {} '.format(t) + '-' * 30 + '\n'
            mse_loss = nn.MSELoss(reduction='sum')

            if self.config.do_client_model_compare:
                l2_mat = np.zeros((len(client_vec), len(client_vec)))
                for i in range(len(client_vec)):
                    for j in range(i + 1, len(client_vec)):
                        l2 = mse_loss(parameters_to_vector(client_vec[i].model.parameters()),
                                      parameters_to_vector(client_vec[j].model.parameters())).item()
                        l2_mat[i][j] = l2
                        l2_mat[j][i] = l2
                np.set_printoptions(precision=3)
                info += 'l2 mat = \n{}'.format(l2_mat)
                validation_dict['client_model_div_mat'] = l2_mat

            if self.config.do_importance_estimation:
                for client in client_vec:
                    # print(client.importance_estimated)
                    info += 'client {} importance estimation = {}\n'.format(client.ID, client.importance_estimated.tolist())
                    validation_dict['importance_estimation']['client_' + str(client.ID)] = client.importance_estimated.tolist()

            if self.config.do_client_eval:
                client_set = [client_vec[i] for i in self.config.client_eval_idx_vec]
                if t == self.config.num_epochs-1:
                    client_set = client_vec
                for i, client in enumerate(client_set):
                    client.model.eval()
                    # client.pclmodel.eval()
                    client.model.to(FLAG.device)
                    # client.pclmodel.to(FLAG.device)
                    info += '*' * 10 + ' Client {} '.format(i) + '*' * 10 + '\n'
                    validation_dict['client_eval']['client_' + str(client.ID)] = {}
                    loader = client.testloader
                    # loss = 0.
                    label_list, preds_list,input_list = [], [],[]
                    if len(loader)==0:
                        continue
                    for x, y in loader:
                        x = x.to(FLAG.device)
                        y = y.to(FLAG.device)


                        out = client.model(x)

                        label_list += y.cpu().numpy().tolist()
                        preds_list += out.cpu().numpy().tolist()
                        # print(x.shape)
                        input_list += x[:, -1].reshape(x.shape[0]).cpu().numpy().tolist()

                    label_list = [100*k for k in label_list]
                    preds_list = [100*k for k in preds_list]
                    input_list = [100*k for k in input_list]
                    mse = mean_squared_error(label_list, preds_list)
                    mae = mean_absolute_error(label_list, preds_list)
                    mape = mean_absolute_percentage_error(label_list, preds_list)
                    
                    mse_na = mean_squared_error(label_list, input_list)
                    mae_na  = mean_absolute_error(label_list, input_list)
                    mape_na  = mean_absolute_percentage_error(label_list, input_list)
                    info += 'Client {0:d} has average ' \
                            'mse = {1:.3f}, mae = {2:.3f}, mape = {3:.3f}\n mse_na = {4:.3f}, mae_na = {5:.3f}, mape_na = {6:.3f}\n'.format(i, mse, mae, mape, mse_na, mae_na, mape_na)
                    validation_dict['client_eval']['client_' + str(client.ID)]['average_mse'] = mse
                    validation_dict['client_eval']['client_' + str(client.ID)]['average_mae'] = mae
                    validation_dict['client_eval']['client_' + str(client.ID)]['average_mape'] = mape
                    validation_dict['client_eval']['client_' + str(client.ID)]['preds_list'] = preds_list
                    validation_dict['client_eval']['client_' + str(client.ID)]['label_list'] = label_list
                    validation_dict['client_eval']['client_' + str(client.ID)]['ds_num'] = client.num_samples
                    validation_dict['client_eval']['client_' + str(client.ID)]['tsds_num'] = client.tsds_num
                    info += '-' * 30 + ' Round {} '.format(t) + '-' * 30 + '\n'

            
            if self.verbose:
                print(info)
            # print(validation_dict)
            return validation_dict


