import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from servers import Server_FedHeart,Sever_Predict
from validators import ValidatorConfig, ValidatorRegressionPersonal
from models import LSTM, TSEncoder
from solvers import ServerSolver,HomonyClientSolver
from utils import init_exp, logger
from clients_preparation import create_clients_ppg_fedheart_pretrain, create_clients_ppg_fedheart, create_clients_multi_fedheart_pretrain,create_clients_multi_fedheart
import torch
import random
import numpy as np
import sys

def run_fedheart_pretrain_ppg(exp_id,cluster_num,reg_weight,pclbatch_size,pcllr,alpha):
    num_epochs = 100
    model_fn = LSTM
    pclmodel_fn = TSEncoder
    server_solver = ServerSolver(estimation_interval=1,
                                 do_selection=False,
                                 selection_size=15
                                 )
    client_solver = HomonyClientSolver(reg_weight=reg_weight,pclbatch_size=pclbatch_size,pcllr=pcllr,alpha=alpha)
    client_vec, preparation_dict = create_clients_ppg_fedheart_pretrain(client_solver,cluster_num=cluster_num)

    validator_config = ValidatorConfig(num_epochs=num_epochs,
                                       do_cluster_eval=True, do_client_eval=True, do_importance_estimation=True,
                                       client_eval_idx_vec=range(preparation_dict['num_clients']))

    logger.log_server_solver(exp_id, server_solver.to_json_dict())
    logger.log_client_solver(exp_id, client_solver.to_json_dict())
    logger.log_client_preparation(exp_id, preparation_dict)
    logger.log_model_description(exp_id, model_fn)
    logger.log_pclmodel_description(exp_id, pclmodel_fn)

    server = Server_FedHeart(model_fn=model_fn, pclmodel_fn=pclmodel_fn, client_vec=client_vec, num_clusters=preparation_dict['num_clusters'],
                    server_solver=server_solver, validator=ValidatorRegressionPersonal(validator_config),
                    exp_id=exp_id)
    server.run(num_global_epochs=num_epochs)

def run_fedheart_pretrain_multi(exp_id,cluster_num,reg_weight,pclbatch_size,pcllr,alpha):
    num_epochs = 100
    model_fn = LSTM
    pclmodel_fn = TSEncoder
    server_solver = ServerSolver(estimation_interval=1,
                                 do_selection=False,
                                 selection_size=63
                                 )
    client_solver = HomonyClientSolver(reg_weight=reg_weight,pclbatch_size=pclbatch_size,pcllr=pcllr,alpha=alpha)

    client_vec, preparation_dict = create_clients_multi_fedheart_pretrain(client_solver,cluster_num=cluster_num)
    validator_config = ValidatorConfig(num_epochs=num_epochs,
                                       do_cluster_eval=True, do_client_eval=True, do_importance_estimation=True,
                                       client_eval_idx_vec=range(preparation_dict['num_clients']))

    logger.log_server_solver(exp_id, server_solver.to_json_dict())
    logger.log_client_solver(exp_id, client_solver.to_json_dict())
    logger.log_client_preparation(exp_id, preparation_dict)
    logger.log_model_description(exp_id, model_fn)
    logger.log_pclmodel_description(exp_id, pclmodel_fn)

    server = Server_FedHeart(model_fn=model_fn, pclmodel_fn=pclmodel_fn, client_vec=client_vec, num_clusters=preparation_dict['num_clusters'],
                    server_solver=server_solver, validator=ValidatorRegressionPersonal(validator_config),
                    exp_id=exp_id)
    server.run(num_global_epochs=num_epochs)

def run_fedheart_pred_ppg(exp_id,cluster_num,reg_weight,pred_len,pretrain_epoch):
    num_epochs = 500
    model_fn = LSTM
    pclmodel_fn = TSEncoder
    server_solver = ServerSolver(estimation_interval=1,
                                 do_selection=True,
                                 selection_size=5
                                 )
    client_solver = HomonyClientSolver(reg_weight=reg_weight)
    
    client_vec, preparation_dict = create_clients_ppg_fedheart(client_solver,cluster_num=cluster_num,pred_len=pred_len)
    validator_config = ValidatorConfig(num_epochs=num_epochs,
                                       do_cluster_eval=True, do_client_eval=True, do_importance_estimation=True,
                                       client_eval_idx_vec=range(preparation_dict['num_clients']))

    logger.log_server_solver(exp_id, server_solver.to_json_dict())
    logger.log_client_solver(exp_id, client_solver.to_json_dict())
    logger.log_client_preparation(exp_id, preparation_dict)
    logger.log_model_description(exp_id, model_fn)
    logger.log_pclmodel_description(exp_id, pclmodel_fn)
    pretrain_epoch = pretrain_epoch
    server = Sever_Predict(model_fn=model_fn, pclmodel_fn=pclmodel_fn, client_vec=client_vec, num_clusters=preparation_dict['num_clusters'],
                    server_solver=server_solver, validator=ValidatorRegressionPersonal(validator_config),pretrain_epoch=pretrain_epoch,
                    exp_id=exp_id)
    server.run(num_global_epochs=num_epochs)

def run_fedheart_pred_multi(exp_id,cluster_num,reg_weight,pred_len, pretrain_epoch):
    num_epochs = 500
    model_fn = LSTM
    pclmodel_fn = TSEncoder
    server_solver = ServerSolver(estimation_interval=1,
                                 do_selection=True,
                                 selection_size=20
                                 )
    client_solver = HomonyClientSolver(reg_weight=reg_weight)
    
    client_vec, preparation_dict = create_clients_multi_fedheart(client_solver,cluster_num=cluster_num,pred_len=pred_len)
    validator_config = ValidatorConfig(num_epochs=num_epochs,
                                       do_cluster_eval=True, do_client_eval=True, do_importance_estimation=True,
                                       client_eval_idx_vec=range(preparation_dict['num_clients']))

    logger.log_server_solver(exp_id, server_solver.to_json_dict())
    logger.log_client_solver(exp_id, client_solver.to_json_dict())
    logger.log_client_preparation(exp_id, preparation_dict)
    logger.log_model_description(exp_id, model_fn)
    logger.log_pclmodel_description(exp_id, pclmodel_fn)
    pretrain_epoch = pretrain_epoch
    server = Sever_Predict(model_fn=model_fn, pclmodel_fn=pclmodel_fn, client_vec=client_vec, num_clusters=preparation_dict['num_clusters'],
                    server_solver=server_solver, validator=ValidatorRegressionPersonal(validator_config),pretrain_epoch=pretrain_epoch,
                    exp_id=exp_id)
    server.run(num_global_epochs=num_epochs)

def main(dataset,method):


    print('dataset: ',dataset)
    print('method: ',method)

    if dataset=='ppg':
        if method == 'pretrained':
            cluster_num = 4
            reg_weight = 0.001
            pclbatch_size=16
            pcllr=1e-3
            alpha=1
            exp_id = 'ppg'
            init_exp(exp_id)
            logger.log_exp_info(exp_id,
                                description='experiment description')

            run_fedheart_pretrain_ppg(exp_id,cluster_num,reg_weight,pclbatch_size,pcllr,alpha)

        if method == 'predict':
            pred_len = 1
            cluster_num = 4

            exp_id = 'ppg'
            init_exp(exp_id)
            logger.log_exp_info(exp_id,
                                description='experiment description')
            reg_weight = 0.001
            pretrain_epoch = 9
            run_fedheart_pred_ppg(exp_id, cluster_num, reg_weight, pred_len, pretrain_epoch)

    elif dataset=='distracted':
        if method =='pretrained':
            cluster_num = 5
            reg_weight = 0.001
            pclbatch_size=16
            pcllr=1e-2
            alpha=1
            exp_id = 'distracted'
            init_exp(exp_id)
            logger.log_exp_info(exp_id,
                                description='experiment description')

            run_fedheart_pretrain_multi(exp_id,cluster_num,reg_weight,pclbatch_size,pcllr,alpha)
        if method=='predict':
            pred_len = 1
            cluster_num = 5
            exp_id = 'distracted'
            init_exp(exp_id)
            logger.log_exp_info(exp_id,
                                description='experiment description')
            reg_weight = 0.001
            pretrain_epoch = 9
            run_fedheart_pred_multi(exp_id,cluster_num,reg_weight,pred_len,pretrain_epoch)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
if __name__ == '__main__':

    set_seed(0)
    dataset = sys.argv[1]
    method = sys.argv[2]
    main(dataset,method)
