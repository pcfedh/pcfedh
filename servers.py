import copy

import torch
import numpy as np
import pcl.builder
from FLAG import device, validation_data_flush_interval
from utils import logger
# from multiprocessing import Pool
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn.functional as F

class Server_FedHeart:
    def __init__(self, model_fn, pclmodel_fn, client_vec, num_clusters, validator, server_solver, exp_id='default_exp'):
        self.model_fn = model_fn
        self.pclmodel_fn = pclmodel_fn
        self.client_vec = client_vec
        self.validator = validator
        self.server_solver = server_solver
        self.exp_id = exp_id
        self.client_init_model = model_fn()
        self.client_init_pclmodel = pclmodel_fn()

        self.num_clients = len(client_vec)
        self.num_clusters = num_clusters
        self.cluster_vec = []
        for _ in range(self.num_clusters):
            self.cluster_vec.append(model_fn())
        for i in range(self.num_clusters):
            self.cluster_vec[i] = self.cluster_vec[i].to(device)

        self.importance_weights_matrix = []
        self._zero_weights = None
        self._zero_pclweights = None
        self.prototypes = None
        self.density = None

        self.start_epoch = 0
        try:
            with open('./log/{}/round.txt'.format(self.exp_id), 'r') as f:
                self.start_epoch = int(f.readline().strip()) + 1
            self.load_saved_data(self.start_epoch - 1)
            
            print(f"Resuming from epoch {self.start_epoch}")
        except:
            print("Starting from scratch or error in reading checkpoint.")

        self.num_samples_vec = np.array([])
        for client in client_vec:
            client.attach_to_server(server=self)
            self.num_samples_vec = np.append(self.num_samples_vec, client.num_samples)
        self.num_samples = np.sum(self.num_samples_vec)
        self.number_samples_fraction_vec = self.num_samples_vec / self.num_samples


        

    def load_saved_data(self, t):
        self.prototypes = torch.load('./log/{}/next_prototypes_{}.pth'.format(self.exp_id, t))
        self.density = torch.load('./log/{}/next_density_{}.pth'.format(self.exp_id, t))
        pclweights = torch.load('./log/{}/next_pclweights_{}.pth'.format(self.exp_id, t))
        self.client_init_pclmodel.load_state_dict(state_dict=pclweights)
        # print("Loaded prototypes, density and pclweights from epoch {}".format(t))
        # print("Prototypes: ", self.prototypes)
        # print("Density: ", self.density)
        # print("PCL Weights: ", pclweights)
        return pclweights


    def get_cluster_model(self, idx):
        return copy.deepcopy(self.cluster_vec[idx].state_dict())

    def generate_zero_weights(self):
        if self._zero_weights is None:
            self._zero_weights = {}
            for key, val in self.client_init_model.state_dict().items():
                self._zero_weights[key] = torch.zeros(size=val.shape, dtype=torch.float32)
        return copy.deepcopy(self._zero_weights)
    
    def generate_zero_pclweights(self):
        if self._zero_pclweights is None:
            self._zero_pclweights = {}
            for key, val in self.client_init_pclmodel.state_dict().items():
                self._zero_pclweights[key] = torch.zeros(size=val.shape, dtype=torch.float32)
        return copy.deepcopy(self._zero_pclweights)

    def run(self, num_global_epochs):
        self._run_fedsoft(num_global_epochs)

    def train_client(self,k):
        new_client = self.client_vec[k].run()
        # print("new_client",new_client.prototypes)
        # self.client_vec[k] = new_client
        # self.client_vec[k].server = self
        return k, new_client.get_model_dict(),new_client.get_pclmodel_dict(),new_client.prototypes,new_client.density

    def _run_fedsoft(self, num_global_epochs):
        validation_dict = {}
        # mp.set_start_method('spawn')
        for t in tqdm(range(self.start_epoch,num_global_epochs)):
            self.time_alpha = t/num_global_epochs
            # Importance estimation
            if t % self.server_solver.estimation_interval == 0:
                self.importance_weights_matrix = []  # dim = (num_clients, num_clusters)
                for client in self.client_vec:
                    client.estimate_importance_weights('fedsoft')
                    self.importance_weights_matrix.append(client.get_importance())
                self.importance_weights_matrix = np.array(self.importance_weights_matrix)
                self.importance_weights_matrix /= np.sum(self.importance_weights_matrix, axis=0)

            # Client selection
            selection = []
            if self.server_solver.do_selection:
                for s in range(self.num_clusters):
                    selection.append(np.random.choice(a=range(self.num_clients), size=self.server_solver.selection_size,
                                                      p=self.importance_weights_matrix[:, s], replace=False).tolist())
                logger.log_client_selection(self.exp_id, t, self._idx_to_id(selection))
            else:
                selection = np.tile(range(self.num_clients), reps=(self.num_clusters, 1))

            # Local updates
            for k in np.unique(np.concatenate(selection).ravel()):
                self.client_vec[k].run()
            
            # with mp.Pool(processes=3) as pool:  
            #     results = pool.map(self.train_client, np.unique(np.concatenate(selection).ravel()))

            # for k, model,pclmodel,prototype,density in results:
            #     print(k,prototype)
            #     self.client_vec[k].model = model
            #     self.client_vec[k].pclmodel = pclmodel
            #     self.client_vec[k].prototypes = prototype
            #     self.client_vec[k].density = density
            #     self.client_vec[k].server = self
            # for client in self.client_vec:
            #     print(client.prototypes)

            # Aggregation
            self._aggregate_fedsoft(selection,t)
            print("Round {} Finish".format(t))
            with open('./log/{}/round.txt'.format(self.exp_id), 'w') as f:
                f.write(str(t))

            # Validation
            # if self.validator is not None:
            #     validation_dict[str(t)] = self.validator.validate(client_vec=self.client_vec,
            #                                                       cluster_vec=self.cluster_vec, t=t)
            #     if t % validation_data_flush_interval == 0 or t == num_global_epochs - 1:
            #         logger.log_validation_data(self.exp_id, validation_dict)

    def _aggregate_fedsoft(self, selection,t):
        for s in range(self.num_clusters):
            next_weights = self.generate_zero_weights()
            selection_weight = self.importance_weights_matrix[selection[s]][s]
            selection_weight_sum = np.sum(selection_weight, axis=0)
            for k in selection[s]:
                if self.server_solver.do_selection:
                    aggregation_weight = self.importance_weights_matrix[k][s]/selection_weight_sum
                    # aggregation_weight = 1. / self.server_solver.selection_size
                else:
                    aggregation_weight = self.importance_weights_matrix[k][s]
                client_weights = self.client_vec[k].get_model_dict()
                for key in next_weights.keys():
                    next_weights[key] += aggregation_weight * client_weights[key].cpu()
            self.cluster_vec[s].load_state_dict(state_dict=next_weights)
        
        pcl_selection = np.unique(np.concatenate(selection).ravel())
        next_pclweights = self.generate_zero_pclweights()
        # print(self.client_vec[selection[0][0]].prototypes)
        # print(self.client_vec[selection[0][0]].prototypes)
        next_prototypes = torch.zeros_like(self.client_vec[selection[0][0]].prototypes).cpu()
        next_density = torch.zeros_like(self.client_vec[selection[0][0]].density).cpu()

        # proto_weight_all = F.softmax(torch.from_numpy(self.importance_weights_matrix), dim=1)
        proto_weight_all = torch.from_numpy(self.importance_weights_matrix)
        print("importance_weights_matrix",self.importance_weights_matrix)
        print("proto_weight_all",np.sum(self.importance_weights_matrix, axis=0))
        for k in pcl_selection:
            pclaggregation_weight = 1. / len(pcl_selection)
            # proto_weight = proto_weight_all[k].cpu()
            proto_weight = proto_weight_all[k]
            client_pclweights = self.client_vec[k].get_pclmodel_dict()
            for key in next_pclweights.keys():
                next_pclweights[key] += pclaggregation_weight * client_pclweights[key].cpu()

            # print(proto_weight.shape)
            # print(self.client_vec[k].prototypes.shape)
            # print(self.client_vec[k].density.shape)
            next_prototypes += proto_weight.reshape(-1, 1) * self.client_vec[k].prototypes.cpu()
            next_density += proto_weight * self.client_vec[k].density.cpu()
        for k in pcl_selection:        
            self.client_vec[k]._pclmodel.load_state_dict(state_dict=next_pclweights)
            self.client_vec[k].pclmodel.update_parameters(self.client_vec[k]._pclmodel)
        self.prototypes = next_prototypes
        self.density = next_density
        print("next_prototypes",next_prototypes)

        # Save next_pclweights, which appears to be a state_dict of a model
        torch.save(next_pclweights, './log/{}/next_pclweights_{}.pth'.format(self.exp_id, t))

        # Save next_prototypes and next_density, which appear to be tensors
        torch.save(next_prototypes, './log/{}/next_prototypes_{}.pth'.format(self.exp_id, t))
        torch.save(next_density, './log/{}/next_density_{}.pth'.format(self.exp_id, t))

    def _idx_to_id(self, mat):
        return mat
        # retval = []
        # for vec in mat:
        #     retval.append([self.client_vec[k].ID for k in vec])
        # return retval

class Sever_Predict(Server_FedHeart):
    def __init__(self, model_fn, pclmodel_fn, client_vec, num_clusters, validator, server_solver,pretrain_epoch, exp_id='default_exp'):
        self.model_fn = model_fn
        self.pclmodel_fn = pclmodel_fn
        self.client_vec = client_vec
        self.validator = validator
        self.server_solver = server_solver
        self.exp_id = exp_id
        self.client_init_model = model_fn()
        self.client_init_pclmodel = pclmodel_fn()
        self.pretrain_epoch = pretrain_epoch

        self.num_clients = len(client_vec)
        self.num_clusters = num_clusters
        self.cluster_vec = []
        for _ in range(self.num_clusters):
            self.cluster_vec.append(model_fn())
        for i in range(self.num_clusters):
            self.cluster_vec[i] = self.cluster_vec[i].to(device)

        self.importance_weights_matrix = []
        self._zero_weights = None
        self._zero_pclweights = None
        self.prototypes = None
        self.density = None

        self.start_epoch = 0
        try:
            self.load_saved_data(self.pretrain_epoch)
            
            print(f"Resuming from epoch {self.pretrain_epoch}")
        except:
            print("Starting from scratch or error in reading checkpoint.")

        self.num_samples_vec = np.array([])
        for client in client_vec:
            client.attach_to_server(server=self)
            self.num_samples_vec = np.append(self.num_samples_vec, client.num_samples)
        self.num_samples = np.sum(self.num_samples_vec)
        self.number_samples_fraction_vec = self.num_samples_vec / self.num_samples


        

    def load_saved_data(self, t):
        self.prototypes = torch.load('./log/{}/next_prototypes_{}.pth'.format(self.exp_id, t))
        self.density = torch.load('./log/{}/next_density_{}.pth'.format(self.exp_id, t))
        pclweights = torch.load('./log/{}/next_pclweights_{}.pth'.format(self.exp_id, t))
        self.client_init_pclmodel.load_state_dict(state_dict=pclweights)
        # print("Loaded prototypes, density and pclweights from epoch {}".format(t))
        # print("Prototypes: ", self.prototypes)
        # print("Density: ", self.density)
        # print("PCL Weights: ", pclweights)
        return pclweights

    def train_client(self,k):
        new_client = self.client_vec[k].run()
        # print("new_client",new_client.prototypes)
        # self.client_vec[k] = new_client
        # self.client_vec[k].server = self
        return k, new_client.get_model_dict(),new_client.get_pclmodel_dict(),new_client.prototypes,new_client.density

    def _run_fedsoft(self, num_global_epochs):
        validation_dict = {}
        # mp.set_start_method('spawn')
        for t in tqdm(range(self.start_epoch,num_global_epochs)):
            self.time_alpha = t/num_global_epochs
            # Importance estimation
            # if t % self.server_solver.estimation_interval == 0:
            if t == 0:
                self.importance_weights_matrix = []  # dim = (num_clients, num_clusters)
                for client in self.client_vec:
                    client.estimate_importance_weights('fedsoft')
                    self.importance_weights_matrix.append(client.get_importance())
                self.importance_weights_matrix = np.array(self.importance_weights_matrix)
                self.importance_weights_matrix /= np.sum(self.importance_weights_matrix, axis=0)

            # Client selection
            selection = []
            if self.server_solver.do_selection:
                for s in range(self.num_clusters):
                    selection.append(np.random.choice(a=range(self.num_clients), size=self.server_solver.selection_size,
                                                      p=self.importance_weights_matrix[:, s], replace=False).tolist())
                logger.log_client_selection(self.exp_id, t, self._idx_to_id(selection))
            else:
                selection = np.tile(range(self.num_clients), reps=(self.num_clusters, 1))

            # Local updates
            for k in np.unique(np.concatenate(selection).ravel()):
                self.client_vec[k].run()
            
            # with mp.Pool(processes=3) as pool:  # 使用4个进程
            #     results = pool.map(self.train_client, np.unique(np.concatenate(selection).ravel()))

            # for k, model,pclmodel,prototype,density in results:
            #     print(k,prototype)
            #     self.client_vec[k].model = model
            #     self.client_vec[k].pclmodel = pclmodel
            #     self.client_vec[k].prototypes = prototype
            #     self.client_vec[k].density = density
            #     self.client_vec[k].server = self
            # for client in self.client_vec:
            #     print(client.prototypes)

            # Aggregation
            self._aggregate_fedsoft(selection,t)
            

            # Validation
            if self.validator is not None:
                if (t+1) % 100 == 0 or t == num_global_epochs - 1:
                    validation_dict[str(t)] = self.validator.validate(client_vec=self.client_vec,
                                                                  cluster_vec=self.cluster_vec, t=t)
                # if t % validation_data_flush_interval == 0 or t == num_global_epochs - 1:
                
                    logger.log_validation_data(self.exp_id, validation_dict)

    def _aggregate_fedsoft(self, selection,t):
        for s in range(self.num_clusters):
            next_weights = self.generate_zero_weights()
            selection_weight = self.importance_weights_matrix[selection[s],s]
            selection_weight_sum = np.sum(selection_weight, axis=0)
            for k in selection[s]:
                if self.server_solver.do_selection:
                    # aggregation_weight = self.importance_weights_matrix[k][s]/selection_weight_sum
                    aggregation_weight = 1. / self.server_solver.selection_size
                else:
                    aggregation_weight = self.importance_weights_matrix[k][s]
                client_weights = self.client_vec[k].get_model_dict()
                for key in next_weights.keys():
                    next_weights[key] += aggregation_weight * client_weights[key].cpu()
            self.cluster_vec[s].load_state_dict(state_dict=next_weights)
        
        # pcl_selection = np.unique(np.concatenate(selection).ravel())
        # next_pclweights = self.generate_zero_pclweights()
        # # print(self.client_vec[selection[0][0]].prototypes)
        # next_prototypes = torch.zeros_like(self.client_vec[selection[0][0]].prototypes).cpu()
        # next_density = torch.zeros_like(self.client_vec[selection[0][0]].density).cpu()

        # # proto_weight_all = F.softmax(torch.from_numpy(self.importance_weights_matrix), dim=1)
        # proto_weight_all = torch.from_numpy(self.importance_weights_matrix)
        # # print("importance_weights_matrix",self.importance_weights_matrix)
        # # print("proto_weight_all",np.sum(self.importance_weights_matrix, axis=0))
        # for k in pcl_selection:
        #     pclaggregation_weight = 1. / len(pcl_selection)
        #     # proto_weight = proto_weight_all[k].cpu()
        #     proto_weight = proto_weight_all[k]
        #     client_pclweights = self.client_vec[k].get_pclmodel_dict()
        #     for key in next_pclweights.keys():
        #         next_pclweights[key] += pclaggregation_weight * client_pclweights[key].cpu()

        #     # print(proto_weight.shape)
        #     # print(self.client_vec[k].prototypes.shape)
        #     # print(self.client_vec[k].density.shape)
        #     next_prototypes += proto_weight.reshape(-1, 1) * self.client_vec[k].prototypes.cpu()
        #     next_density += proto_weight * self.client_vec[k].density.cpu()
        # for k in pcl_selection:        
        #     self.client_vec[k]._pclmodel.load_state_dict(state_dict=next_pclweights)
        #     self.client_vec[k].pclmodel.update_parameters(self.client_vec[k]._pclmodel)
        # self.prototypes = next_prototypes
        # self.density = next_density
        # print("next_prototypes",next_prototypes)

