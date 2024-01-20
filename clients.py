import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import faiss
import FLAG
import torch.nn.functional as F
from losses import hierarchical_contrastive_loss
import json
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error , mean_absolute_error,mean_absolute_percentage_error
import copy
from torch.utils.data import Subset

    
def _eval_with_pooling(x):
    x = F.max_pool1d(
            x.transpose(1, 2),
            kernel_size = x.size(1),
        ).transpose(1, 2).squeeze(1)
    
    return x

def take_per_row(A, indx, num_elem):
  
    all_indx = indx[:,None] + np.arange(num_elem)
    # print('Finished take_per_row',datetime.now().time())
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


class Client_FedHeart_Predction:
    def __init__(self, ID, ds, pcl_ds, ts_ds, solver, tag=''):
        self.ID = ID
        self.tag = tag
        self.ds = ds
        self.pcl_ds = pcl_ds
        self.ts_ds = ts_ds
        self.num_samples = len(ds)
        self.tsds_num = len(ts_ds)
        self.solver = solver
        self.loader = DataLoader(self.ds, batch_size=self.solver.batch_size,shuffle=True)
        self.pclloader = DataLoader(self.pcl_ds, batch_size=self.solver.batch_size,shuffle=True)
        self.pclloader = self.loader
        self.testloader = DataLoader(self.ts_ds, batch_size=self.solver.batch_size*5,shuffle=False)
        self.temperature = 0.2
        
        self.device = FLAG.device
        self.random = np.random.RandomState(seed=ID)
        self.server = None
        self.model = None
        self.pclmodel = None
        self.optimizer = None
        self.pcloptimizer = None
        self.lr_scheduler = None
        self.importance_estimated = []
        self.prototypes = None
        self.density = None
        self.exp_id = None

    def attach_to_server(self, server):
        self.server = server
        self.model = copy.deepcopy(server.client_init_model)
        self.model = self.model.to(FLAG.device)
        self._pclmodel = copy.deepcopy(server.client_init_pclmodel)
        self._pclmodel = self._pclmodel.to(FLAG.device)
        self.pclmodel = torch.optim.swa_utils.AveragedModel(self._pclmodel)
        self.pclmodel.update_parameters(self._pclmodel)
        self.pclmodel = self.pclmodel.to(FLAG.device)
        self.optimizer = self.solver.optimizer_type(params=self.model.parameters(), lr=self.solver.lr)
        self.pcloptimizer = self.solver.pcloptimizer_type(params=self._pclmodel.parameters(), lr=self.solver.pcllr)
        self.lr_scheduler = self.solver.lr_scheduler_type(optimizer=self.optimizer,
                                                          step_size=self.solver.lr_step,
                                                          gamma=self.solver.lr_gamma)
        self.prototypes = server.prototypes
        self.density = server.density
        self.exp_id = server.exp_id

    def estimate_importance_weights(self,method):
        if len(self.importance_estimated)>0:
            return 
        with torch.no_grad():
            table = torch.zeros((self.num_samples, self.server.num_clusters))
            start_idx = 0
            nst_cluster_sample_count = [0] * self.server.num_clusters
            sample_loader = DataLoader(self.ds, batch_size=256)
            if self.server.prototypes == None:
                for s in range(self.server.num_clusters):
                    nst_cluster_sample_count[s] = self.solver.count_smoother * self.num_samples
                self.importance_estimated = np.array([1.0 * nst / self.num_samples for nst in nst_cluster_sample_count])
            else:
                for x, y in sample_loader:
                    x = x.to(FLAG.device)
                    y = y.to(FLAG.device)
                    x_fea = self._pclmodel(x)
                    x_fea = _eval_with_pooling(x_fea)
                    x_fea = nn.functional.normalize(x_fea, dim=1)
                    self.prototypes = nn.functional.normalize(self.prototypes, dim=1)
                    similarities = torch.mm(x_fea,self.prototypes.to(FLAG.device).t())
                    table[start_idx:start_idx + len(x)] = similarities
                    start_idx += len(x)
                max_sim_idx = torch.argmax(table, dim=1).cpu()
                for s in range(self.server.num_clusters):
                    nst_cluster_sample_count[s] += torch.sum(max_sim_idx == s)
                for s in range(self.server.num_clusters):
                    if nst_cluster_sample_count[s] == 0:
                        nst_cluster_sample_count[s] = self.solver.count_smoother * self.num_samples
                self.importance_estimated = np.array([1.0 * nst / self.num_samples for nst in nst_cluster_sample_count])
           
    def get_importance(self, count=True):
        if count:
            return [ust * self.num_samples for ust in self.importance_estimated]
        else:
            return self.importance_estimated

    def get_model_dict(self):
        return copy.deepcopy(self.model.state_dict())

    def get_pclmodel_dict(self):
        return copy.deepcopy(self._pclmodel.state_dict())

    def get_proto_loss(self,cluster_result):
        proto_labels = []
        proto_logits = []
        im2cluster,prototypes,density,q_fea = cluster_result[0],cluster_result[1],cluster_result[2],cluster_result[3]
        prototypes = prototypes.to(self.device)
        density = density.to(self.device)
        pos_proto_id = im2cluster.to(self.device)
        pos_prototypes = prototypes[pos_proto_id]    
        all_proto_id = [i for i in range(len(prototypes))]
           
        if set(all_proto_id)==set(pos_proto_id.tolist()):
            proto_selected = pos_prototypes
            temp_proto = density[pos_proto_id]  
        else:
            neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
            neg_proto_id = list(neg_proto_id)            
            neg_prototypes = prototypes[neg_proto_id]    
            proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
            temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).to(self.device)],dim=0)]  
        
        # compute prototypical logits
        logits_proto = torch.mm(q_fea,proto_selected.t())
        
        # targets for prototype assignment
        labels_proto = torch.linspace(0, q_fea.size(0)-1, steps=q_fea.size(0)).long().to(self.device) # the podditive is in [0,1,2,3....]
        
        # scaling temperatures for the selected prototypes  
        logits_proto /= temp_proto
        
        proto_labels.append(labels_proto)
        proto_logits.append(logits_proto)

        return proto_logits, proto_labels

    def run(self):
        self.model.train()

        self._local_train()

        self.model.eval()



    def _local_train(self):

        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(self.solver.local_epoch):
            for data_predict in self.loader:
                x, y = data_predict
            

                x = x.to(FLAG.device)
                y = y.to(FLAG.device)

                out = self.model(x)

                loss = self.solver.criterion(out, y)

                mse_loss = nn.MSELoss(reduction='sum')
                for i, cluster in enumerate(self.server.cluster_vec):
                    l2 = None
                    for (param_local, param_cluster) in zip(self.model.parameters(), cluster.parameters()):
                        if l2 is None:
                            l2 = mse_loss(param_local, param_cluster)
                        else:
                            l2 += mse_loss(param_local, param_cluster)
                    loss += self.solver.reg_weight / 2 * self.importance_estimated[i] * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()

    def eval(self, model=None):
        if model is None:
            model = self.model
        model.eval()

        loss = 0.
        with torch.no_grad():
            for x, y in self.loader:
                x = x.to(FLAG.device)
                y = y.to(FLAG.device)
                out = model(x)
                loss += self.solver.criterion(out, y).item()
        return loss

    def save(self, path):
        filename = path + "client_" + str(self.ID)
        import pickle
        client_dict = {"ID": self.ID, "tag": self.tag}
        with open(filename + "_dict.pkl", "wb") as f:
            pickle.dump(client_dict, f)
        torch.save(self.ds, filename + "_ds.pth")

    def load(self, path):
        filename = path + "client_" + str(self.ID)
        import pickle
        with open(filename + "_dict.pkl", "rb") as f:
            client_dict = pickle.load(f)
        self.ID = client_dict["ID"]
        self.tag = client_dict["tag"]
        self.ds = torch.load(filename + "_ds.pth")


class Client_FedHeart:
    def __init__(self, ID, ds, pcl_ds, ts_ds, solver, tag=''):
        self.ID = ID
        self.tag = tag
        self.ds = ds
        self.pcl_ds = pcl_ds
        self.ts_ds = ts_ds
        self.num_samples = len(ds)
        self.tsds_num = len(ts_ds)
        self.solver = solver

        self.pclloader = DataLoader(self.pcl_ds, batch_size=self.solver.pclbatch_size,shuffle=True)

        self.temperature = 0.2

        self.device = FLAG.device
        self.random = np.random.RandomState(seed=ID)
        self.server = None
        self.model = None
        self.pclmodel = None
        self.optimizer = None
        self.pcloptimizer = None
        self.lr_scheduler = None
        self.importance_estimated = []
        self.prototypes = None
        self.density = None
        self.exp_id = None

    def attach_to_server(self, server):
        self.server = server
        self.model = copy.deepcopy(server.client_init_model)
        self.model = self.model.to(FLAG.device)
        self._pclmodel = copy.deepcopy(server.client_init_pclmodel)
        self._pclmodel = self._pclmodel.to(FLAG.device)
        self.pclmodel = torch.optim.swa_utils.AveragedModel(self._pclmodel)
        self.pclmodel.update_parameters(self._pclmodel)
        self.pclmodel = self.pclmodel.to(FLAG.device)
        self.optimizer = self.solver.optimizer_type(params=self.model.parameters(), lr=self.solver.lr)
        self.pcloptimizer = self.solver.pcloptimizer_type(params=self._pclmodel.parameters(), lr=self.solver.pcllr)
        self.lr_scheduler = self.solver.lr_scheduler_type(optimizer=self.optimizer,
                                                          step_size=self.solver.lr_step,
                                                          gamma=self.solver.lr_gamma)
        self.plr_scheduler = self.solver.plr_scheduler_type(optimizer=self.pcloptimizer,
                                                          step_size=self.solver.plr_step,
                                                          gamma=self.solver.plr_gamma)
        self.alpha = self.solver.alpha
        self.prototypes = server.prototypes
        self.density = server.density
        self.exp_id = server.exp_id

    def estimate_importance_weights(self,method):
        with torch.no_grad():
            table = torch.zeros((self.num_samples, self.server.num_clusters))
            start_idx = 0
            nst_cluster_sample_count = [0] * self.server.num_clusters
            sample_loader = DataLoader(self.ds, batch_size=256)
            if self.server.prototypes == None:
                for s in range(self.server.num_clusters):
                    nst_cluster_sample_count[s] = self.solver.count_smoother * self.num_samples
                self.importance_estimated = np.array([1.0 * nst / self.num_samples for nst in nst_cluster_sample_count])
            else:
                for x, y in sample_loader:
                    x = x.to(FLAG.device)
                    y = y.to(FLAG.device)
                    x_fea = self._pclmodel(x)
                    x_fea = _eval_with_pooling(x_fea)
                    x_fea = nn.functional.normalize(x_fea, dim=1)
                    self.prototypes = nn.functional.normalize(self.prototypes, dim=1)
                    similarities = torch.mm(x_fea,self.prototypes.t().to(FLAG.device))
                    table[start_idx:start_idx + len(x)] = similarities
                    start_idx += len(x)
                max_sim_idx = torch.argmax(table, dim=1).cpu()
                for s in range(self.server.num_clusters):
                    nst_cluster_sample_count[s] += torch.sum(max_sim_idx == s)
                for s in range(self.server.num_clusters):
                    if nst_cluster_sample_count[s] == 0:
                        nst_cluster_sample_count[s] = self.solver.count_smoother * self.num_samples
                self.importance_estimated = np.array([1.0 * nst / self.num_samples for nst in nst_cluster_sample_count])

    def get_importance(self, count=True):
        if count:
            return [ust * self.num_samples for ust in self.importance_estimated]
        else:
            return self.importance_estimated

    def get_model_dict(self):
        return copy.deepcopy(self.model.state_dict())

    def get_pclmodel_dict(self):
        return copy.deepcopy(self._pclmodel.state_dict())

    def get_proto_loss(self,cluster_result):
        proto_labels = []
        proto_logits = []
        im2cluster,prototypes,density,q_fea = cluster_result[0],cluster_result[1],cluster_result[2],cluster_result[3]
        prototypes = prototypes.to(self.device)
        density = density.to(self.device)
        pos_proto_id = im2cluster.to(self.device)
        pos_prototypes = prototypes[pos_proto_id]
        all_proto_id = [i for i in range(len(prototypes))]

        if set(all_proto_id)==set(pos_proto_id.tolist()):
            proto_selected = pos_prototypes
            temp_proto = density[pos_proto_id]
        else:
            neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
            neg_proto_id = list(neg_proto_id)
            neg_prototypes = prototypes[neg_proto_id]
            proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
            temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).to(self.device)],dim=0)]

        # compute prototypical logits
        logits_proto = torch.mm(q_fea,proto_selected.t())

        # targets for prototype assignment
        labels_proto = torch.linspace(0, q_fea.size(0)-1, steps=q_fea.size(0)).long().to(self.device) # the podditive is in [0,1,2,3....]

        # scaling temperatures for the selected prototypes
        logits_proto /= temp_proto

        proto_labels.append(labels_proto)
        proto_logits.append(logits_proto)

        return proto_logits, proto_labels

    def run(self):
        self.model.train()
        self.pclmodel.train()
        self.model.to(FLAG.device)
        self.pclmodel.to(FLAG.device)

        self._local_train()

        self.model.eval()
        self.pclmodel.eval()
        self._local_cluster()



    def _local_train(self):

        criterion = torch.nn.CrossEntropyLoss()
        self.prototypes = self.server.prototypes
        self.density = self.server.density

        writer = SummaryWriter(f'log/{self.exp_id}/{self.ID}')
        for _ in range(self.solver.plocal_epoch):
            for data_pcl in self.pclloader:

                q, _ = data_pcl

                q = q.to(FLAG.device)
                if self.prototypes != None:
                    q_fea = self._pclmodel(q,mask='all_true')
                    q_fea = _eval_with_pooling(q_fea)
                    q_fea = nn.functional.normalize(q_fea, dim=1)
                    self.prototypes = nn.functional.normalize(self.prototypes, dim=1)
                    similarities = torch.mm(q_fea,self.prototypes.t().to(FLAG.device))
                    max_sim_idx = torch.argmax(similarities, dim=1).cpu()
                else:
                    max_sim_idx = None
                    q_fea = None

                ts_l = q.size(1)

                crop_l = np.random.randint(low=ts_l-20, high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=q.size(0))

                out1 = self._pclmodel(take_per_row(q, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                out2 = self._pclmodel(take_per_row(q, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]

                loss_info = hierarchical_contrastive_loss(
                    out1,
                    out2,

                )

                '''proto loss'''
                loss_proto = 0
                if self.prototypes is not None:
                    output_proto, target_proto = self.get_proto_loss([max_sim_idx,self.prototypes,self.density,q_fea])

                    # ProtoNCE loss
                    if output_proto is not None:
                        loss_proto = 0
                        for proto_out,proto_target in zip(output_proto, target_proto):
                            loss_proto += criterion(proto_out, proto_target)

                '''proto loss'''

                time_alpha = self.server.time_alpha
                loss_all = self.alpha*loss_info + loss_proto

                self.pcloptimizer.zero_grad()
                loss_all.backward()
                self.pcloptimizer.step()

                self.pclmodel.update_parameters(self._pclmodel)
        self.plr_scheduler.step()
        if self.ID==0 and torch.is_tensor(loss_proto):
            print("loss_all: ", loss_all.item(), "loss_proto: ", loss_proto.item(), "loss_info: ", loss_info.item())
        if torch.is_tensor(loss_proto):
            try:
                with open('./log/{}/loss_values_{}.json'.format(self.exp_id,self.ID), 'r') as f:
                    data = json.load(f)
                order_num = max(int(key) for key in data.keys()) + 1
            except (FileNotFoundError, ValueError):
                data = {}
                order_num = 1
            # Add new losses
            data[order_num] = {
                'loss_all': loss_all.item(),
                'loss_proto': loss_proto.item(),
                'loss_info': loss_info.item(),
            }
            writer.add_scalar('Loss/All', loss_all.item(), global_step=order_num)
            writer.add_scalar('Loss/Proto', loss_proto.item(), global_step=order_num)
            writer.add_scalar('Loss/Info', loss_info.item(), global_step=order_num)
            writer.close()
            # Save the data
            with open('./log/{}/loss_values_{}.json'.format(self.exp_id,self.ID), 'w') as f:
                json.dump(data, f, indent=4)

    def eval(self, model=None):
        if model is None:
            model = self.model
        model.eval()

        loss = 0.
        with torch.no_grad():
            for x, y in self.loader:
                x = x.to(FLAG.device)
                y = y.to(FLAG.device)
                out = model(x)
                loss += self.solver.criterion(out, y).item()
        return loss

    def save(self, path):
        filename = path + "client_" + str(self.ID)
        import pickle
        client_dict = {"ID": self.ID, "tag": self.tag}
        with open(filename + "_dict.pkl", "wb") as f:
            pickle.dump(client_dict, f)
        torch.save(self.ds, filename + "_ds.pth")

    def load(self, path):
        filename = path + "client_" + str(self.ID)
        import pickle
        with open(filename + "_dict.pkl", "rb") as f:
            client_dict = pickle.load(f)
        self.ID = client_dict["ID"]
        self.tag = client_dict["tag"]
        self.ds = torch.load(filename + "_ds.pth")

    def run_kmeans(self,x,initial_centers):
        """
        Args:
            x: data to be clustered
        """

        results = {'im2cluster':[],'centroids':[],'density':[]}

        x_ = x.numpy()

        d = x_.shape[1]
        k = self.server.num_clusters

        clus = faiss.Kmeans(d, k, niter=1)
        clus.gpu = True
        clus.verbose = True

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        if initial_centers is None:
            clus.train(x_)
        else:

            clus.train(x_,init_centroids=initial_centers.numpy(),weights=None)
        index.add(clus.centroids)
        D, I = index.search(x_, 1)

        im2cluster = [int(n[0]) for n in I]

        centroids = clus.centroids
        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                density[i] = d

        #if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = self.temperature*density/density.mean()  #scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda(FLAG.device)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        # im2cluster = torch.LongTensor(im2cluster_).cuda(FLAG.device)
        density = torch.Tensor(density).cuda(FLAG.device)

        # results['centroids'].append(centroids)
        # results['density'].append(density)
        # results['im2cluster'].append(im2cluster)

        return centroids,density

    def compute_features(self, eval_loader, model):
        # print('Computing features...')
        model.eval()

        features = 0

        for i, data in enumerate(eval_loader):
            with torch.no_grad():
                data_ = data[0].cuda(FLAG.device)

                feat = model(data_)
                feat = _eval_with_pooling(feat)
                feat = nn.functional.normalize(feat, dim=1)

                if torch.is_tensor(features):
                    features = torch.cat([features,feat.cpu()],dim=0)
                else:
                    features = feat.cpu()


        return features.cpu()

    def _local_cluster(self):

        features = self.compute_features(self.pclloader, self.pclmodel)
        prototypes,density = self.run_kmeans(features, self.prototypes)

        self.density = density.cpu()
        self.prototypes = prototypes.cpu()



