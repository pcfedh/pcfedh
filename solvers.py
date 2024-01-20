import torch
import torch.nn as nn


class ServerSolver:
    def __init__(self,
                 do_selection=True,           # whether to do client selection, if false selection_size is ignored
                 estimation_interval=1,       # [tau] time interval to estimate the importance weights
                 selection_size=30            # [K] size of selected clients for each cluster
                 ):
        self.do_selection = do_selection
        self.estimation_interval = estimation_interval
        self.selection_size = selection_size

    def to_json_dict(self):
        param_dict = {
            'do_selection': self.do_selection,
            'estimation_interval': self.estimation_interval,
            'selection_size': self.selection_size
        }
        return param_dict


class BaseClientSolver:
    def __init__(self):
        self.batch_size = -1
        self.optimizer_type = None
        self.criterion = None
        self.local_epoch = -1
        self.lr = -1.
        self.lr_scheduler_type = None
        self.lr_step = -1
        self.lr_gamma = -1.
        self.classification = False
        self.num_classes = -1
        self.reg_weight = -1.            # [lambda] regularization coefficient
        self.count_smoother = -1.        # [sigma] smoother for zero data match

    def to_json_dict(self):
        param_dict = {
            'batch_size': self.batch_size,
            'optimizer_type': self.optimizer_type.__name__,
            'criterion': type(self.criterion).__name__,
            'local_epoch': self.local_epoch,
            'lr': self.lr,
            'lr_scheduler_type': self.lr_scheduler_type.__name__,
            'lr_step': self.lr_step,
            'lr_gamma': self.lr_gamma,
            'classification': self.classification,
            'num_classes': self.num_classes,
            'reg_weight': self.reg_weight,
            'count_smoother': self.count_smoother
        }
        return param_dict



class HomonyClientSolver(BaseClientSolver):
    def __init__(self, reg_weight=0.01,pclbatch_size=32, pcllr=1e-2,alpha=0.5):
        super(HomonyClientSolver, self).__init__()
        self.batch_size = 64
        self.optimizer_type = torch.optim.Adam
        self.criterion = nn.MSELoss()
        self.estimate_criterion = nn.MSELoss(reduction='none')
        self.local_epoch = 5
        self.lr = 1e-2
        self.lr_scheduler_type = torch.optim.lr_scheduler.StepLR
        self.lr_step = 100
        self.lr_gamma = 0.5
        self.classification = False
        self.sequence = False

        self.plocal_epoch = 5
        self.pclbatch_size = pclbatch_size
        self.pcloptimizer_type = torch.optim.AdamW
        self.pcllr = pcllr
        self.plr_scheduler_type = torch.optim.lr_scheduler.StepLR
        self.plr_step = 5
        self.plr_gamma = 0.5
        self.reg_weight = reg_weight
        self.count_smoother = 0.0001
        self.alpha = alpha

        self.rand_percent, self.layer_idx, self.eta = 10, 2, 0.1

