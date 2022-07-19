from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import numpy as np

class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Joint, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.loader = None

    def begin_task(self, loader):
        if self.loader is None:
            self.loader = loader
            return
        else:
            self.loader.dataset.data = np.concatenate((self.loader.dataset.data, loader.dataset.data))
            self.loader.dataset.targets = np.concatenate((self.loader.dataset.targets, loader.dataset.targets))
        loader.dataset.data = self.loader.dataset.data.copy()
        loader.dataset.targets = self.loader.dataset.targets.copy()


    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        self.opt.zero_grad()
        if self.args.cl_default:
            labels = labels.to(self.device)
            outputs = self.net.module.backbone(inputs1.to(self.device))
            loss = self.loss(outputs, labels).mean()
            data_dict = {'loss': loss, 'penalty': 0}
        else:
            data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True))
            loss = data_dict['loss'].mean()
            data_dict['loss'] = data_dict['loss'].mean()
            # outputs = self.net.module.backbone(inputs1.to(self.device))
            data_dict['penalty'] = 0

        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.lr_scheduler.optimizer.param_groups[0]['lr']})

        return data_dict
