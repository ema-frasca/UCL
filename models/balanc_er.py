import torch
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug

class BalancEr(ContinualModel):
    NAME = 'balanc_er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(BalancEr, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.task = 0

    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        self.opt.zero_grad()
        if self.args.cl_default:
            labels = labels.to(self.device)
            outputs = self.net.module.backbone(inputs1.to(self.device))
            loss = self.loss(outputs, labels).mean()
            data_dict = {'loss': loss, 'penalty': 0}
        else:
            tot_inputs = inputs1.to(self.device)
            tot_inputs2 = inputs2.to(self.device)
            if self.task > 0:
                buf_qty = int(self.args.train.batch_size * (self.task / (self.task + 1)))
                stream_qty = self.args.train.batch_size - buf_qty
                buf_inputs, buf_inputs2 = self.buffer.get_data(buf_qty, transform=self.transform)
                idxes = torch.randperm(len(tot_inputs))[:stream_qty]
                tot_inputs = tot_inputs[idxes]
                tot_inputs2 = tot_inputs2[idxes]
                tot_inputs = torch.cat((tot_inputs, buf_inputs), dim=0)
                tot_inputs2 = torch.cat((tot_inputs2, buf_inputs2), dim=0)
            data_dict = self.net.forward(tot_inputs, tot_inputs2)
            loss = data_dict['loss'].mean()
            data_dict['loss'] = data_dict['loss'].mean()
            # outputs = self.net.module.backbone(inputs1.to(self.device))
            data_dict['penalty'] = 0

        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.lr_scheduler.optimizer.param_groups[0]['lr']})
        self.buffer.add_data(examples=notaug_inputs, logits=inputs2)

        return data_dict

    def end_task(self, train_loader):
        self.task += 1
