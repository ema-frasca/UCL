from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
# from augmentations import get_aug
import torch

class CCIC(ContinualModel):
    NAME = 'ccic'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(CCIC, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.task = 0

    def end_task(self, _):
        self.task += 1

    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        self.opt.zero_grad()
        if self.args.cl_default:
            assert False
            labels = labels.to(self.device)
            outputs = self.net.module.backbone(inputs1.to(self.device))
            loss = self.loss(outputs, labels).mean()
            data_dict = {'loss': loss, 'penalty': 0}
        else:
            data_dict = self.net.forward(inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True), full_out=True)
            loss = data_dict['loss'].mean()
            data_dict['loss'] = data_dict['loss'].mean()
            data_dict['penalty'] = 0
            data_dict['ccic'] = 0

        if not self.buffer.is_empty():
            buf_inputs, buf_inputs2, tl = self.buffer.get_data(
                self.args.train.batch_size, transform=self.transform)
            buf_dict = self.net.forward(buf_inputs, buf_inputs2, full_out=True)
            buf_loss = buf_dict['loss'].mean()
            data_dict['penalty'] = buf_loss
            loss += data_dict['penalty']

            if self.task > 0:
                all_z1 = torch.cat([buf_dict['z1'], data_dict['z1']], dim=0)
                all_z2 = torch.cat([buf_dict['z2'], data_dict['z2']], dim=0)
                all_tl = torch.cat([torch.ones(len(buf_dict['z1'])).to(self.device) * self.task, tl], dim=0)
                
                sourcer = torch.arange(len(all_z1)).to(self.device)
                matcher = torch.randperm(len(all_z1)).to(self.device)
                doom_mask = all_tl[sourcer] == all_tl[matcher]
                
                all_z1 = all_z1[sourcer][~doom_mask]
                all_z2 = all_z2[matcher][~doom_mask]

                if len(all_z1):
                    if self.args.train.ccic_dist == 'eucl':
                        dist = -(all_z1 - all_z2).pow(2).mean()
                    elif self.args.train.ccic_dist == 'cos':
                        z1_norm = F.normalize(all_z1, dim=1)
                        z2_norm = F.normalize(all_z2, dim=1)
                        dist = F.cosine_similarity(z1_norm, z2_norm, dim=1).mean() + 1
                    data_dict['ccic'] = dist * self.args.train.alpha
                    loss += data_dict['ccic']

        loss.backward()
        self.opt.step()

        for k in list(data_dict.keys()):
            if k not in ['loss', 'penalty', 'ccic']:
                del data_dict[k]

        data_dict.update({'lr': self.lr_scheduler.optimizer.param_groups[0]['lr']})
        self.buffer.add_data(examples=notaug_inputs, logits=inputs2, task_labels=torch.ones(len(notaug_inputs)).to(self.device) * self.task)

        return data_dict
