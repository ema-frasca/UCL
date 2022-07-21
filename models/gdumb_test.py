from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
import torch

class GDUMBtest(ContinualModel):
    NAME = 'gdumb_test'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(GDUMBtest, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.task = 0

    def end_task(self, _):
        self.task += 1

    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        self.opt.zero_grad()
        if self.args.cl_default:
            assert False
        else:
            inputs1, inputs2 = inputs1.to(self.device, non_blocking=True), inputs2.to(self.device, non_blocking=True)
            notaug_inputs = notaug_inputs.to(self.device, non_blocking=True)
            data_dict = {}
            loss = torch.tensor(0.0).to(self.device)
            data_dict['loss'] = 0
            data_dict['penalty'] = 0
            data_dict['ccic'] = 0

        if not self.buffer.is_empty() and self.task > 0:
            buf_inputs, buf_inputs2, tl = self.buffer.get_data(
                self.args.train.batch_size, transform=self.transform)
            buf_dict = self.net.forward(buf_inputs, buf_inputs2, full_out=True)
            buf_loss = buf_dict['loss'].mean()
            data_dict['penalty'] = buf_loss
            loss += data_dict['penalty']

        if loss != 0:
            loss.backward()
        self.opt.step()

        for k in list(data_dict.keys()):
            if k not in ['loss', 'penalty', 'ccic']:
                del data_dict[k]

        data_dict.update({'lr': self.lr_scheduler.optimizer.param_groups[0]['lr']})
        self.buffer.add_data(examples=notaug_inputs, logits=inputs2, task_labels=torch.ones(len(notaug_inputs)).to(self.device) * self.task)

        return data_dict
