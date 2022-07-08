from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug

class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(Er, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)

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

        if not self.buffer.is_empty():
            buf_inputs, buf_inputs2 = self.buffer.get_data(
                self.args.train.batch_size, transform=self.transform)
            buf_loss = self.net.forward(buf_inputs, buf_inputs2)['loss'].mean()
            data_dict['penalty'] = buf_loss
            loss += data_dict['penalty']

        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.lr_scheduler.optimizer.param_groups[0]['lr']})
        self.buffer.add_data(examples=notaug_inputs, logits=inputs2)

        return data_dict
