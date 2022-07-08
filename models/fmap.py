import torch
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from augmentations import get_aug
from utils.spectral_analysis import calc_cos_dist, calc_euclid_dist, calc_ADL_knn, normalize_A, find_eigs

class FMap(ContinualModel):
    NAME = 'fmap'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, len_train_loader, transform):
        super(FMap, self).__init__(backbone, loss, args, len_train_loader, transform)
        self.buffer = Buffer(self.args.model.buffer_size, self.device)
        self.task = 0

    def begin_task(self, train_loader):
        pass

    def get_evects(self, features: torch.Tensor):
        if self.args.train.fm_cos_dist:
            dists = calc_cos_dist(features)
        else:
            dists = calc_euclid_dist(features)
        A, D, L = calc_ADL_knn(dists, k=self.args.train.fm_knn_k, symmetric=True)
        L = torch.eye(A.shape[0]).to(A.device) - normalize_A(A, D)
        evals, evects = find_eigs(L, n_pairs=self.args.train.fm_dim)
        return evects, evals

    def fmap_loss(self, features1: torch.Tensor, features2: torch.Tensor):
        ev1, evals1 = self.get_evects(features1)
        ev2, evals2 = self.get_evects(features2)

        # egap = self.args.train.fm_dim
        eval_diff = evals1[2:] - evals1[1:-1]
        egap = torch.argmax(eval_diff).item() + 1
        if self.args.train.fm_dim_gap:
            ev1 = ev1[:, :egap]
            ev2 = ev2[:, :egap]

        c = ev2.T @ ev1
        target = torch.eye(c.shape[0]).to(c.device)

        return torch.square(target - c.abs()).sum(), egap

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
            buf_inputs, buf_labels, buf_features1 = self.buffer.get_data(
                self.args.train.batch_size, transform=self.transform)
            buf_features2 = self.net.module.backbone(buf_inputs, return_features=not self.args.train.fm_logits)
            data_dict['penalty'], egap = self.fmap_loss(buf_features1, buf_features2)
            data_dict['egap'] = egap
            loss += data_dict['penalty'] * self.args.train.fm_weight

        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.lr_scheduler.optimizer.param_groups[0]['lr']})

        return data_dict

    def end_task(self, train_loader):
        if self.args.train.fm_knn_dynamic and self.task > 0:
            self.args.train.fm_knn_k = self.args.train.fm_knn_k * self.task // (self.task + 1)
        self.task += 1
        assert self.args.model.fm_balanced is True
        n_data = {}
        if (self.buffer.is_full()):
            buffer_x, buffer_labels, buffer_features = self.buffer.get_all_data(self.transform)
            buffer_not_aug_x = self.buffer.examples
            id_labels = buffer_labels.unique()
            for l in id_labels.tolist():
                n_data[l] = {'inputs': buffer_x[buffer_labels == l], 'not_aug_inputs': buffer_not_aug_x[buffer_labels == l], 'len': (buffer_labels == l).sum().item()}
        tload = iter(train_loader)
        is_full = False
        target_len = 0
        while not is_full:
            (inputs, _, not_aug_inputs), labels = next(tload)
            id_labels = labels.unique()
            ids = torch.cat((id_labels, torch.tensor(list(n_data.keys()), dtype=id_labels.dtype))).unique()
            target_len = self.args.model.buffer_size // len(ids)
            for l in id_labels.tolist():
                if l not in n_data.keys():
                    n_data[l] = {'inputs': inputs[labels == l], 'not_aug_inputs': not_aug_inputs[labels == l], 'len': (labels == l).sum().item()}
                elif n_data[l]['len'] < target_len:
                    n_data[l]['inputs'] = torch.cat((n_data[l]['inputs'], inputs[labels == l]))
                    n_data[l]['not_aug_inputs'] = torch.cat((n_data[l]['not_aug_inputs'], not_aug_inputs[labels == l]))
                    n_data[l]['len'] += (labels == l).sum().item()

            is_full = True
            for key, data in n_data.items():
                if data['len'] < target_len:
                    is_full = False
                    break
                else:
                    data['inputs'] = data['inputs'][:target_len]
                    data['not_aug_inputs'] = data['not_aug_inputs'][:target_len]
                    data['len'] = target_len

        self.buffer.empty()
        for key, data in n_data.items():
            self.net.eval()
            with torch.no_grad():
                features = self.net.module.backbone(data['inputs'].to(self.device), return_features=True)
            self.net.train()
            self.buffer.add_data(data['not_aug_inputs'].to(self.device), logits=features, labels=torch.full((target_len,), key))
