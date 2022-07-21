import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from tools.knn_monitor import kmeans_monitor
from datasets import get_dataset
from datetime import datetime
from utils import create_if_not_exists, random_id
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from typing import Tuple
from utils.wandbsc import WandbLogger, innested_vars


def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if classifier is not None:
                outputs = classifier(outputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


def main(device, args):
    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)

    # define model
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))

    # logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    bb_name = 'bt' if args.model.name == 'barlowtwins' else 'ss' if args.model.name == 'simsiam' else ''
    name = model.name if hasattr(model, 'name') else model.NAME
    name += '_' + bb_name
    wblog = WandbLogger(model.args, name=name)
    accuracy = 0

    for t in range(dataset.N_TASKS):
        train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
        if hasattr(model, 'begin_task'):
            model.begin_task(train_loader)
        global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training Task {t + 1}/{dataset.N_TASKS}')
        for epoch in global_progress:
            model.train()
            results, results_mask_classes = [], []

            local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}',
                                  disable=args.hide_progress)
            for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
                data_dict = model.observe(images1, labels, images2, notaug_images)
                wblog({'train': {**data_dict, 'epoch': epoch, 'task': t}})
                # logger.update_scalers(data_dict)

            global_progress.set_postfix(data_dict)

            if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
                for i in range(len(dataset.test_loaders)):
                    acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i],
                                                dataset.test_loaders[i], device, args.cl_default, task_id=t,
                                                k=min(args.train.knn_k, len(memory_loader.dataset)))
                    results.append(acc)
                mean_acc = np.mean(results)
                wblog({'eval': {'acc-mean': mean_acc, 'epoch': epoch, 'task': t, 'te': t*args.train.stop_at_epoch+epoch,
                                **{f'acc-{i}': acc for i, acc in enumerate(results)}}})

            epoch_dict = {"epoch": epoch, "accuracy": mean_acc}
            global_progress.set_postfix(epoch_dict)
            # logger.update_scalers(epoch_dict)

        if args.cl_default:
            accs = evaluate(model.net.module.backbone, dataset, device)
            results.append(accs[0])
            results_mask_classes.append(accs[1])
            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if hasattr(model, 'end_task'):
            model.end_task(train_loader)

        for i in range(len(dataset.test_loaders)):
            acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i],
                                        dataset.test_loaders[i], device, args.cl_default, task_id=t,
                                        k=min(args.train.knn_k, len(memory_loader.dataset)))
            results.append(acc)
        mean_acc = np.mean(results)

        kmm, kmm_s, _ = kmeans_monitor(model.net.module.backbone, dataset.test_loaders, args.cl_default, (t+1) * dataset.N_CLASSES_PER_TASK)

        wblog({'test': {'acc-mean': mean_acc, 'task': t,
                        **{f'acc-{i}': acc for i, acc in enumerate(results)},
                        'kmm-mean': kmm, 'kmm-std': kmm_s}})

        if args.save_checks:
            chech_dir = os.path.join(args.ckpt_dir, args.dataset.name, f'{name}-{wblog.run_id}')
            create_if_not_exists(chech_dir)
            if t == 0:
                save_dict = innested_vars(model.args)
                filename = f'args.pyd'
                with open(os.path.join(chech_dir, filename), 'w') as f:
                    f.write(str(save_dict))
            model_path = os.path.join(chech_dir, f"task_{t}.pth")
            torch.save(model.net.module.backbone.state_dict(), model_path)
            print(f"Backbone saved to {model_path}")
        # torch.save({
        #   'epoch': epoch+1,
        #   'state_dict':model.net.state_dict()
        # }, model_path)
        # print(f"Task Model saved to {model_path}")
        # with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        #   f.write(f'{model_path}')


    # if args.eval is not False and args.cl_default is False:
    #     args.eval_from = model_path


if __name__ == "__main__":
    args = get_args()
    # args.id = random_id()
    # if args.cl_model is not None:
    #     args.model.cl_model = args.cl_model
    main(device=args.device, args=args)
    # completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    # os.rename(args.log_dir, completed_log_dir)
    # print(f'Log file has been saved to {completed_log_dir}')
