import argparse
import os
import torch

import numpy as np
import torch
import random

import re 
import yaml

import shutil
import warnings

from datetime import datetime
from ast import literal_eval
from utils.conf import base_path, base_path_dataset


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def get_namespace_item(dictionary: Namespace, key_array: list):
    """
    Get the item from a nested dictionary.
    """
    if len(key_array) == 0:
        return vars(dictionary)
    else:
        return get_namespace_item(vars(dictionary)[key_array[0]], key_array[1:])


def try_eval(var):
    try:
        return literal_eval(var)
    except:
        return var


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default=base_path_dataset())
    parser.add_argument('--log_dir', type=str, default=base_path()+'logs')
    parser.add_argument('--ckpt_dir', type=str, default=base_path()+'checkpoints')
    parser.add_argument('--ckpt_dir_1', type=str, default=base_path()+'checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--cl_default', action='store_true')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--ood_eval', action='store_true',
                        help='Test on the OOD set')

    parser.add_argument('--wandb', action='store_true',
                        help='Wandb activate')
    parser.add_argument('--custom_log', action='store_true',
                        help='My personal log.')
    parser.add_argument('--save_checks', action='store_true',
                        help='Save checkpoints.')
    # parser.add_argument('--cl_model', type=str, default=None)
    args, unknown = parser.parse_known_args()
    assert len(vars(args)) > 0, "No arguments provided"

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    if len(unknown) > 0:
        unargs = {}
        last_arg = ''
        for arg in unknown:
            if arg.startswith('--'):
                arg = arg[2:]
                unargs[arg] = True
                last_arg = arg
            else:
                unargs[last_arg] = try_eval(arg)
        for key, value in unargs.items():
            key_arr = key.split('.')
            obj = get_namespace_item(args, key_arr[:-1])
            obj[key_arr[-1]] = value
            # vars(args)[key] = value

    if args.debug:
        if args.train: 
            args.train.batch_size = 64
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
        if args.eval: 
            args.eval.batch_size = 64
            args.eval.num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0


    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    # args.log_dir = os.path.join(args.log_dir, 'in-progress_'+datetime.now().strftime('%m%d%H%M%S_')+args.name)
    #
    # os.makedirs(args.log_dir, exist_ok=False)
    # print(f'creating file {args.log_dir}')
    # os.makedirs(args.ckpt_dir, exist_ok=True)
    #
    # shutil.copy2(args.config_file, args.log_dir)
    set_deterministic(args.seed)


    vars(args)['aug_kwargs'] = {
        'name': args.model.name,
        'image_size': args.dataset.image_size
    }
    vars(args)['dataset_kwargs'] = {
        # 'name':args.model.name,
        # 'image_size': args.dataset.image_size,
        'dataset': args.dataset.name,
        'data_dir': args.data_dir,
        'download': args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
        # 'drop_last': True,
        # 'pin_memory': True,
        # 'num_workers': args.dataset.num_workers,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    return args
