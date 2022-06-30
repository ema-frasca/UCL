import wandb
from argparse import Namespace
from arguments import Namespace as CustomNamespace
from utils import random_id


def innested_vars(args: Namespace):
    new_args = vars(args).copy()
    for key, value in new_args.items():
        if isinstance(value, Namespace) or isinstance(value, CustomNamespace):
            new_args[key] = innested_vars(value)
    return new_args


class WandbLogger:
    def __init__(self, args: Namespace, prj='rodo-ucl', entity='ema-frasca', name=None):
        self.active = args.wandb
        if self.active:
            if name is not None:
                name += f'-{random_id(5)}'
            wandb.init(project=prj, entity=entity, config=innested_vars(args), name=name)

    def __call__(self, obj: any):
        if wandb.run:
            wandb.log(obj)
