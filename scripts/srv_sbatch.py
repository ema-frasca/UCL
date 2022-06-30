import os
import sys
from argparse import ArgumentParser
from random import randint

parser = ArgumentParser(description='srv-sbatch for ucl', allow_abbrev=False)
parser.add_argument('--path', type=str, required=True, help='File with sbatch list.')
parser.add_argument('--sh_out', type=str, required=True, help='Shell file to execute for sbatch.')
parser.add_argument('--nickname', type=str, default=None, help='Job name.')
parser.add_argument('--cycles', type=int, default=1, help='The number of cycles.')
parser.add_argument('--max_jobs', type=int, default=None, help='The maximum number of jobs.')
parser.add_argument('--exclude', type=str, default='', help='excNodeList.')
parser.add_argument('--n_gpus', type=int, default=1, help='Number of requested GPUs per job')
parser.add_argument('--custom_time', default=None, type=str, help='customizes sbatch time')
parser.add_argument('--user_name', type=str, default='efrascaroli')
parser.add_argument('--envname', type=str, default='SRV-Continual')
parser.add_argument('--sbacciu', action='store_true', help='Sbatch just after finishing to generate the file')
parser.add_argument('--one_node', action='store_true', help='Queue everything on one node only')
parser.add_argument('--dev', action='store_true', help='Sbatch on dev gpu')
args = parser.parse_args()

red = int(args.cycles)

with open(args.path) as f:
    sbacci = f.read().splitlines()

# low cuda ram exclude: --exclude=ajeje,germano,helmut,carabbaggio,lurcanio,aimagelab-srv-10,vegeta

if args.one_node:
    sbacci = [f'/homes/{args.user_name}/.conda/envs/{args.envname}/bin/python main.py {x} ;'
              for x in sbacci if not x.startswith('#') and len(x.strip())] * red
else:
    sbacci = [f'\' /homes/{args.user_name}/.conda/envs/{args.envname}/bin/python main.py {x} \''
              for x in sbacci if not x.startswith('#') and len(x.strip())] * red


max_jobs = args.max_jobs if args.max_jobs is not None else len(sbacci)
nickname = args.nickname if args.nickname is not None else 'der-verse'

gridsh = '''#!/bin/bash
#SBATCH -p <partition>
#SBATCH --job-name=<nick>
#SBATCH --array=0-<lung>%<mj>
#SBATCH --nodes=1
<time>
#SBATCH --output="/homes/<user>/output/std/<nick>_%A_%a.out"
#SBATCH --error="/homes/<user>/output/std/<nick>_%A_%a.err"
#SBATCH --gres=gpu:<ngpu>
<xcld>

arguments=(
<REPLACEME>
)

sleep $(($RANDOM % 20)); ${arguments[$SLURM_ARRAY_TASK_ID]}
'''

if args.one_node:
    gridsh = '''#!/bin/bash
#SBATCH -p <partition>
#SBATCH --job-name=<nick>
#SBATCH --nodes=1
<time>
#SBATCH --output="/homes/<user>/output/std/<nick>_%j.out"
#SBATCH --error="/homes/<user>/output/std/<nick>_%j.err"
#SBATCH --gres=gpu:<ngpu>
<xcld>

<REPLACEME>
'''

gridsh = gridsh.replace('<partition>', 'dev' if args.dev else 'prod')
gridsh = gridsh.replace('<user>', args.user_name)
gridsh = gridsh.replace('<REPLACEME>', '\n'.join(sbacci))
gridsh = gridsh.replace('<lung>', str(len(sbacci)-1))
gridsh = gridsh.replace('<xcld>', ('#SBATCH --exclude='+args.exclude) if len(args.exclude) else '')
gridsh = gridsh.replace('<nick>', nickname)
gridsh = gridsh.replace('<mj>', str(max_jobs))
gridsh = gridsh.replace('<ngpu>', str(args.n_gpus))
gridsh = gridsh.replace('<time>', ('#SBATCH --time=' + args.custom_time) if args.custom_time is not None else '')

with open(args.sh_out, 'w') as f:
    f.write(gridsh)

if args.sbacciu:
    # /homes/efrascaroli/Continual/sbatch/ready_for_sbatch.sh
    os.system(f'sbatch {args.sh_out}')
