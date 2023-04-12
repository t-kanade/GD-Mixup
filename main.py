import wandb
import argparse
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import basicDataset
from gMixup import MixupDataset
from GDM import Agent
from utils import prepare_dataset_x
from comparsion import compare



parser = argparse.ArgumentParser()
parser.add_argument('--gMixup', type=bool, default=False, help='whether to use gmixup')
parser.add_argument('--GD', type=bool, default=False, help='whether to use GD')
parser.add_argument('--gpu', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--dataset_name', type=str, default='DD')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for origin data in dataloader')
parser.add_argument('--model', type=str, default='GCN', help='GCN, GIN')
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--nconvs', type=int, default=3)
parser.add_argument('--pooling', type=str, default='mean', help='mean or sum')
parser.add_argument('--net_norm', type=str, default='none',
                    help='batchnorm, layernorm, instancenorm, groupnorm, none')
parser.add_argument('--epochs', type=int, default=50)
# mixup parameters
parser.add_argument('--mixup_fixed', type=int, default=0.5, help='use fixed mixup rate 0.5 or set it 0')
parser.add_argument('--lam_range', type=str, default="[0.005, 0.01]")
parser.add_argument('--aug_ratio', type=float, default=0.15)
parser.add_argument('--aug_num', type=int, default=10)
# distillation parameters
parser.add_argument('--distill_epochs', type=int, default=200, help='the epochs use in GD')
parser.add_argument('--init', type=str, default='real', help='initialize the synthetic data with real data')
parser.add_argument('--stru_discrete', type=int, default=1, help='create discrete structure in synthetic data')
parser.add_argument('--lr_adj', type=float, default=0.0001)
parser.add_argument('--lr_feat', type=float, default=0.0001)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--ipc', type=int, default=0,
                    help=' set the number of condensed samples per class artificially')
parser.add_argument('--reduction_rate', type=float, default=0.1,
                    help='if ipc=0, this param  will be enabled. set the number of condensed samples by the dataset')
parser.add_argument('--outer', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--dis_metric', type=str, default='mse', help='distance metric')
parser.add_argument('--syn_bs', type=int, default=32,
                    help='the batch size for the function prepare_graph to generate synthetic data')
parser.add_argument('--beta', type=float, default=0.1, help='coefficient for the regularization term')
parser.add_argument('--save', type=int, default=0, help='whether to save the condensed graphs')

args = parser.parse_args()



device = args.gpu
if device == 'cuda':
    torch.cuda.set_device(args.gpu_id)

print(args)

# load original data
data = basicDataset(args)
# [trainset, origin_TrainLoader, origin_ValLoader, origin_TestLoader] = data.packed_data
[dataset, train_nums, train_val_nums] = data.packed_data

if args.gMixup == True:
    # create mixup dataset
    MixupD = MixupDataset(data.packed_data, args)
    # [trainset, num_feats, num_classes] = MixupD.mixup_trainset
    [dataset, train_nums, train_val_nums] = MixupD.mixup_data
    dataset = prepare_dataset_x(dataset)

num_feats = dataset[0].x.shape[1]
num_classes = dataset[0].y.shape[0]

trainset = dataset[:train_nums]
valset = dataset[train_nums:train_val_nums]
testset = dataset[train_val_nums:]

test_loader = DataLoader(testset, batch_size=args.test_bs, shuffle=False)
val_loader = DataLoader(valset, batch_size=args.test_bs, shuffle=False)
train_loader = DataLoader(trainset, batch_size=args.test_bs, shuffle=True)

packed_data = [trainset, val_loader, num_feats, num_classes]

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.GD == True:
    if args.dataset_name == 'CIFAR10':
        nnodes_syn_ = 118
    elif args.dataset_name == 'DD':
        nnodes_syn = 285
    elif args.dataset_name == 'MUTAG':
        nnodes_syn = 18
    elif args.dataset_name == 'NCI1':
        nnodes_syn = 30
    elif args.dataset_name == 'ogbg-molhiv':
        nnodes_syn = 26
    elif args.dataset_name == 'ogbg-molbbbp':
        nnodes_syn = 24
    elif args.dataset_name == 'ogbg-molbace':
        nnodes_syn = 34
    else:
        raise NotImplementedError

    GD_Agent = Agent(packed_data, args, device, nnodes_syn)
    train_loader = GD_Agent.train()


compare_data = [train_loader, val_loader, test_loader, num_feats, num_classes]

compare_Agent = compare(args, device, compare_data)