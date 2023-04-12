import os.path as osp

import random
import numpy as np

import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from utils import prepare_dataset_onehot_y



class Complete(object):
    def __call__(self, data):
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)
        return data



class basicDataset:
    def __init__(self, args):
        # random seed setting
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        #Load dataset
        dataset_path = osp.join(args.data_path, args.dataset_name)


        if args.dataset_name in ['DD', 'MUTAG', 'NCI1', 'REDDIT-BINARY']:
            dataset = TUDataset(dataset_path, name=args.dataset_name, transform=T.Compose([Complete()]), use_node_attr=True)
            dataset.data.edge_attr = None
            train_nums = int(len(dataset) * 0.7)
            train_val_nums = int(len(dataset) * 0.8)

        dataset = list(dataset)

        for graph in dataset:
            graph.y = graph.y.view(-1)

        dataset = prepare_dataset_onehot_y(dataset)

        random.shuffle(dataset)



        # train_dataset = dataset[: train_nums]
        # random.shuffle(train_dataset)
        # val_dataset = dataset[train_nums: train_val_nums]
        # test_dataset = dataset[train_val_nums:]
        # nnodes = [x.num_nodes for x in train_dataset]
        # print('mean #nodes:', np.mean(nnodes), 'max #nodes:', np.max(nnodes))

        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # self.packed_data = [train_dataset, train_loader, val_loader, test_loader]
        self.packed_data = [dataset, train_nums, train_val_nums]



class SparseTensorDataset(basicDataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)