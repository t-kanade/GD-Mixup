from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Batch, Data

from model import GCN, DenseGCN
from utils import match_loss, mixup_cross_entropy_loss
from dataset import SparseTensorDataset

onehot_criterion = torch.nn.BCEWithLogitsLoss()

class GM_Agent:
    def __init__(self, packed_data, args, device, nnodes_syn):
        [trainset, valLoader, num_feats, num_classes] = packed_data
        self.data = trainset
        self.val = valLoader
        self.args = args
        self.device = device
        self.train_lenth = len(trainset)
        self.num_feats = num_feats
        self.num_classes = num_classes

        train_npList = np.ndarray((self.train_lenth,), dtype=object)
        for i in range(self.train_lenth):
            train_npList[i] = trainset[i]
        self.train_npList = train_npList
        """
        the numpy array can use integer array indexing, helping us to get random batches with index
        """

        #split the class in the trainset
        self.prepare_train_indices(if_mixup=args.gMixup)

        #get the lenth of each class
        self.real_len_class = {}
        self.syn_len_class = {}
        for key, value in self.real_indices_class.items():
            self.real_len_class[key] = len(value)
            self.syn_len_class[key] = int(self.real_len_class[key]*self.args.reduction_rate)

        #generate synthetic labels
        self.labels_syn = self.generate_syn_labels()
        self.labels_syn = self.labels_syn.to(self.device)
        """
        now we have several self variables:
        :self.real_indices_class: dict, key: classes in real trainset, value: list of indices, such as {0:[1,3,15,27], 1:[2,4,16,28], 2:[5,6,17,29]}
        :self.nnodes_all: np.array, number of nodes for each graph
        :self.real_len_class: dict, key: classes in real trainset, value: lenth of each class, means the numbers of data in this class, such as {0:4, 1:4, 2:4}
        :self.syn_len_class: dict, key: classes in synthetic trainset, value: lenth of each class, generate by self.real_len_class * self.args.reduction_rate, such as {0:2, 1:2, 2:2}
        :self.labels_syn: torch.tensor, synthetic labels, such as [[0,1,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        :self.syn_indices_class: dict, key: classes in synthetic trainset, value: list range of each class, such as {0:[0,2], 1:[2,4], 2:[4,6]}
        """

        #generate synthetic adjs and feats
        if args.gMixup == True:
            range_c = num_classes+1
        else:
            range_c = num_classes

        self.feats_syn = torch.rand(size=(self.labels_syn.shape[0], nnodes_syn, self.num_feats),
                                    dtype=torch.float, requires_grad=True, device=self.device)
        self.adjs_syn = torch.rand(size=(self.labels_syn.shape[0], nnodes_syn, nnodes_syn),
                                    dtype=torch.float, requires_grad=True, device=self.device)

        if args.init == 'random':
            if args.stru_discrete:
                adj_init = torch.log(self.adjs_syn) - torch.log(1 - self.adjs_syn)
                adj_init = adj_init.clamp(-10, 10)
                self.adjs_syn.data.copy_(adj_init)
        elif args.init == 'real':
            for c in range(range_c):
                ind = self.syn_indices_class[c]
                feat_real, adj_real = self.prepare_graphs(c, batch_size=ind[1] - ind[0],
                                                          max_node_size=nnodes_syn, to_dense=True)
                self.feats_syn.data[ind[0]: ind[1]] = feat_real[:, :nnodes_syn].detach().data
                self.adjs_syn.data[ind[0]: ind[1]] = adj_real[:, :nnodes_syn, :nnodes_syn].detach().data
            if args.stru_discrete:
                self.adjs_syn.data.copy_(self.adjs_syn * 10 - 5)  # max:5; min:-5

        else:
            raise NotImplementedError

        self.sparsity = self.adjs_syn.mean().item()

        print('adj.shape:', self.adjs_syn.shape, 'feat.shape:', self.feats_syn.shape)

        #define the optimizer of synthetic adjs and feats
        self.optimizer_adj = torch.optim.Adam([self.adjs_syn], lr=args.lr_adj)
        self.optimizer_feat = torch.optim.Adam([self.feats_syn], lr=args.lr_feat)
        self.weights = []



    def train(self):
        args = self.args

        if args.gMixup == True:
            range_c = self.num_classes + 1
        else:
            range_c = self.num_classes

        self.acc = 0

        #define the one-step model
        for it in range(args.distill_epochs):
            model_syn = DenseGCN(nfeat=self.num_feats, nhid=args.hidden,
                                 net_norm=args.net_norm, pooling=args.pooling,
                                 dropout=0.0, nclass=self.num_classes,
                                 nconvs=args.nconvs, args=args).to(self.device)
            model_real = GCN(nfeat=self.num_feats, nhid=args.hidden,
                             net_norm=args.net_norm, pooling=args.pooling,
                             dropout=0.0, nclass=self.num_classes,
                             nconvs=args.nconvs, args=args).to(self.device)

            model_real.load_state_dict(model_syn.state_dict())
            model_real_parameters = list(model_real.parameters())
            model_syn_parameters = list(model_syn.parameters())
            # optimizer = torch.optim.Adam(model_syn.parameters(), lr=args.lr_model)

            loss_avg = 0
            for ol in range(args.outer):
                # outer loop is used to optimize synthetic graph
                feats_syn = self.feats_syn
                adjs_syn = self.adjs_syn

                if args.stru_discrete:
                    adjs_syn = self.get_discrete_graphs(adjs_syn, inference=False)

                loss = 0
                # 定义损失函数，先只考虑DD数据集，其他的后面再管
                # 需要：1. real graph的数据 2. synthetic graph的数据
                # 具体来说，需要len(real graph) // batch_size ,然后写for循环。
                for c in range(range_c):
                    #the data_real is a torch_geometric.data.Batch object
                    #the data_real.y is a [64] tensor because of the batch_size=32 * class=2
                    data_real = self.prepare_graphs(c, batch_size=args.syn_bs)
                    ind = self.syn_indices_class[c]
                    feat_syn_c = feats_syn[ind[0]:ind[1]]
                    adj_syn_c = adjs_syn[ind[0]: ind[1]]

                    # the data_real.y is a (num_graphs, num_classes) tensor, so the shape[0] is the number of graphs
                    # labels_real = torch.reshape(data_real.y, (-1, 2))
                    labels_real = data_real.y.view(-1, self.num_classes)
                    labels_real = labels_real.to(self.device)

                    labels_syn = self.labels_syn[ind[0]:ind[1]]
                    output_real = model_real(data_real)
                    # be onehot output


                    loss_real = onehot_criterion(output_real, labels_real)
                    gw_real = torch.autograd.grad(loss_real, model_real_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = model_syn(feat_syn_c, adj_syn_c)
                    loss_syn = onehot_criterion(output_syn, labels_syn)
                    gw_syn = torch.autograd.grad(loss_syn, model_syn_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args, self.device)

                loss_reg = F.relu(torch.sigmoid(self.adjs_syn).mean() - self.sparsity)

                loss_avg += loss.item()
                loss = loss + self.args.beta * loss_reg

                self.optimizer_adj.zero_grad()
                self.optimizer_feat.zero_grad()

                loss.backward()

                self.optimizer_adj.step()
                self.optimizer_feat.step()
                if not self.args.stru_discrete:
                    self.clip()
            
            if it % 20 == 0:
                print('Condensation - Iter:', it, 'loss:', loss_avg)
                print('sparsity loss', loss_reg.item())

            #test the synthetic dataset
            if (it+1) % 100 == 0:
                #firstly, we should trans syn data to a list[Data(edge_index=[2, xxx], x=[xxx, 89], y=[2])]
                feats_syn = self.feats_syn.detach()
                adjs_syn = self.adjs_syn.detach()
                if args.stru_discrete:
                    adjs_syn = self.get_discrete_graphs(adjs_syn, inference=True)

                #convert adjs_syn to edge_index
                sample = np.ndarray((adjs_syn.size(0),), dtype=np.object)
                for i in range(adjs_syn.size(0)):
                    x = feats_syn[i]
                    adj = adjs_syn[i]
                    g = adj.nonzero().T
                    y = self.labels_syn[i]
                    single_data = Data(x=x, edge_index=g, y=y)
                    sample[i] = (single_data)

                #Then convert the syn data to Tensor and get a Loader
                syn_data = SparseTensorDataset(sample)

                from torch_geometric.loader import DataLoader
                train_loader = DataLoader(syn_data, batch_size=args.batch_size, shuffle=True)
                val_loader = self.val

                best_acc = self.test_syn(args, train_loader, val_loader, 500)

                if best_acc > self.acc:
                    self.acc = best_acc
                    self.best_syn_data = train_loader

            if it == 400:
                self.optimizer_adj = torch.optim.Adam([self.adjs_syn], lr=0.1*args.lr_adj) # optimizer for synthetic data
                self.optimizer_feat = torch.optim.Adam([self.feats_syn], lr=0.1*args.lr_feat) # optimizer for

        return self.best_syn_data






    def test_syn(self, args, train_loader, val_loader, epochs=50):
        model_test = GCN(nfeat=self.num_feats, nhid=args.hidden,
                        net_norm=args.net_norm, pooling=args.pooling,
                        dropout=0.0, nclass=self.num_classes,
                        nconvs=args.nconvs, args=args).to(self.device)

        optimizer = torch.optim.Adam(model_test.parameters(), lr=args.lr_model, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        best_val_acc = 0
        for epoch in range(epochs):
            #train test model
            model_test.train()
            loss_all = 0
            graph_all = 0
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                output = model_test(data)
                y = data.y.view(-1, self.num_classes)
                loss = mixup_cross_entropy_loss(output, y)
                loss.backward()
                loss_all += loss.item() * data.num_graphs
                graph_all += data.num_graphs
                optimizer.step()
            train_loss = loss_all / graph_all

            model_test.eval()
            correct = 0
            total = 0
            loss = 0
            for data in val_loader:
                data = data.to(self.device)
                output = model_test(data)
                pred = output.max(dim=1)[1]
                y = data.y.view(-1, self.num_classes)
                loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
                y = y.max(dim=1)[1]
                correct += pred.eq(y).sum().item()
                total += data.num_graphs
            acc = correct / total
            val_loss = loss / total

            scheduler.step()

            if acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = acc

        print('Epoch: {:03d}, best Val Acc: {:.5f}'.format(best_epoch, best_val_acc))
        return best_val_acc






    def prepare_train_indices(self, if_mixup=False):
        '''
        prepare indices for each class, the original labels must have value 1.0
        To be specific, we prepare n+1 indices for n onehot class and a mixup class
        :self.real_indices_class: dict, key: class index, value: list of indices
        :self.nnodes_all: np.array, number of nodes for each graph
        '''

        dataset = self.data
        if if_mixup == True:
            indices_class = {self.num_classes: []}
        else:
            indices_class = {}

        nnodes_all = []
        labels = []
        for idx, graph in enumerate(dataset):
            if 1.0 in graph.y:
                index = torch.where(graph.y == 1.0)[0].item()
                if index not in indices_class:
                    indices_class[index] = [idx]
                else:
                    indices_class[index].append(idx)
            else:
                indices_class[self.num_classes].append(idx)
            nnodes_all.append(graph.num_nodes)

            labels.append(graph.y)

        self.nnodes_all = np.array(nnodes_all)
        self.real_indices_class = indices_class

    def generate_syn_labels(self):
        '''
        generate synthetic labels for each class. The indices are generated by self.syn_len_class
        :param labels_train:
        :return: labels_syn
        '''
        labels_syn = []
        self.syn_indices_class = {}
        temp = 0
        for key, value in self.syn_len_class.items():
            if key == self.num_classes:
                label = [float(1/self.num_classes)]*self.num_classes
            else:
                label = [0.]*self.num_classes
                label[key] = 1.
            labels_syn += [label.copy() for i in range(self.syn_len_class[key])]
            self.syn_indices_class[key] = [temp, temp+self.syn_len_class[key]]
            temp += self.syn_len_class[key]

        return torch.tensor(labels_syn)

    def prepare_graphs(self, c, batch_size, max_node_size=None, to_dense=False, idx_selected=None):
        '''
        shuffle the indices in class c and prepare batches with np.array
        :param c:
        :param batch_size:
        :param max_node_size:
        :param to_dense:
        :return:
        '''
        if idx_selected is None:
            if max_node_size is None:
                idx_shuffle = np.random.permutation(self.real_indices_class[c])[:batch_size]
                sampled = self.train_npList[idx_shuffle]
            else:
                indices = np.array(self.real_indices_class[c])[self.nnodes_all[self.real_indices_class[c]] <= max_node_size]
                idx_shuffle = np.random.permutation(indices)[:batch_size]
                sampled = self.train_npList[idx_shuffle]
        else:
            sampled = self.train_npList[idx_selected]
        data = Batch.from_data_list(sampled)
        if to_dense:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x, mask = to_dense_batch(x, batch=batch, max_num_nodes=max_node_size)
            adj = to_dense_adj(edge_index, batch=batch, max_num_nodes=max_node_size)
            return x.to(self.device), adj.to(self.device)
        else:
            # the data is a batch_size DataBatch
            return data.to(self.device)

    def get_discrete_graphs(self, adj, inference):
        # hasattr(self, ‘cnt’) checks if the object self has an attribute named cnt.
        # If the object has the attribute, the function returns True
        if not hasattr(self, 'cnt'):
            self.cnt = 0

        if self.args.dataset_name not in ['CIFAR10']:
            adj = (adj.transpose(1,2) + adj) / 2

        if not inference:
            N = adj.size()[1]
            vals = torch.rand(adj.size(0) * N * (N+1) // 2)
            vals = vals.view(adj.size(0), -1).to(self.device)
            i, j = torch.triu_indices(N, N)
            epsilon = torch.zeros_like(adj)
            epsilon[:, i, j] = vals
            epsilon.transpose(1,2)[:, i, j] = vals

            tmp = torch.log(epsilon) - torch.log(1-epsilon)
            self.tmp = tmp
            adj = tmp + adj
            t0 = 1
            tt = 0.01
            end_iter = 200
            t = t0*(tt/t0)**(self.cnt/end_iter)
            if self.cnt == end_iter:
                print('===reached the end of anealing...')
            self.cnt += 1

            t = max(t, tt)
            adj = torch.sigmoid(adj/t)
            adj = adj * (1-torch.eye(adj.size(1)).to(self.device))
        else:
            adj = torch.sigmoid(adj)
            adj = adj * (1-torch.eye(adj.size(1)).to(self.device))
            adj[adj> 0.5] = 1
            adj[adj<= 0.5] = 0
        return adj

    def clip(self):
        # clip the weights
        self.adjs_syn.data.clamp_(min=0, max=1)



