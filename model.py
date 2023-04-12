import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, DenseGCNConv, GINConv
from torch_geometric.nn import LayerNorm, InstanceNorm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nconvs=3, dropout=0, net_norm='none', pooling='mean', **kwargs):
        super(GCN, self).__init__()

        if nconvs ==1:
            nhid = nclass

        self.convs = nn.ModuleList([])
        self.convs.append(GCNConv(nfeat, nhid))
        for _ in range(nconvs-1):
            self.convs.append(GCNConv(nhid, nhid))

        self.norms = nn.ModuleList([])
        for _ in range(nconvs):
            if nconvs == 1:  norm = nn.Identity()
            elif net_norm == 'none':
                norm = nn.Identity()
            elif net_norm == 'batchnorm':
                norm = BatchNorm1d(nhid)
            elif net_norm == 'layernorm':
                norm = nn.LayerNorm([nhid], elementwise_affine=True)
            elif net_norm == 'instancenorm':
                norm = InstanceNorm(nhid, affine=False) #pyg
            elif net_norm == 'groupnorm':
                norm = nn.GroupNorm(4, nhid, affine=True)
            self.norms.append(norm)

        self.lin3 = Linear(nhid, nclass)
        self.dropout = dropout
        self.pooling = pooling


    def forward(self, data, if_embed=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.dropout !=0 and self.training:
            x_mask = torch.distributions.bernoulli.Bernoulli(self.dropout).sample([x.size(0)]).to('cuda').unsqueeze(-1)
            x = x_mask * x

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.perform_norm(i, x)
            x = F.relu(x)

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch=batch)
        if self.pooling == 'sum':
            x = global_add_pool(x, batch=batch)
        if if_embed:
            return x

        x = F.log_softmax(self.lin3(x), dim=-1)
        return x


    def perform_norm(self, i, x):
        batch_size, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = self.norms[i](x)
        x = x.view(batch_size, num_channels)
        return x

    def embed(self, data):
        return self.forward(data, if_embed=True)



class DenseGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nconvs=3, dropout=0, net_norm='none', pooling='mean', **kwargs):
        super(DenseGCN, self).__init__()

        if nconvs ==1:
            nhid = nclass

        self.convs = nn.ModuleList([])
        self.convs.append(DenseGCNConv(nfeat, nhid))
        for _ in range(nconvs-1):
            self.convs.append(DenseGCNConv(nhid, nhid))

        self.norms = nn.ModuleList([])
        for _ in range(nconvs):
            if nconvs == 1:  norm = nn.Identity()
            elif net_norm == 'none':
                norm = nn.Identity()
            elif net_norm == 'batchnorm':
                norm = BatchNorm1d(nhid)
            elif net_norm == 'layernorm':
                norm = nn.LayerNorm([nhid], elementwise_affine=True)
            elif net_norm == 'instancenorm':
                norm = InstanceNorm(nhid, affine=False) #pyg
            elif net_norm == 'groupnorm':
                norm = nn.GroupNorm(4, nhid, affine=True)
            self.norms.append(norm)

        self.lin3 = Linear(nhid, nclass)
        self.dropout = dropout
        self.pooling = pooling

    def forward(self, x, adj, mask=None, if_embed=False):
        if self.dropout !=0:
            x_mask = torch.distributions.bernoulli.Bernoulli(self.dropout).sample([x.size(0), x.size(1)]).to('cuda').unsqueeze(-1)
            x = x_mask * x

        for i in range(len(self.convs)):
            x = self.convs[i](x, adj, mask)
            x = self.perform_norm(i, x)
            x = F.relu(x)

        if self.pooling == 'sum':
            x = x.sum(1)
        if self.pooling == 'mean':
            x = x.mean(1)
        if if_embed:
            return x

        x = F.log_softmax(self.lin3(x), dim=-1)
        return x

    def embed(self, x, adj, mask=None):
        return self.forward(x, adj, mask, if_embed=True)

    def perform_norm(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = self.norms[i](x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x


class GIN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GIN, self).__init__()

        # if data.x is None:
        #   data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
        # dataset.data.edge_attr = None

        # num_features = dataset.num_features
        dim = num_hidden

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        # x = global_add_pool(x, batch)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)