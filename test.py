# import wandb
#
# import argparse
# import random
# import numpy as np
# import torch
# from torch.autograd import Variable
# from torch.optim.lr_scheduler import StepLR
#
# from dataset import basicDataset
# from model import GCN
#
# def mixup_cross_entropy_loss(input, target, size_average=True):
#     """Origin: https://github.com/moskomule/mixup.pytorch
#     in PyTorch's cross entropy, targets are expected to be labels
#     so to predict probabilities this loss is needed
#     suppose q is the target and p is the input
#     loss(p, q) = -\sum_i q_i \log p_i
#     """
#     assert input.size() == target.size()
#     assert isinstance(input, Variable) and isinstance(target, Variable)
#     loss = - torch.sum(input * target)
#     return loss / input.size()[0] if size_average else loss
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--gMixup', type=bool, default=True, help='whether to use gmixup')
# parser.add_argument('--gpu', type=str, default='cuda', help='cuda or cpu')
# parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
# parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--data_path', type=str, default='./')
# parser.add_argument('--dataset_name', type=str, default='DD')
# parser.add_argument('--loader_bs', type=int, default=128, help='batch size for origin data in dataloader')
# #mixup parameters
# parser.add_argument('--mixup_fixed', type=int, default=0.5, help='use fixed mixup rate 0.5 or set it 0')
# parser.add_argument('--lam_range', type=str, default="[0.005, 0.01]")
# parser.add_argument('--aug_ratio', type=float, default=0.15)
# parser.add_argument('--aug_num', type=int, default=10)
# parser.add_argument('--batch_size', type=int, default=32)
# #distillation parameters
# parser.add_argument('--epochs', type=int, default=500)
# parser.add_argument('--init', type=str, default='real', help='initialize the synthetic data with real data')
# parser.add_argument('--stru_discrete', type=int, default=1, help='create discrete structure in synthetic data')
# parser.add_argument('--lr_adj', type=float, default=0.0001)
# parser.add_argument('--lr_feat', type=float, default=0.0001)
# parser.add_argument('--lr_model', type=float, default=0.0005)
# parser.add_argument('--ipc', type=int, default=0, help=' set the number of condensed samples per class artificially')
# parser.add_argument('--reduction_rate', type=float, default=0.1, help='if ipc=0, this param  will be enabled. set the number of condensed samples by the dataset')
# parser.add_argument('--pooling', type=str, default='sum', help='mean or sum')
# parser.add_argument('--net_norm', type=str, default='batchnorm', help='batchnorm, layernorm, instancenorm, groupnorm, none')
# parser.add_argument('--hidden', type=int, default=64)
# parser.add_argument('--nconvs', type=int, default=3)
# parser.add_argument('--outer', type=int, default=1)
# parser.add_argument('--inner', type=int, default=0)
# parser.add_argument('--dis_metric', type=str, default='mse', help='distance metric')
# parser.add_argument('--beta', type=float, default=0.1, help='coefficient for the regularization term')
# parser.add_argument('--save', type=int, default=0, help='whether to save the condensed graphs')
#
# args = parser.parse_args()
#
#
# device = args.gpu
# if device == 'cuda':
#     torch.cuda.set_device(args.gpu_id)
#
# print(args)
#
#
# # load original data
# data = basicDataset(args)
# [trainset, origin_TrainLoader, origin_ValLoader, origin_TestLoader] = data.packed_data
#
#
# num_features = trainset[0].x.shape[1]
# num_classes = trainset[0].y.shape[0]
#
# model_real = GCN(nfeat=num_features, nhid=args.hidden,
#                      net_norm=args.net_norm, pooling=args.pooling,
#                      dropout=0.5, nclass=num_classes,
#                      nconvs=args.nconvs, args=args).to(device)
#
# optimizer_real = torch.optim.Adam(model_real.parameters(), lr=args.lr_model)
# scheduler = StepLR(optimizer_real, step_size=100, gamma=0.5)
#
#
# for epoch in range(args.epochs):
#     model_real.train()
#     train_loss = 0
#     graph_all = 0
#     for data in origin_TrainLoader:
#         data = data.to(device)
#         optimizer_real.zero_grad()
#         output = model_real(data)
#         y = data.y.view(-1, num_classes)
#         loss = mixup_cross_entropy_loss(output, y)
#         loss.backward()
#         train_loss += loss.item() * data.num_graphs
#         graph_all += data.num_graphs
#         optimizer_real.step()
#
#     train_loss  /= graph_all
#
#     model_real.eval()
#     correct = 0
#     total = 0
#     val_loss = 0
#     for data in origin_ValLoader:
#         data = data.to(device)
#         output = model_real(data)
#         pred = output.max(dim=1)[1]
#         y = data.y.view(-1, num_classes)
#         val_loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
#         y = y.max(dim=1)[1]
#         correct += pred.eq(y).sum().item()
#         total += data.num_graphs
#     acc = correct / total
#     loss = val_loss / total
#
#     scheduler.step()
#
#     print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}, Val Acc: {:.5f}'.format(epoch, train_loss, val_loss, acc))
#
#     if args.save and epoch % 50 == 0:
#         torch.save(model_real.state_dict(), './model_real_{}.pth'.format(epoch))
