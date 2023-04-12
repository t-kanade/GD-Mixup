import wandb

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from utils import mixup_cross_entropy_loss
from model import GCN, GIN

class compare:
    def __init__(self, args, device, data, epochs=50):
        [train_loader, val_loader, test_loader, num_feats, num_classes] = data
        self.device = device
        self.num_feats = num_feats
        self.num_classes = num_classes

        if args.model == 'GCN':
            model = GCN(nfeat=num_feats, nhid=args.hidden,
                        net_norm=args.net_norm, pooling=args.pooling,
                        dropout=0.0, nclass=num_classes,
                        nconvs=args.nconvs, args=args).to(device)
        elif args.model == 'GIN':
            model = GIN(num_features=num_feats, num_classes=num_classes, num_hidden=args.hidden).to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_model)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        # model_syn = GCN(nfeat=num_feats, nhid=args.hidden,
        #                 net_norm=args.net_norm, pooling=args.pooling,
        #                 dropout=0.0, nclass=num_classes,
        #                 nconvs=args.nconvs, args=args).to(device)
        # optimizer_syn = torch.optim.Adam(model_syn.parameters(), lr=args.lr_model)
        # scheduler_syn = StepLR(optimizer_syn, step_size=100, gamma=0.5)

        best_acc = 0
        best_loss = 0

        for epoch in range(epochs):
            model, train_loss = self.train(model, train_loader, optimizer)
            # print('Epoch: {:03d}, Train Loss: {:.5f}'.format(epoch, train_loss))



            val_acc, val_loss = self.test(model, val_loader)
            test_acc, test_loss = self.test(model, test_loader)
            scheduler.step()

            wandb.log({'train loss': train_loss, 'test acc': test_acc,
                       'test loss': test_loss})

            if test_acc > best_acc:
                best_acc = test_acc
                best_loss = test_loss


            # print('val acc: {:05f}, test acc: {:05f}'.format(val_acc, test_acc))

        print("Finally, it's the result:")
        print('Best acc: {:05f}, Best loss: {:05f}'.format(best_acc, best_loss))

        # wandb.log({'best_acc': best_acc, 'best_loss': best_loss, 'seed': args.seed})






    def train(self, model, train_loader, optimizer):
        model.train()
        loss_all = 0
        graph_all = 0
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            y = data.y.view(-1, self.num_classes)
            loss = mixup_cross_entropy_loss(output, y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            graph_all += data.num_graphs
            optimizer.step()
        train_loss = loss_all / graph_all

        return model, train_loss

    def test(self, model, loader):
        model.eval()
        correct = 0
        total = 0
        loss = 0
        for data in loader:
            data = data.to(self.device)
            output = model(data)
            pred = output.max(dim=1)[1]
            y = data.y.view(-1, self.num_classes)
            loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
            y = y.max(dim=1)[1]
            correct += pred.eq(y).sum().item()
            total += data.num_graphs
        acc = correct / total
        loss = loss / total
        return acc, loss