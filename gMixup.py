import random
import numpy as np

from utils import split_class_graph, stat_graph, align_graphs
from utils import universal_svd, two_graphons_mixup




class MixupDataset:
    def __init__(self, packed_data, args):

        [dataset, train_nums, train_val_nums] = packed_data
        lam_range = eval(args.lam_range)

        avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, \
        median_num_edges, median_density = stat_graph(dataset[:train_nums])


        print('Train set: avg_num_nodes: {:.2f}, avg_num_edges: {:.2f}, avg_density: {:.2f}, '
              'median_num_nodes: {:.2f}, median_num_edges: {:.2f}, median_density: {:.2f}'.format(
            avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density))

        resolution = int(median_num_nodes)

        class_graphs = split_class_graph(dataset[:train_nums])
        graphons = []

        for label, graphs in class_graphs:
            print('label: {}, num_graphs: {}'.format(label, len(graphs)))
            align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
                graphs, padding=True, N=resolution)
            print(f"aligned graph: {align_graphs_list[0].shape}")

            #the align_graphs_list is a list with many array, every array is a two dimension array with dtype=float32.
            graphon = universal_svd(align_graphs_list, threshold=0.2)
            graphons.append((label, graphon))

        for label, graphon in graphons:
            print(f"graphon info: label: {label}; mean: {graphon.mean()} shape: {graphon.shape}")

        num_sample = int(train_nums * args.aug_ratio / args.aug_num)
        # changeless mixup or not
        if args.mixup_fixed:
            lam_list = [args.mixup_fixed] * args.aug_num
        else:
            lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(args.aug_num,))

        random.seed(args.seed)
        new_graph = []


        for lam in lam_list:
            print(f"lam: {lam}")
            print(f"num_sample: {num_sample}")
            two_graphon = random.sample(graphons, 2)
            new_graph += two_graphons_mixup(two_graphon, la=lam, num_sample=num_sample)
            print(f"label {new_graph[-1].y}")

        dataset = new_graph + dataset

        num_mixup = len(new_graph)
        train_nums = train_nums + num_mixup
        train_val_nums = train_val_nums + num_mixup

        self.mixup_data = [dataset, train_nums, train_val_nums]

        # trainset = prepare_dataset_x(trainset)
        #
        # print(f"num_features: {trainset[0].x.shape}")
        # print(f"num_classes: {trainset[0].y.shape}")
        #
        # num_features = trainset[-1].x.shape[1]
        # num_classes = trainset[-1].y.shape[0]
        #
        # random.shuffle(trainset)

        # self.mixup_trainset = [trainset, num_features, num_classes]
