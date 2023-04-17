



class Meta_Agent:
    def __init__(self, packed_data, args, device, nnodes_syn):
        [trainset, valLoader, num_feats, num_classes] = packed_data