import torch
from torch_geometric.nn import MLP, global_add_pool, HeteroConv, SimpleConv, GATConv

def get_HGAT_simple(args, device, dropout=0.5, hidden_channels=64,
                   num_layers=4, out_channels=16):
    SConv = HeteroConv({
            ('normal', 'identity', 'normal'): SimpleConv('add'),
            ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
        })

    HConvs = torch.nn.ModuleList()
    for _ in range(num_layers):
        conv = HeteroConv({
           ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
           ('normal', 'orig', 'normal'): GATConv((-1, -1), hidden_channels, add_self_loops=True),
           ('supernodes', 'toNor', 'normal'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
        }, aggr='sum')
        HConvs.append(conv)

    class HGAT_simple(torch.nn.Module):
        def __init__(self):
            super(HGAT_simple, self).__init__()
            self.supinit = SConv
            self.convs = HConvs
            self.readout = global_add_pool
            self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                                   norm="batch_norm", dropout=dropout)

        def forward(self, data):
            x_dict, edge_index_dict, batch_dict = (data.x_dict, data.edge_index_dict, data.collect('batch'))
            x_dict = self.supinit(x_dict, edge_index_dict)

            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}

            x_dict = {key: self.readout(x_dict[key], batch_dict[key]) for key in x_dict.keys()}
            x = torch.stack(tuple(x_dict.values()), dim=0).sum(dim=0)

            x = self.classifier(x)
            return x


    model = HGAT_simple().to(device)
    return model

