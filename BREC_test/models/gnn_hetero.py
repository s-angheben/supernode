import torch
from torch_geometric.nn import MLP, global_add_pool, HeteroConv, SimpleConv, GATConv, GINConv, GCNConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

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

def get_HGIN_simple(args, device, dropout=0.5, hidden_channels=32,
                   num_layers=4, out_channels=16):
    SConv = HeteroConv({
            ('normal', 'identity', 'normal'): SimpleConv('add'),
            ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
        })

    HConvs = torch.nn.ModuleList()
    for _ in range(num_layers):
        conv = HeteroConv({
           ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
           ('normal', 'orig', 'normal'): GINConv(MLP([-1, hidden_channels, hidden_channels])),
           ('supernodes', 'toNor', 'normal'): GINConv(MLP([-1, hidden_channels, hidden_channels])),
        }, aggr='sum')
        HConvs.append(conv)

    class HGIN_simple(torch.nn.Module):
        def __init__(self):
            super(HGIN_simple, self).__init__()
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


    model = HGIN_simple().to(device)
    return model

def get_HGIN_norm(args, device, dropout=0.5, hidden_channels=64,
                   num_layers=4, out_channels=16):
    SConv = HeteroConv({
            ('normal', 'identity', 'normal'): SimpleConv('add'),
            ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
        })

    Conv1 = HeteroConv({
           ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
           ('normal', 'orig', 'normal'): GINConv(
                Sequential(Linear(1, hidden_channels),
                           BatchNorm1d(hidden_channels), ReLU(),
                           Linear(hidden_channels, hidden_channels), ReLU()
                           )),
           ('supernodes', 'toNor', 'normal'): GINConv(
                Sequential(Linear(1, hidden_channels),
                           BatchNorm1d(hidden_channels), ReLU(),
                           Linear(hidden_channels, hidden_channels), ReLU()
                           )),
        }, aggr='sum')

    HConvs = torch.nn.ModuleList()
    for _ in range(num_layers-1):
        conv = HeteroConv({
           ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
           ('normal', 'orig', 'normal'): GINConv(
                Sequential(Linear(hidden_channels, hidden_channels),
                           BatchNorm1d(hidden_channels), ReLU(),
                           Linear(hidden_channels, hidden_channels), ReLU()
                           )),
           ('supernodes', 'toNor', 'normal'): GINConv(
                Sequential(Linear(hidden_channels, hidden_channels),
                           BatchNorm1d(hidden_channels), ReLU(),
                           Linear(hidden_channels, hidden_channels), ReLU()
                           )),
        }, aggr='sum')
        HConvs.append(conv)

    class HGIN_norm(torch.nn.Module):
        def __init__(self):
            super(HGIN_norm, self).__init__()
            self.supinit = SConv
            self.conv1 = Conv1
            self.convs = HConvs
            self.readout = global_add_pool
            self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                                   norm="batch_norm", dropout=dropout)

        def forward(self, data):
            x_dict, edge_index_dict, batch_dict = (data.x_dict, data.edge_index_dict, data.collect('batch'))
            x_dict = self.supinit(x_dict, edge_index_dict)
            x_dict = self.conv1(x_dict, edge_index_dict)

            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}

            x_dict = {key: self.readout(x_dict[key], batch_dict[key]) for key in x_dict.keys()}
            x = torch.stack(tuple(x_dict.values()), dim=0).sum(dim=0)

            x = self.classifier(x)
            return x


    model = HGIN_norm().to(device)
    return model


def get_HGCN_simple(args, device, dropout=0.5, hidden_channels=64,
                   num_layers=4, out_channels=16):
    SConv = HeteroConv({
            ('normal', 'identity', 'normal'): SimpleConv('add'),
            ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
        })

    HConvs = torch.nn.ModuleList()
    for _ in range(num_layers):
        conv = HeteroConv({
           ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
           ('normal', 'orig', 'normal'): GCNConv(-1, hidden_channels, add_self_loops=True),
           ('supernodes', 'toNor', 'normal'): GCNConv(-1, hidden_channels, add_self_loops=False),
        }, aggr='sum')
        HConvs.append(conv)

    class HGCN_simple(torch.nn.Module):
        def __init__(self):
            super(HGCN_simple, self).__init__()
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


    model = HGCN_simple().to(device)
    return model
