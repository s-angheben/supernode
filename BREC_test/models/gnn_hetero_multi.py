import torch
from torch_geometric.nn import MLP, global_add_pool, HeteroConv, SimpleConv, GATConv, GINConv
from torch_geometric.nn import HGTConv, Linear

def get_HGAT_multi_simple(args, device, supnodes_name, dropout=0.5, hidden_channels=64,
                   num_layers=4, out_channels=16):
    SConv_dict = {
            ('normal', 'identity', 'normal'): SimpleConv('add'),
            }
    for supnode_type in supnodes_name:
        SConv_dict |= {('normal', 'toSup', supnode_type) : SimpleConv('add')}


    SConv = HeteroConv(SConv_dict, aggr='sum')

    HConvs = torch.nn.ModuleList()
    for _ in range(num_layers):
        Conv_dict = {("normal", "orig", "normal") : GATConv((-1, -1), hidden_channels, add_self_loops=True)}

        for supnode_type in supnodes_name:
            Conv_dict |= {("normal", "toSup", supnode_type) : SimpleConv('add'),
                          (supnode_type, "toNor", "normal") : GATConv((-1, -1), hidden_channels, add_self_loops=False)}

        conv = HeteroConv(Conv_dict, aggr='sum')
        HConvs.append(conv)

    class HGAT_simple_multi(torch.nn.Module):
        def __init__(self):
            super(HGAT_simple_multi, self).__init__()
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


    model = HGAT_simple_multi().to(device)
    return model

def get_HGIN_multi_simple(args, device, supnodes_name, dropout=0.5, hidden_channels=64,
                   num_layers=4, out_channels=16):
    SConv_dict = {
            ('normal', 'identity', 'normal'): SimpleConv('add'),
            }
    for supnode_type in supnodes_name:
        SConv_dict |= {('normal', 'toSup', supnode_type) : SimpleConv('add')}


    SConv = HeteroConv(SConv_dict, aggr='sum')

    HConvs = torch.nn.ModuleList()
    for _ in range(num_layers):
        Conv_dict = {("normal", "orig", "normal") :  GINConv(MLP([-1, hidden_channels, hidden_channels]))}

        for supnode_type in supnodes_name:
            Conv_dict |= {("normal", "toSup", supnode_type) : SimpleConv('add'),
                          (supnode_type, "toNor", "normal") : GINConv(MLP([-1, hidden_channels, hidden_channels]))}

        conv = HeteroConv(Conv_dict, aggr='sum')
        HConvs.append(conv)

    class HGIN_simple_multi(torch.nn.Module):
        def __init__(self):
            super(HGIN_simple_multi, self).__init__()
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

    model = HGIN_simple_multi().to(device)
    return model

def get_HGIN_multi_all(args, device, supnodes_name, dropout=0.5, hidden_channels=64,
                   num_layers=4, out_channels=16):
    SConv_dict = {
            ('normal', 'identity', 'normal'): SimpleConv('add'),
            }
    for supnode_type in supnodes_name:
        SConv_dict |= {('normal', 'toSup', supnode_type) : SimpleConv('add')}


    SConv = HeteroConv(SConv_dict, aggr='sum')

    HConvs = torch.nn.ModuleList()
    for _ in range(num_layers):
        Conv_dict = {("normal", "orig", "normal") :  GINConv(MLP([-1, hidden_channels, hidden_channels]))}

        for supnode_type in supnodes_name:
            Conv_dict |= {("normal", "toSup", supnode_type) : GINConv(MLP([-1, hidden_channels, hidden_channels])),
                          (supnode_type, "toNor", "normal") : GINConv(MLP([-1, hidden_channels, hidden_channels]))}

        conv = HeteroConv(Conv_dict, aggr='sum')
        HConvs.append(conv)

    class HGIN_simple_multi(torch.nn.Module):
        def __init__(self):
            super(HGIN_simple_multi, self).__init__()
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

    model = HGIN_simple_multi().to(device)
    return model

def get_HGT_multi(args, device, data, supnodes_name, hidden_channels=64, num_layers=4,
                  dropout=0.5, num_heads=4, out_channels=16):

    class HGT_multi(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
            super().__init__()

            self.lin_dict = torch.nn.ModuleDict()
            for node_type in data.node_types:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                               num_heads, group='sum')
                self.convs.append(conv)

            self.readout = global_add_pool
            self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                                   norm="batch_norm", dropout=dropout)

        def forward(self, data):
            x_dict, edge_index_dict, batch_dict = (data.x_dict, data.edge_index_dict, data.collect('batch'))

            x_dict = {
                node_type: self.lin_dict[node_type](x).relu_()
                for node_type, x in x_dict.items()
            }

            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)

            x_dict = {key: self.readout(x_dict[key], batch_dict[key]) for key in x_dict.keys()}
            x = torch.stack(tuple(x_dict.values()), dim=0).sum(dim=0)

            x = self.classifier(x)
            return x

    model = HGT_multi(hidden_channels, out_channels, num_heads, num_layers).to(device)
    return model
