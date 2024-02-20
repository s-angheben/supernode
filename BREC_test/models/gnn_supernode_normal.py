import torch
from torch_geometric.nn import SimpleConv, GINConv, MLP, global_add_pool, GATConv


# GIN for normal nodes and add for supernodes
def get_GIN_Sadd(args, device, dropout=0.5, hidden_channels=64,
                             num_layers=4, out_channels=16):

    class GIN_Sadd(torch.nn.Module):
        def __init__(self):
            super(GIN_Sadd, self).__init__()
            self.num_layers = num_layers

            self.SInit = SimpleConv('add')
            self.convs = torch.nn.ModuleList()
            self.Sconv = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(GINConv(MLP([-1, hidden_channels, hidden_channels]),
                                          train_eps=True))
                self.Sconv.append(SimpleConv('add'))

            self.readout = global_add_pool
            self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                                   norm="batch_norm", dropout=dropout)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            supernode_mask = data.S > 0
            supernode_edge_mask = data.edge_S > 0
            normalnode_mask = data.S <= 0

            ## initialize supernode values
            x2 = self.SInit(x, edge_index, supernode_edge_mask)
            x[supernode_mask] = x2[supernode_mask]

            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x2 = self.Sconv[i](x, edge_index, supernode_edge_mask)
                x[supernode_mask] = x2[supernode_mask]

            x = self.readout(x, batch)
            x = self.classifier(x)

            return x

    model = GIN_Sadd().to(device)
    return model

# GIN for normal nodes and GIN for supernodes
def get_GIN_SGIN(args, device, dropout=0.5, hidden_channels=64,
                             num_layers=4, out_channels=16):

    class GIN_SGIN(torch.nn.Module):
        def __init__(self):
            super(GIN_SGIN, self).__init__()
            self.num_layers = num_layers

            self.SInit = SimpleConv('add')
            self.convs = torch.nn.ModuleList()
            self.Sconv = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(GINConv(MLP([-1, hidden_channels, hidden_channels]),
                                          train_eps=True))
                self.Sconv.append(GINConv(MLP([-1, hidden_channels, hidden_channels]),
                                          train_eps=False))

            self.readout = global_add_pool
            self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                                   norm="batch_norm", dropout=dropout)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            supernode_mask = data.S > 0
            supernode_edge_mask = data.edge_S > 0
            normalnode_mask = data.S <= 0

            ## initialize supernode values
            x2 = self.SInit(x, edge_index, supernode_edge_mask)
            x[supernode_mask] = x2[supernode_mask]

            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x2 = self.Sconv[i](x, edge_index)
                x[supernode_mask] = x2[supernode_mask]

            x = self.readout(x, batch)
            x = self.classifier(x)

            return x

    model = GIN_SGIN().to(device)
    return model

def get_GIN_SGIN_noSINIT(args, device, dropout=0.5, hidden_channels=64,
                             num_layers=4, out_channels=16):

    class GIN_SGIN_noSINIT(torch.nn.Module):
        def __init__(self):
            super(GIN_SGIN_noSINIT, self).__init__()
            self.num_layers = num_layers

            self.convs = torch.nn.ModuleList()
            self.Sconv = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(GINConv(MLP([-1, hidden_channels, hidden_channels]),
                                          train_eps=True))
                self.Sconv.append(GINConv(MLP([-1, hidden_channels, hidden_channels]),
                                          train_eps=False))

            self.readout = global_add_pool
            self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                                   norm="batch_norm", dropout=dropout)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            supernode_mask = data.S > 0
            supernode_edge_mask = data.edge_S > 0
            normalnode_mask = data.S <= 0

            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x2 = self.Sconv[i](x, edge_index)
                x[supernode_mask] = x2[supernode_mask]

            x = self.readout(x, batch)
            x = self.classifier(x)

            return x

    model = GIN_SGIN_noSINIT().to(device)
    return model

# GAT for normal nodes and add for supernodes
def get_GAT_Sadd(args, device, dropout=0.5, hidden_channels=64,
                             num_layers=4, out_channels=16):

    class GAT_Sadd(torch.nn.Module):
        def __init__(self):
            super(GAT_Sadd, self).__init__()
            self.num_layers = num_layers

            self.SInit = SimpleConv('add')
            self.convs = torch.nn.ModuleList()
            self.Sconv = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(GATConv((-1, -1), hidden_channels, add_self_loops=True))
                self.Sconv.append(SimpleConv('add'))

            self.readout = global_add_pool
            self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                                   norm="batch_norm", dropout=dropout)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            supernode_mask = data.S > 0
            supernode_edge_mask = data.edge_S > 0
            normalnode_mask = data.S <= 0

            ## initialize supernode values
            x2 = self.SInit(x, edge_index, supernode_edge_mask)
            x[supernode_mask] = x2[supernode_mask]

            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x2 = self.Sconv[i](x, edge_index, supernode_edge_mask)
                x[supernode_mask] = x2[supernode_mask]

            x = self.readout(x, batch)
            x = self.classifier(x)

            return x

    model = GAT_Sadd().to(device)
    return model
