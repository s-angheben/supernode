import torch
from torch_geometric.nn import GIN, GAT, MLP, global_add_pool

def get_GIN_normal(args, device, dropout=0.5, hidden_channels=64,
                   num_layers=4, out_channels=16):
    in_channels = 1

    class GNN(torch.nn.Module):
        def __init__(self):
            super(GNN, self).__init__()
            self.gnn = GIN(in_channels, hidden_channels, num_layers,
                        dropout=dropout, jk='cat')
            self.readout = global_add_pool
            self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                        norm="batch_norm", dropout=dropout)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.gnn(x, edge_index)
            x = self.readout(x, batch)
            x = self.classifier(x)
            return x

    model = GNN().to(device)
    return model

def get_GAT_normal(args, device, dropout=0.5, hidden_channels=64,
                   num_layers=4, out_channels=16):
    in_channels = 1

    class GNN(torch.nn.Module):
        def __init__(self):
            super(GNN, self).__init__()
            self.gnn = GAT(in_channels, hidden_channels, num_layers,
                        dropout=dropout, jk='cat')
            self.readout = global_add_pool
            self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                        norm="batch_norm", dropout=dropout)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.gnn(x, edge_index)
            x = self.readout(x, batch)
            x = self.classifier(x)
            return x

    model = GNN().to(device)
    return model
