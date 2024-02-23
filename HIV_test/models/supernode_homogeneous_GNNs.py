from torch_geometric.nn import MLP, GINConv
from torch_geometric.nn import global_add_pool
from .supernode_homogeneous_GNN_template import S_GNN
import torch

def get_SGIN_model(in_channels: int, out_channels: int,
                 hidden_channels: int = 32, num_layers: int = 3,
                 dropout: float = 0.5):
    convs = torch.nn.ModuleList()
    Sconv = torch.nn.ModuleList()
    for _ in range(num_layers):
        convs.append(GINConv(MLP([-1, hidden_channels, hidden_channels]),
                                              train_eps=True))
        Sconv.append(GINConv(MLP([-1, hidden_channels, hidden_channels]),
                                              train_eps=False))

    readout = global_add_pool
    classifier = MLP([hidden_channels, hidden_channels, out_channels],
                                   norm="batch_norm", dropout=dropout)
    model = S_GNN(convs, Sconv, readout, classifier, out_channels, num_layers)
    model_log = {"model": "S_GNN", "type": "supernode_homogeneous_GNNs",
                 "hidden_channels": hidden_channels,
                 "num_layers": num_layers, "dropout": dropout}

    return model, model_log
