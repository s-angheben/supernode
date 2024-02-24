from torch_geometric.nn import MLP, GINConv, HeteroConv, SimpleConv
from torch_geometric.nn import global_add_pool
from .supernode_hetero_multi_GNN_template import S_GNN_multi
import torch

def get_SHGIN_multi(in_channels: int, out_channels: int,
                    supnodes_name, hidden_channels: int = 32,
                    num_layers: int = 3, dropout: float = 0.5):
    Sinit_dict = {
            ('normal', 'identity', 'normal'): SimpleConv('add'),
            }

    for supnode_type in supnodes_name:
        Sinit_dict |= {('normal', 'toSup', supnode_type) : SimpleConv('add')}

    Sinit = HeteroConv(Sinit_dict, aggr='sum')

    HConvs = torch.nn.ModuleList()
    for _ in range(num_layers):
        Conv_dict = {("normal", "orig", "normal") :  GINConv(MLP([-1, hidden_channels, hidden_channels]))}
        for supnode_type in supnodes_name:
            Conv_dict |= {("normal", "toSup", supnode_type) : GINConv(MLP([-1, hidden_channels, hidden_channels])), #SimpleConv('add'),
                          (supnode_type, "toNor", "normal") : GINConv(MLP([-1, hidden_channels, hidden_channels]))}
        conv = HeteroConv(Conv_dict, aggr='sum')
        HConvs.append(conv)

        Conv_dict = {("normal", "identity", "normal") : SimpleConv('add')}
        for supnode_type in supnodes_name:
            Conv_dict |= {("normal", "toSup", supnode_type) : SimpleConv('add')}
        conv = HeteroConv(Conv_dict, aggr='sum')
        HConvs.append(conv)

    readout = global_add_pool

    classifier = MLP([hidden_channels, hidden_channels, out_channels], norm="batch_norm", dropout=dropout)

    model = S_GNN_multi(Sinit, HConvs, readout, classifier, out_channels, num_layers)
    model_log = {"model": "S_GIN_multi", "type": "supernode_heterogeneous_GNNs", "hidden_channels": hidden_channels, "num_layers": num_layers, "dropout": dropout}
    return model, model_log
