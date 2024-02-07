from torch_geometric.nn import MLP, GCN, GIN, SimpleConv, HeteroConv, GraphConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from .MyConv import IdentityConv
from .H_GNN_template import H1_GNN
import torch

def get_HTEST_model(out_channels: int,
                    hidden_channels: int = 64, num_layers: int = 3,
                    dropout: float = 0.5):
        SConv = HeteroConv({
            ('normal', 'void', 'normal'): SimpleConv('add'),
            ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
        })

        HConvs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
               ('supernodes', 'toNor', 'normal'): GraphConv(-1, hidden_channels, add_self_loops=False),
#                ('normal', 'toSup', 'supernodes'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
               ('normal', 'toSup', 'supernodes'): SimpleConv('add'),
               ('normal', 'orig', 'normal'): GraphConv(-1, hidden_channels, add_self_loops=False),
#               ('normal', 'orig', 'normal'): GATConv((-1, -1), hidden_channels, add_self_loops=True),
            }, aggr='sum')
            HConvs.append(conv)
#        HConvs.append(SConv)

        readout = global_add_pool

        classifier = MLP([hidden_channels, hidden_channels, out_channels],
                        norm="batch_norm", dropout=dropout)

        model = H1_GNN(SConv, HConvs, readout, classifier, out_channels)
#        log = __log_model(model, "GCNS2",
#                          """
#                          Apply gnn on all nodes and then supconv
#                          classifier and readout on all the nodes.
#                          """
#                            )
        return model, "HTEST"

def __log_model(model, model_name: str, description: str):
    log = {
            "model": model_name,
            "description": description,
            "supconv": model.supconv.__class__.__name__ + "(" +
                "aggr=" + model.supconv.conv.aggr + ")",
            "gnn": model.gnn.__class__.__name__ + "(" +
                     "in_channels=" + model.gnn.in_channels.__str__() + "," +
                     "out_channels=" + model.gnn.out_channels.__str__() + "," +
                     "hidden_channels=" + str(model.gnn.hidden_channels) + "," +
                     "num_layer=" + model.gnn.num_layers.__str__() + "," +
                     "dropout=" + model.gnn.dropout.__str__() + "," +
                     "jk=" + model.gnn.jk.__str__() + ")",
           "classifier": model.classifier.__class__.__name__ + "(" +
               "in_channels=" + model.classifier.in_channels.__str__() + "," +
               "out_channels=" + model.classifier.out_channels.__str__() + "," +
               "channel_list=" + str(model.classifier.channel_list) + "," +
               "num_layers=" + str(model.classifier.num_layers) + "," +
               "norm=" + str(model.classifier.norms) + "," +
               "dropout=" + model.classifier.dropout.__str__() + ")",
           "readout": model.readout.__name__,
           "loss": model.loss_fn.__name__,
           "metric": model.accuracy.__class__.__name__,
           }
    return log

