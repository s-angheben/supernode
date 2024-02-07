from torch_geometric.nn import MLP, GCN, GIN, SimpleConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from .S1_GNN_template import S1_GNN

def get_GCNS1_model(in_channels: int, out_channels: int,
                    hidden_channels: int = 64, num_layers: int = 3,
                    dropout: float = 0.5):
        supconv = SimpleConv("add")
        gnn = GCN(in_channels, hidden_channels, num_layers,
                        dropout=dropout, jk='cat')
        classifier = MLP([hidden_channels, hidden_channels, out_channels],
                        norm="batch_norm", dropout=dropout)
        readout = global_add_pool

        model = S1_GNN(supconv, gnn, classifier, readout, out_channels)
        log = __log_model(model, "GCNS1",
                          """
                          Apply supconv first on supernode and then gnn,
                          classifier and readout on all the nodes.
                          """
                            )
        return model, log

def get_GINS1_model(in_channels: int, out_channels: int,
                    hidden_channels: int = 64, num_layers: int = 3,
                    dropout: float = 0.5):
        supconv = SimpleConv("add")
        gnn = GIN(in_channels, hidden_channels, num_layers,
                        dropout=dropout, jk='cat')
        classifier = MLP([hidden_channels, hidden_channels, out_channels],
                        norm="batch_norm", dropout=dropout)
        readout = global_add_pool

        model = S1_GNN(supconv, gnn, classifier, readout, out_channels)
        log = __log_model(model, "GINS1",
                          """
                          Apply supconv first on supernode and then gnn,
                          classifier and readout on all the nodes.
                          """
                            )
        return model, log

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

