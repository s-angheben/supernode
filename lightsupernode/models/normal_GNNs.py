from torch_geometric.nn import MLP, GCN, GIN
from torch_geometric.nn import global_mean_pool, global_add_pool
from .normal_GNN_template import Normal_GNN

def get_GCNN_model(in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.5):

    gnn = GCN(in_channels, hidden_channels, num_layers,
                       dropout=dropout, jk='cat')
    classifier = MLP([hidden_channels, hidden_channels, out_channels],
                       norm="batch_norm", dropout=dropout)
    readout = global_add_pool

    model = Normal_GNN(gnn, classifier, readout, out_channels)
    log = __log_model(model, "GCNN", "Normal GNN")
    return model, log

def get_GINN_model(in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.5):
    gnn = GIN(in_channels, hidden_channels, num_layers,
              dropout=dropout, jk='cat')
    classifier = MLP([hidden_channels, hidden_channels, out_channels],
                              norm="batch_norm", dropout=dropout)
    readout = global_add_pool

    model = Normal_GNN(gnn, classifier, readout, out_channels)
    log = __log_model(model, "GINN", "Normal GIN")
    return model, log

def __log_model(model, model_name: str, description: str):
    log = {
            "model": model_name,
            "description": description,
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

