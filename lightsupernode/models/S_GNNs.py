from torch_geometric.nn import MLP, SimpleConv, GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from .S_module import SupConv
from .S_GNN_template import S_GNN

def get_custom_model(gnn, description, in_channels: int, out_channels: int,
                    hidden_channels: int = 64,
                    dropout: float = 0.5):

        classifier = MLP([hidden_channels, hidden_channels, out_channels],
                        norm="batch_norm", dropout=dropout)
        readout = global_add_pool

        model = S_GNN(gnn, classifier, readout, out_channels)
        log = __log_model(model, gnn, "GCNCustom", description)
        return model, log


def __log_model(model, gnn, model_name: str, description: str):
    gnn_string = '-'.join([conv['name'] for conv in gnn])
    log = {
            "model": model_name,
            "description": description,
            "gnn": gnn_string,
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
