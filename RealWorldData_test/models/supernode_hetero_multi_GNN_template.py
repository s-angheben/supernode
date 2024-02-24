import torch
import torch.nn.functional as F
import lightning as L
import torchmetrics

class S_GNN_multi(L.LightningModule):
    def __init__(self, supinit, convs, readout, classifier, out_channels, num_layers):
        super(S_GNN_multi, self).__init__()
        self.supinit = supinit
        self.num_layers = num_layers

        self.convs = convs
        self.readout = readout
        self.classifier = classifier

        self.loss_fn = F.cross_entropy

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_channels)

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

    def training_step(self, data, batch_idx):
        loss, y_hat, y = self._step(data, batch_idx)
        accuracy = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy},
                      prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        loss, y_hat, y = self._step(data, batch_idx)
        accuracy = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy},
                      prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
        return loss

    def test_step(self, data, batch_idx):
        loss, y_hat, y = self._step(data, batch_idx)
        accuracy = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy},
                      prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
        return loss

    def _step(self, data, batch_idx):
        y_hat = self.forward(data)
        loss = self.loss_fn(y_hat, data['label'].y)
        return loss, y_hat, data['label'].y

    def predict_step(self, data, batch_idx, dataloader_idx):
        y_hat = self.forward(data)
        preds = torch.argmax(y_hat, dim=-1)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

