import torch
import torch.nn.functional as F
import lightning as L
import torchmetrics

class S_GNN(L.LightningModule):
    def __init__(self, convs, Sconv, readout, classifier, out_channels, num_layers):
        super(S_GNN, self).__init__()
        self.num_layers = num_layers

        self.convs = convs
        self.Sconv = Sconv
        self.classifier = classifier
        self.readout = readout

        self.loss_fn = F.cross_entropy

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_channels)
        self.auroc = torchmetrics.AUROC(task='multiclass', num_classes=out_channels)

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

    def training_step(self, data, batch_idx):
        loss, y_hat, y = self._step(data, batch_idx)
        accuracy = self.accuracy(y_hat.softmax(dim=-1), y)
        auroc = self.auroc(y_hat.softmax(dim=-1), y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_auroc': auroc},
                      prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        loss, y_hat, y = self._step(data, batch_idx)
        accuracy = self.accuracy(y_hat.softmax(dim=-1), y)
        auroc = self.auroc(y_hat.softmax(dim=-1), y)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_auroc': auroc},
                      prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
        return loss

    def test_step(self, data, batch_idx):
        loss, y_hat, y = self._step(data, batch_idx)
        accuracy = self.accuracy(y_hat.softmax(dim=-1), y)
        auroc = self.auroc(y_hat.softmax(dim=-1), y)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_auroc': auroc},
                      prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
        return loss

    def _step(self, data, batch_idx):
        y_hat = self.forward(data)
        loss = self.loss_fn(y_hat, data.y)
        return loss, y_hat, data.y

    def predict_step(self, data, batch_idx, dataloader_idx):
        y_hat = self.forward(data)
        preds = torch.argmax(y_hat, dim=-1)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

