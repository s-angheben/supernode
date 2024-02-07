#import torch
#import torch.nn.functional as F
#import lightning as L
#import torchmetrics
#
#from .S_module import *
#from torch_geometric.nn import SimpleConv, GCNConv, GINConv
#
#class S_GNN(L.LightningModule):
#    def __init__(self, gnn, classifer, readout, out_channels):
#        super(S_GNN, self).__init__()
#
#        self.conv1 = AllConv(GCNConv(7, 64))
#        self.conv2 = SupConv(GCNConv(64, 64))
#
#        self.convs = torch.nn.ModuleList()
#        for conv in gnn:
#            self.convs.append(NormalConv(conv['conv']))
#        self.gnn = gnn
#        self.classifier = classifer
#        self.readout = readout
#
#        self.loss_fn = F.cross_entropy
#
#        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_channels)
#        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=out_channels)
#
#    def forward(self, x, edge_index, S, edge_S, batch):
#        x = self.conv1(x, edge_index, S, edge_S)
#        x = self.conv2(x, edge_index, S, edge_S)
#
#        x = self.readout(x, batch)
#        x = self.classifier(x)
#        return x
#
#    def training_step(self, data, batch_idx):
#        loss, y_hat, y = self._step(data, batch_idx)
#        accuracy = self.accuracy(y_hat.softmax(dim=-1), y)
#        f1_score = self.f1_score(y_hat.softmax(dim=-1), y)
#        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score},
#                      prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
#        return loss
#
#    def validation_step(self, data, batch_idx):
#        loss, y_hat, y = self._step(data, batch_idx)
#        accuracy = self.accuracy(y_hat.softmax(dim=-1), y)
#        f1_score = self.f1_score(y_hat.softmax(dim=-1), y)
#        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1_score': f1_score},
#                      prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
#        return loss
#
#    def test_step(self, data, batch_idx):
#        loss, y_hat, y = self._step(data, batch_idx)
#        accuracy = self.accuracy(y_hat.softmax(dim=-1), y)
#        f1_score = self.f1_score(y_hat.softmax(dim=-1), y)
#        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_f1_score': f1_score},
#                      prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
#        return loss
#
#    def _step(self, data, batch_idx):
#        y_hat = self.forward(data.x, data.edge_index, data.S, data.edge_S, data.batch)
#        loss = self.loss_fn(y_hat, data.y)
#        return loss, y_hat, data.y
#
#    def predict_step(self, data, batch_idx, dataloader_idx):
#        y_hat = self.forward(data.x, data.edge_index, data.S, data.edge_S, data.batch)
#        preds = torch.argmax(y_hat, dim=-1)
#        return preds
#
#    def configure_optimizers(self):
#        return torch.optim.Adam(self.parameters(), lr=0.01)
#