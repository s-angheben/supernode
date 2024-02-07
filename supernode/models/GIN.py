import torch
from torch_geometric.nn import MLP, GINConv, SimpleConv, global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers):
        super().__init__()
        torch.manual_seed(12345)

        self.supconv = SimpleConv("add")
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, num_classes],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, supernode_mask, edge_mask, batch):
        if supernode_mask is not None:
            # 0. compute supernode initial features
            x2 = self.supconv(x, edge_index, edge_mask)
            x[supernode_mask] = x2[supernode_mask]

        # 1. Obtain node embeddings
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        # 2. Readout layer
        # batch-wise graph-level-outputs by adding node features across the node dimension
        x = global_add_pool(x, batch)

        # 3. Apply a final classifier
        return self.mlp(x)
