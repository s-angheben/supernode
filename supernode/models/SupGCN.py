import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SimpleConv
from torch_geometric.nn import global_mean_pool

class SupGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(SupGCN, self).__init__()
        torch.manual_seed(12345)

        #  x_i_{l+1} = W_{l+1} * SUM_{j=N(i)} e_{i,j}/sqrt(d_i, d_j) * x_l_j
        # W[i=num_output_features, j=num_input_features]
        # d_i = 1+SUM{j=N(i) e_{i, j}}
        self.supconv = SimpleConv("add")
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, supernode_mask, edge_mask, batch):
        # 1. compute supernode initial features
        x2 = self.supconv(x, edge_index, edge_mask)
        x[supernode_mask] = x2[supernode_mask]

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 3. Readout layer
        # averaging node features across the node dimension
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

