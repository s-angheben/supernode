import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, SimpleConv
from torch_geometric.nn import global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GNN, self).__init__()
        torch.manual_seed(12345)

        # neighborhood normalization decreases the expressivity of GNNs in
        # distinguishing certain graph structures.
        # adds a simple skip-connection to the GNN layer in order to preserve central node information
        #  x_i_{l+1} = W_1_{l+1} * x_i_l +  W_2{l+1} * SUM_{j=N(i)} x_l_j
        self.supconv = SimpleConv("add")
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, supernode_mask, edge_mask, batch):
        if supernode_mask is not None:
            # 0. compute supernode initial features
            x2 = self.supconv(x, edge_index, edge_mask)
            x[supernode_mask] = x2[supernode_mask]

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
