from typing import List, Optional, Union
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

class IdentityConv(MessagePassing):
    def __init__(self, **kwargs):
        super(IdentityConv, self).__init__(aggr='add', **kwargs)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def update(self, aggr_out):
        return aggr_out

