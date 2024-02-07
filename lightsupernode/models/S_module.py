#from torch import nn
#import torch
#
## only edges with one supernode are used in the message passing and only supernode update
#class SupConv(nn.Module):
#    def __init__(self, conv):
#        super(SupConv, self).__init__()
#        self.conv = conv
#
#    def forward(self, x, edge_index, S, edge_S):
#        supernode_mask = S > 0
#        edge_weight = torch.where(edge_S > 0,
#                                  torch.tensor(1.0),
#                                  torch.tensor(0.0))
#        x2 = self.conv(x, edge_index, edge_S)
#        x[supernode_mask] = x2[supernode_mask]
#        return x
#
## only edges without supernode are used in the message passing and only normal node update
#class NormalConv(nn.Module):
#    def __init__(self, conv):
#        super(NormalConv, self).__init__()
#        self.conv = conv
#
#    def forward(self, x, edge_index, S, edge_S):
#        normalnode_mask = S <= 0
#        edge_weight = torch.where(edge_S <= 0,
#                                  torch.tensor(1).to(torch.float32),
#                                  torch.tensor(0).to(torch.float32))
#        x2 = self.conv(x, edge_index, edge_weight)
#        x[normalnode_mask] = x2[normalnode_mask]
#        return x
#
## all edge used in the message passing but only normal node update
#class HybridConv(nn.Module):
#    def __init__(self, conv):
#        super(HybridConv, self).__init__()
#        self.conv = conv
#
#    def forward(self, x, edge_index, S, edge_S):
#        normalnode_mask = S <= 0
#        x2 = self.conv(x, edge_index)
#        x[normalnode_mask] = x2[normalnode_mask]
#        return x
#
#class AllConv(nn.Module):
#    def __init__(self, conv):
#        super(AllConv, self).__init__()
#        self.conv = conv
#
#    def forward(self, x, edge_index, S, edge_S):
#        x = self.conv(x, edge_index)
#        return x
