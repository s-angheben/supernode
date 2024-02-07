import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.utils import from_networkx, to_networkx
from .concepts import *

def supernode_edges(supernode, concept):
    return [(supernode, n) for n in concept]

def add_supernode_normal(G, concepts, cname, features):
    for concept in concepts:
        supernode_name = G.number_of_nodes()
#        G.add_node(supernode_name, x=features, ntype=cname, N=concept, S=1)
        G.add_node(supernode_name, x=features, ntype=cname, S=1)
        G.add_edges_from(supernode_edges(supernode_name, concept), S=1)

@functional_transform('add_supernodes')
class AddSupernodes(BaseTransform):
    def __init__(self, concepts_list) -> None:
        self.concepts_list = concepts_list

    def forward(self, data: Data) -> Data:
        supernode_feature = [1.0] * data.num_features
        G = to_networkx(data, to_undirected=True, node_attrs=["x"], graph_attrs=["y"])
        nx.set_node_attributes(G, "ORIG", "ntype")
#        nx.set_node_attributes(G, [0], "N")
        nx.set_node_attributes(G, 0, "S")
        nx.set_edge_attributes(G, 0, "S")

        # find all the concepts in the graph on the original graph only
        for concept in self.concepts_list:
            concept["concepts_nodes"] = concept["fun"](G, *concept["args"])

        for concept in self.concepts_list:
            add_supernode_normal(G, concept["concepts_nodes"], concept["name"], supernode_feature)

        data_with_supernodes = from_networkx(G)
        return data_with_supernodes

concepts_list_ex = [
        {"name": "GCB",  "fun": cycle_basis, "args": []},
        {"name": "GMC",  "fun": max_cliques, "args": []},
        {"name": "GLP2", "fun": line_paths,  "args": [2]}
    ]
