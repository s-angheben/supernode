import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData
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
        G.add_edges_from(supernode_edges(supernode_name, concept), S=[1.0])

@functional_transform('add_supernodes')
class AddSupernodes(BaseTransform):
    def __init__(self, concepts_list) -> None:
        self.concepts_list = concepts_list

    def forward(self, data: Data) -> Data:

        supernode_feature_default = [0.0] * data.num_features
        G = to_networkx(data, to_undirected=True, node_attrs=["x"], graph_attrs=["y"])
        nx.set_node_attributes(G, "ORIG", "ntype")
#        nx.set_node_attributes(G, [0], "N")
        nx.set_node_attributes(G, 0, "S")
        nx.set_edge_attributes(G, [1.0], "S")

        # find all the concepts in the graph on the original graph only
        for concept in self.concepts_list:
            concept["concepts_nodes"] = concept["fun"](G, *concept["args"])

        for concept in self.concepts_list:
            if "features" in concept:
                supernode_feature = concept["features"]
            else:
                supernode_feature = supernode_feature_default
            add_supernode_normal(G, concept["concepts_nodes"], concept["name"], supernode_feature)

        data_with_supernodes = from_networkx(G)
        return data_with_supernodes

concepts_list_ex = [
        {"name": "GCB",  "fun": cycle_basis, "args": []},
        {"name": "GMC",  "fun": max_cliques, "args": []},
        {"name": "GLP2", "fun": line_paths,  "args": [2]}
    ]


def add_supernode(G, concepts, cname, features):
    for concept in concepts:
        supernode_name = G.number_of_nodes()
#        G.add_node(supernode_name, x=features, ntype=cname, N=concept, S=1)
        G.add_node(supernode_name, x=features, ntype=cname, S=1)
        G.add_edges_from(supernode_edges(supernode_name, concept), S=[1.0])

@functional_transform('add_supernodes_hetero')
class AddSupernodesHetero(BaseTransform):
    def __init__(self, concepts_list) -> None:
        self.concepts_list = concepts_list

    def forward(self, data: Data) -> HeteroData:
        data_with_supernodes = HeteroData({
            'normal'    : {'x' : data.x.float()},
            ('normal', 'orig', 'normal'  )   : { 'edge_index': data.edge_index, 'edge_attr' : data.edge_attr},
        })
        t1 = torch.arange(data.x.shape[0])
        data_with_supernodes['normal', 'identity', 'normal'].edge_index = torch.stack([t1, t1], dim=0).long()

        G = to_networkx(data, to_undirected=True, node_attrs=["x"])

        found_concepts = []
        # find all the concepts in the graph on the original graph only
        for concept in self.concepts_list:
            comp = concept["fun"](G, *concept["args"])
            if len(comp) != 0:
                found_concepts = found_concepts + comp

        current_supernode = 0
        from_normal = []
        to_sup      = []
        supnodes    = []
        for concept in found_concepts:
            supnodes.append(current_supernode)
            for node in concept:
                from_normal.append(node)
                to_sup.append(current_supernode)
            current_supernode += 1

        if current_supernode != 0:
            toSup_edges = torch.Tensor((from_normal, to_sup)).long()
            toNor_edges = torch.Tensor((to_sup, from_normal)).long()
            data_with_supernodes['supernodes'].x = torch.ones(len(found_concepts), data.num_features)
            #data_with_supernodes['supernodes'].x = torch.zeros(len(found_concepts), data.num_features)
            data_with_supernodes['normal', 'toSup', 'supernodes'].edge_index = toSup_edges
            data_with_supernodes['supernodes', 'toNor', 'normal'].edge_index = toNor_edges

            t2 = torch.arange(len(found_concepts))
            data_with_supernodes['supernodes', 'identity', 'supernodes'].edge_index = torch.stack([t2, t2], dim=0).long()

        else:
            data_with_supernodes['supernodes'].x = torch.zeros(1, data.num_features)

        return data_with_supernodes

@functional_transform('add_supernodes_hetero_multi')
class AddSupernodesHeteroMulti(BaseTransform):
    def __init__(self, concepts_list) -> None:
        self.concepts_list = concepts_list

    def forward(self, data: Data) -> HeteroData:
        data_with_supernodes = HeteroData({
            'normal'    : {'x' : data.x.float()},
            ('normal', 'orig', 'normal'  )   : { 'edge_index': data.edge_index, 'edge_attr' : data.edge_attr},
#            ('normal', 'orig', 'normal'  )   : { 'edge_index': data.edge_index},
        })
        t1 = torch.arange(data.x.shape[0])
        data_with_supernodes['normal', 'identity', 'normal'].edge_index = torch.stack([t1, t1], dim=0).long()

        G = to_networkx(data, to_undirected=True, node_attrs=["x"])

        # find all the concepts in the graph on the original graph only
        for concept in self.concepts_list:
            concept_name = concept["name"]
            comp = concept["fun"](G, *concept["args"])
            if len(comp) != 0:
                current_supernode = 0
                from_normal = []
                to_sup      = []
                supnodes    = []
                for concept in comp:
                    supnodes.append(current_supernode)
                    for node in concept:
                        from_normal.append(node)
                        to_sup.append(current_supernode)
                    current_supernode += 1

                toSup_edges = torch.Tensor((from_normal, to_sup)).long()
                toNor_edges = torch.Tensor((to_sup, from_normal)).long()
                #data_with_supernodes[concept_name].x = torch.zeros(len(comp), data.num_features)
                data_with_supernodes[concept_name].x = torch.ones(len(comp), data.num_features)
                data_with_supernodes['normal', 'toSup', concept_name].edge_index = toSup_edges
                data_with_supernodes[concept_name, 'toNor', 'normal'].edge_index = toNor_edges
                t2 = torch.arange(len(comp))
                data_with_supernodes[concept_name, 'identity', concept_name].edge_index = torch.stack([t2, t2], dim=0).long()
            else:
                data_with_supernodes[concept_name].x = torch.zeros(1, data.num_features)

        return data_with_supernodes

def my_to_networkx(data):
    normalnodes    = list(range(data['normal'].x.shape[0]))
    normal_ei      = data['normal', 'orig', 'normal'].edge_index
    norm_to_sup_ei = data['normal', 'toSup', 'supernodes'].edge_index
    supernodes = list(range(data['normal'].x.shape[0], data['normal'].x.shape[0]+data['supernodes'].x.shape[0]))
    mapping = dict(list(zip(list(range(data['supernodes'].x.shape[0])), supernodes)))

    G = nx.Graph()
    G.add_nodes_from(normalnodes)
    G.add_edges_from(list(zip(normal_ei[0].tolist(), normal_ei[1].tolist())))
    G.add_nodes_from(supernodes)
    G.add_edges_from(list(zip(norm_to_sup_ei[0].tolist(), list(map(lambda key: mapping[key], norm_to_sup_ei[1].tolist())))))

    dictA = {**{key: 'ORIG' for key in normalnodes}, **{key: 'SUP' for key in supernodes}}
    nx.set_node_attributes(G, dictA, "ntype")

    return G

def my_to_networkx_multi(data):
    normalnodes    = list(range(data['normal'].x.shape[0]))
    normal_ei      = data['normal', 'orig', 'normal'].edge_index

    nodes_type = list(data.x_dict.keys())
    nodes_type.remove('normal')
    print(nodes_type)

    curr_num = data['normal'].x.shape[0]
    tensor_list = []
    supernodes_list = []
    for node_type in nodes_type:
        norm_to_sup_ei = data['normal', 'toSup', node_type].edge_index.clone()
        next_num = curr_num+data[node_type].x.shape[0]
        supernodes = torch.arange(curr_num, next_num)
        supernodes_list.append(supernodes)

        for i in range(len(supernodes)):
            t2 = norm_to_sup_ei[1]
            t2[t2 == i] = supernodes[i]

        tensor_list.append(norm_to_sup_ei)
        curr_num = next_num


    norm_to_sup_ei = torch.cat(tensor_list, dim=1)
    supernodes     = torch.cat(supernodes_list, dim=0)


    G = nx.Graph()
    G.add_nodes_from(normalnodes)
    G.add_edges_from(list(zip(normal_ei[0].tolist(), normal_ei[1].tolist())))
    G.add_nodes_from(supernodes.tolist())
    G.add_edges_from(list(zip(norm_to_sup_ei[0].tolist(), norm_to_sup_ei[1].tolist())))

    dictA = {**{key: 'ORIG' for key in normalnodes}, **{key: 'SUP' for key in supernodes.tolist()}}
    nx.set_node_attributes(G, dictA, "ntype")
    return G
