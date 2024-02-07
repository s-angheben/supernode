import networkx as nx
import random
import os
import json

#def _select_random_edge_from_cycle(cycle):
#    idx = random.randrange(len(cycle))
#    if idx == len(cycle)-1:
#        return (cycle[idx], cycle[0])
#    else:
#        return (cycle[idx], cycle[idx+1])
#
#def remove_cycles(G):
#    current_cycle_basis = cycle_basis(G)
#    while len(current_cycle_basis) != 0:
#        cycle = current_cycle_basis[random.randrange(len(current_cycle_basis))]
#        (u, v) = _select_random_edge_from_cycle(cycle)
#        G.remove_edge(u, v)
#        current_cycle_basis = cycle_basis(G)

def add_trivial_node_feature(G):
    for node in G.nodes():
        G.nodes[node]["x"] = [1.0]

def add_random_edges(G, n):
    nodes = list(G.nodes())
    nodes_num = len(nodes) - 1
    new_edges = []
    for _ in range(n):
        u = nodes[random.randrange(nodes_num)]
        v = nodes[random.randrange(nodes_num)]
        new_edges.append((u,v))

    G.add_edges_from(new_edges)

# create a dataset to check the loop
def create_dataset_tree_cycle(parent_dir, dataset_name, graph_num, cycle_proportion=0.5, node_num=80, cycle_level=10):
    path = os.path.join(parent_dir, dataset_name)
    os.mkdir(path)
    num_cycle_graph = int(graph_num * cycle_proportion)
    for cycle_graph in range(num_cycle_graph):
        graph_path = os.path.join(path, str(cycle_graph))
        G = nx.random_tree(node_num)
        add_trivial_node_feature(G)
        G.graph['y'] = 1                            # cycle
        add_random_edges(G, cycle_level)
        nx.write_gml(G, graph_path)

    for tree in range(num_cycle_graph, graph_num):
        graph_path = os.path.join(path, str(tree))
        G = nx.random_tree(node_num)
        add_trivial_node_feature(G)
        G.graph['y'] = 0                            # nocycle
        nx.write_gml(G, graph_path)

    info_data = {
            "size"        : graph_num,
            "cycle_graphs": (0, num_cycle_graph),
            "tree_graphs" : (num_cycle_graph, graph_num),
            }
    info_path = os.path.join(path, "info.json")
    with open(info_path, 'w') as f:
        json.dump(info_data, f)

