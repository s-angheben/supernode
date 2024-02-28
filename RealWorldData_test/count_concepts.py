import torch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import MoleculeNet, TUDataset
from tqdm import tqdm

from data.concepts import *
import statistics

concept_list_dict = {
    "maxcliques":                   [ {"name": "GMC", "fun": max_cliques, "args": []} ],
    "cyclebasis":                   [ {"name": "GCB", "fun": cycle_basis, "args": []} ],
    "maxcliques_cyclebasis":        [ {"name": "GMC", "fun": max_cliques, "args": []},
                                      {"name": "GCB", "fun": cycle_basis, "args": []} ],
    "linepaths":                    [ {"name": "GLP", "fun": line_paths, "args": []} ],
    "k_edge_comp":                  [ {"name": "kecomp", "fun": k_edge_comp, "args": []} ],
    "constell":                     [ {"name": "const", "fun": stars_constellation, "args": []} ],
    "star2":                        [ {"name": "star2", "fun": star, "args": []} ],
    "maxcliques_cyclebasis_star2":  [ {"name": "GCB", "fun": cycle_basis, "args": []},
                                      {"name": "GMC", "fun": max_cliques, "args": []},
                                      {"name": "star2", "fun": star, "args": []} ],
    "maxlines":                     [ {"name": "maxline", "fun": max_lines, "args": []} ],
    "minlines":                     [ {"name": "minline", "fun": min_lines, "args": []} ],
    "k_core":                       [ {"name": "k_core", "fun": k_core, "args": []} ],
    "degree_centrality":            [ {"name": "deg_cent", "fun": degree_centrality, "args": []} ],
    "comm_modul":                   [ {"name": "comm_mod", "fun": comm_modularity, "args": []} ],
    "cycb_maxcliq_star2_minl_maxl": [ {"name": "GCB", "fun": cycle_basis, "args": []},
                                      {"name": "GMC", "fun": max_cliques, "args": []},
                                      {"name": "star2", "fun": star, "args": []},
                                      {"name": "minline", "fun": min_lines, "args": []},
                                      {"name": "maxline", "fun": max_lines, "args": []} ],
}

def squeeze_y(data):
    data.y = data.y.squeeze(1).long()
    return data

def count_stats(dataset, dataset_name):
    print("Dataset: ", dataset_name)
    n_nodes = [data.num_nodes for data in dataset]
    n_edges = [data.num_edges for data in dataset]
    node_degree = [data.num_edges / data.num_nodes for data in dataset]
    print("\t {0:20} N_nodes_mean={1:>10.4f}\tN_edges={2:>10.4f}\taverage_node_degree={3:>10.4f}".format(
        "HIV", statistics.mean(n_nodes), statistics.mean(n_edges), statistics.mean(node_degree)))

    for name, concepts in concept_list_dict.items():
        print("Concepts: ", name)

        population = [
            sum(len(concept["fun"](to_networkx(data, to_undirected=True)), *concept["args"]) for concept in concepts)
            for data in dataset
        ]

        print("\tmean={0:>10.4f}\tvar={1:>10.4f}\tmax={2:>10.4f}\tmin={3:>10.4f}".format(
            statistics.mean(population), statistics.variance(population),
            max(population), min(population)
            ))

def main():
#    dataset = MoleculeNet("./dataset/Molecule_stat", name="HIV",
#                              pre_transform=squeeze_y)
#    count_stats(dataset, "HIV")
#
#    print("\n")
#
#    dataset = TUDataset("./dataset/TUDataset_stat", name="PROTEINS")
#    count_stats(dataset, "PROTEINS")
#
#    print("\n")

    dataset = TUDataset("./dataset/TUDataset_stat", name="IMDB-BINARY")
    count_stats(dataset, "PROTEINS")


if __name__ == "__main__":
    main()
