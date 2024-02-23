import torch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import MoleculeNet
from tqdm import tqdm

from data.concepts import *
import statistics

def squeeze_y(data):
    data.y = data.y.squeeze(1).long()
    return data

def main():
    normal_data_dir = "./dataset/Molecule_normal"
    dataset = MoleculeNet(normal_data_dir, name="HIV",
                              pre_transform=squeeze_y)

    concepts_list = [
          ( "cycle_basis",  {"name": "GCB", "fun": cycle_basis, "args": []}            ),
          ( "max_cliques",  {"name": "GMC", "fun": max_cliques, "args": []}            ),
          ( "line_pahts" ,  {"name": "GLP2", "fun": line_paths, "args": []}            ),
          ( "k_edge_comp",  {"name": "kecomp", "fun": k_edge_comp, "args": []}         ),
          ( "k_comp",       {"name": "kcomp", "fun": k_comp, "args": []}               ),
          ( "star",         {"name": "star3", "fun": star, "args": []}                ),
          ( "constell",     {"name": "const", "fun": stars_constellation, "args": []}  ),
          ( "maxnode_lines", {"name": "maxline", "fun": max_lines, "args": []}  ),
          ( "minnode_lines", {"name": "minline", "fun": min_lines, "args": []}  ),
          ( "k_core",       {"name": "kcore", "fun": k_core, "args": []}  ),
          ( "degree_centrality",{"name": "dcentr", "fun": degree_centrality, "args": []}  ),
          ( "comm_modularity",{"name": "com_mod", "fun": comm_modularity, "args": []}  ),
        ]

    for name, concept in concepts_list:
        print("Concept: ", name)
        total_concept_num = 0
        total_graph_num   = 0

        for part_name, part_range in part_dict.items():
            part_concept_num = 0
            part_graph_num   = 0
            population = []
            n_nodes = []
            n_edges = []

            for id in tqdm(range(part_range[0], part_range[1])):
                dataset_traintest = dataset[
                    id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
                ]
                dataset_reliability = dataset[
                    (id + SAMPLE_NUM)
                    * NUM_RELABEL
                    * 2 : (id + SAMPLE_NUM + 1)
                    * NUM_RELABEL
                    * 2
                ]

                population += [len(concept["fun"](to_networkx(data, to_undirected=True)), *concept["args"]) for data in dataset_traintest]
#                n_nodes = [data.num_nodes for data in dataset_traintest]
#                n_edges = [data.num_edges for data in dataset_traintest]
#                node_degree = [data.num_edges / data.num_nodes for data in dataset_traintest]



            print("\t {0:20} mean={1:>10.4f}\tvar={2:>10.4f}\tmax={3}\tmin={4}".format(part_name, statistics.mean(population), statistics.variance(population), max(population), min(population)))
#            print("\t {0:20} N_nodes_mean={1:>10.4f}\tN_edges={2:>10.4f}\tnode_degree={3}".format(part_name, statistics.mean(n_nodes), statistics.mean(n_edges), statistics.mean(node_degree)))


if __name__ == "__main__":
    main()
