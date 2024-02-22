from BRECDataset_v3 import BRECDataset
import numpy as np
import torch
import torch_geometric
import torch_geometric.loader
from multiprocessing import Process

import hashlib
import os.path as osp
import os

from concepts.concepts import *
from concepts.transformations import AddSupernodes, AddSupernodesHeteroMulti

CHUNK_SIZE = 5000
DATASET_LEN = 51200

def makefeatures(data):
    data.x = torch.ones((data.num_nodes, 1))
    return data


def create_concept_dataset_normal(dataset, concept, name):
    if not osp.exists(f'./Data/{name}'):
        os.makedirs(f'./Data/{name}')
        for i in range(len(dataset) // CHUNK_SIZE + 1):
            start_idx = i * CHUNK_SIZE
            end_idx = min((i + 1) * CHUNK_SIZE, DATASET_LEN)
            chunk = dataset[start_idx : end_idx]
            transformed_dataset = [AddSupernodes(concept)(data) for data in chunk]
            torch.save(
                transformed_dataset,
                f'./Data/{name}/transformed_dataset_chunk_{i}.pth',
            )

def create_concept_dataset_multi(dataset, concept, name):
    if not osp.exists(f'./Data/{name}'):
        os.makedirs(f'./Data/{name}')
        for i in range(len(dataset) // CHUNK_SIZE + 1):
            start_idx = i * CHUNK_SIZE
            end_idx = min((i + 1) * CHUNK_SIZE, DATASET_LEN)
            chunk = dataset[start_idx : end_idx]
            transformed_dataset = [AddSupernodesHeteroMulti(concept)(data) for data in chunk]
            torch.save(
                transformed_dataset,
                f'./Data/{name}/transformed_dataset_chunk_{i}.pth',
            )

def main():
    dataset = BRECDataset(
            dataset_path="/home/sam/Documents/network/supernode/dataset/BREC_raw",
            name="BREC_vanilla",
            pre_transform=makefeatures
            )

    concepts_list = [
#          ( "cycle_basis", [ {"name": "GCB", "fun": cycle_basis, "args": []}            ]),
#          ( "max_cliques", [ {"name": "GMC", "fun": max_cliques, "args": []}            ]),
#          ( "line_pahts" , [ {"name": "GLP2", "fun": line_paths, "args": []}            ]),
#          ( "k_edge_comp", [ {"name": "kecomp", "fun": k_edge_comp, "args": []}         ]),
#          ( "constell",    [ {"name": "const", "fun": stars_constellation, "args": []}  ]),
#          ( "cyclebasis_and_maxcliques", [ {"name": "GCB", "fun": cycle_basis, "args": []},
#                                           {"name": "GMC", "fun": max_cliques, "args": []} ]),

#          ( "maxnode_lines", [{"name": "maxline", "fun": max_lines, "args": []}]  ),
#          ( "minnode_lines", [{"name": "minline", "fun": min_lines, "args": []}]  ),
#          ( "k_core", [{"name": "k_core", "fun": k_core, "args": []}]  ),
#          ( "degree_centrality", [{"name": "deg_cent", "fun": degree_centrality, "args": []}]  ),
#          ( "comm_modul", [{"name": "comm_mod", "fun": comm_modularity, "args": []}]  ),
#          ( "star2",        [{"name": "star2", "fun": star, "args": []}]                ),
          ( "cycb_maxcliq_star2", [ {"name": "GCB", "fun": cycle_basis, "args": []},
                                    {"name": "GMC", "fun": max_cliques, "args": []},
                                    {"name": "star2", "fun": star, "args": []}
                                  ]),
#          ( "cycb_maxcliq_star2_minl_maxl", [ {"name": "GCB", "fun": cycle_basis, "args": []},
#                                              {"name": "GMC", "fun": max_cliques, "args": []},
#                                              {"name": "star2", "fun": star, "args": []}
#                                            ]),
        ]

    procs = []
    for name, concept in concepts_list:
        name_transf = f"TBREC_supernode_normal_precalc_{name}"
        proc = Process(target=create_concept_dataset_normal, args=(dataset, concept, name_transf))

#        name_transf = f"TBREC_supernode_multi_precalc_{name}"
#        proc = Process(target=create_concept_dataset_multi, args=(dataset, concept, name_transf))

        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
