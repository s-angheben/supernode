import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.utils import from_networkx, to_networkx
import torch
from typing import List
from distutils.dir_util import copy_tree
import json
import os.path as osp
import concepts
import utils

class Dataset_tree_cycle(Dataset):
    def __init__(self, root, dataset_path, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.dataset_path = dataset_path
        with open(osp.join(dataset_path, "info.json")) as f:
            info_data = json.load(f)
            self.size         = info_data["size"]
            self.cycle_graphs = info_data["cycle_graphs"]
            self.tree_graphs  = info_data["tree_graphs"]
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)

    @property
    def raw_file_names(self) -> List[str]:
        return [str(i) for i in range(self.size)]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{i}.pt' for i in range(self.size)]

    def download(self) -> None:
       copy_tree(self.dataset_path, self.raw_dir)

    def process(self) -> None:
        for raw_file_name in self.raw_file_names:
            raw_file_path = osp.join(self.raw_dir, raw_file_name)
            G = nx.read_gml(raw_file_path)
            data = from_networkx(G)
            data.y = data.y.unsqueeze(0)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'{raw_file_name}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'{idx}.pt'))
        return data


# mantains the original dataset tree/cycle proportion
def create_dataloader(dataset, batch_size=10, train_prop=0.7, test_prop=0.2, val_prop=0.1):
    if train_prop + test_prop + val_prop != 1:
        raise Exception("probabilities doesn't sumup to 1")

    cycle_graphs = dataset[dataset.cycle_graphs[0]: dataset.cycle_graphs[1]]
    tree_graphs  = dataset[dataset.tree_graphs[0]: dataset.tree_graphs[1]]

    cycle_graphs = cycle_graphs.shuffle()
    tree_graphs  = tree_graphs.shuffle()

    part1c_end = int(len(cycle_graphs) * train_prop)
    part1t_end = int(len(tree_graphs) * train_prop)

    part2c_end = int(len(cycle_graphs) * (train_prop + test_prop))
    part2t_end = int(len(tree_graphs) * (train_prop + test_prop))

    train_dataset = cycle_graphs[:part1c_end] + tree_graphs[:part1t_end]
    test_dataset  = cycle_graphs[part1c_end:part2c_end] + tree_graphs[part1t_end:part2t_end]
    val_dataset   = cycle_graphs[part2c_end:] + tree_graphs[part2t_end:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


@functional_transform('add_supernodes')
class AddSupernodes(BaseTransform):
    def __init__(self, concepts_list) -> None:
        self.concepts_list = concepts_list

    def forward(self, data: Data) -> Data:
        G = to_networkx(data, to_undirected=True, node_attrs=["x"], graph_attrs=["y"])

        # find all the concepts in the graph on the original graph only
        for concept in self.concepts_list:
            concept["concepts_nodes"] = concept["fun"](G, *concept["args"])

        for concept in self.concepts_list:
            add_supernode_normal(G, concept["concepts_nodes"], [1.0])

        data_with_supernodes = from_networkx(G)
        return data_with_supernodes

concepts_list_ex = [
        {"name": "GCB",  "fun": cycle_basis, "args": []},
        {"name": "GMC",  "fun": max_cliques, "args": []},
        {"name": "GLP2", "fun": line_paths,  "args": [2]}
    ]
