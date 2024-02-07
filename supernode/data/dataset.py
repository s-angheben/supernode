import networkx as nx
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.utils import from_networkx
import torch
from typing import List
from distutils.dir_util import copy_tree
import json
import os.path as osp

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


class Dataset_tree_cycle_Memory(InMemoryDataset):
    def __init__(self, root, dataset_path, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.dataset_path = dataset_path
        with open(osp.join(dataset_path, "info.json")) as f:
            info_data = json.load(f)
            self.size         = info_data["size"]
            self.cycle_graphs = info_data["cycle_graphs"]
            self.tree_graphs  = info_data["tree_graphs"]
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [str(i) for i in range(self.size)]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self) -> None:
       copy_tree(self.dataset_path, self.raw_dir)

    def process(self) -> None:
        data_list = []

        for raw_file_name in self.raw_file_names:
            raw_file_path = osp.join(self.raw_dir, raw_file_name)
            G = nx.read_gml(raw_file_path)
            data = from_networkx(G)
            data.y = data.y.unsqueeze(0)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


