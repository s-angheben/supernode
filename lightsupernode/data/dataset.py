import networkx as nx
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.utils import from_networkx
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
from typing import List
from distutils.dir_util import copy_tree
import json
import os.path as osp
import lightning as L
from .dataloader import create_dataloader

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



class Dataset_tree_cycle_module(L.LightningDataModule):
    def __init__(self, root, dataset_path,
                 transform=None, pre_transform=None, pre_filter=None, force_reload=False,
                 batch_size=64, num_workers=4, train_prop=0.6, test_prop=0.2, val_prop=0.2):
        super().__init__()
        self.root = root
        self.dataset_path = dataset_path
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.force_reload = force_reload
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = val_prop

    def setup(self, stage=None):
        self.dataset = Dataset_tree_cycle_Memory(self.root, self.dataset_path,
                                                 self.transform, self.pre_transform,
                                                 self.pre_filter, self.force_reload)

        self.train, self.test, self.val = create_dataloader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            train_prop=self.train_prop, test_prop=self.test_prop, val_prop=self.val_prop
            )

    def train_dataloader(self):
        return self.train

    def test_dataloader(self):
        return self.test

    def val_dataloader(self):
        return self.val



class MutagDataModule(L.LightningDataModule):
    def __init__(self, data_dir, transform=None, pre_transform=None, pre_filter=None,
                 batch_size=64, num_workers=4,
                 train_prop=0.6, test_prop=0.2, val_prop=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = val_prop


    def setup(self, stage=None):
        self.dataset = TUDataset(self.data_dir, name="MUTAG",
                                 transform=self.transform,
                                 pre_transform=self.pre_transform,
                                 pre_filter=self.pre_filter)
        self.dataset = self.dataset.shuffle()

    def train_dataloader(self):
        return DataLoader(self.dataset[:self.train_prop], self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset[self.train_prop:self.train_prop+self.val_prop],
                          self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset[self.train_prop+self.val_prop:],
                          self.batch_size, num_workers=self.num_workers)




