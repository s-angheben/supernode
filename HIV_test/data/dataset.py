from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
import torch
import lightning as L
import hashlib
import os.path as osp
import os
from .transformation import AddSupernodes, AddSupernodesHeteroMulti
from tqdm import tqdm
import numpy as np

def squeeze_y(data):
    data.y = data.y.squeeze(1).long()
    return data

CHUNK_SIZE = 5000

class MoleculeHIVNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir,
                 batch_size=64, num_workers=4,
                 train_prop=0.6, test_prop=0.2, val_prop=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = val_prop


    def setup(self, stage=None):
        self.dataset = MoleculeNet(self.data_dir, name="HIV",
                                 pre_transform=squeeze_y)
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


class MoleculeHIVNetDataModule_supernode_homogenous(L.LightningDataModule):
    def __init__(self, concepts_list,
                 batch_size=64, num_workers=4,
                 train_prop=0.6, test_prop=0.2, val_prop=0.2,
                 ):
        super().__init__()
        self.normal_data_dir = "./dataset/Molecule_normal"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = val_prop
        self.concepts_list = concepts_list
        path_name = ''.join(map(lambda x: x['name'] + str(x['args']), concepts_list))
        hash_name = hashlib.sha256(path_name.encode('utf-8')).hexdigest()
        self.dataset_name = f"HIV_supernode_homogenous_{hash_name}"


    def setup(self, stage=None):
        dataset = MoleculeNet(self.normal_data_dir, name="HIV",
                              pre_transform=squeeze_y)
        dataset_len = len(dataset)

        if not osp.exists(f'./dataset/{self.dataset_name}'):
            print(f"Creating dataset with following concepts: {self.concepts_list}")
            os.makedirs(f'./dataset/{self.dataset_name}')
            for i in range(dataset_len // CHUNK_SIZE + 1):
                start_idx = i * CHUNK_SIZE
                end_idx = min((i + 1) * CHUNK_SIZE, dataset_len)
                chunk = dataset[start_idx : end_idx]
                transformed_dataset = [AddSupernodes(self.concepts_list)(data) for data in chunk]
                torch.save(
                    transformed_dataset[start_idx : end_idx],
                    f'./dataset/{self.dataset_name}/transformed_dataset_chunk_{i}.pth',
                )

        loaded_dataset = []
        num_chunks = dataset_len // CHUNK_SIZE + 1
        print("loading data")
        for i in tqdm(range(num_chunks)):
            chunk = torch.load(f'./dataset/{self.dataset_name}/transformed_dataset_chunk_{i}.pth')
            loaded_dataset.extend(chunk)

        self.train_dataset = loaded_dataset[:int(self.train_prop * dataset_len)]
        self.val_dataset = loaded_dataset[int(self.train_prop * dataset_len):int((self.train_prop + self.val_prop) * dataset_len)]
        self.test_dataset = loaded_dataset[int((self.train_prop + self.val_prop) * dataset_len):]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          self.batch_size, num_workers=self.num_workers)
