from torch_geometric.datasets import MoleculeNet, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
import torch
import lightning as L

from torch_geometric.transforms.compose import HeteroData

from .transformation import AddSupernodes, AddSupernodesHeteroMulti

def squeeze_y(data):
    data.y = data.y.squeeze(1).long()
    return data

CHUNK_SIZE = 5000
HIV_DATASET_LEN = 41127

class MoleculeHIVNetDataModule(L.LightningDataModule):
    def __init__(self, data_dir, transform=None, pre_transform=None,
                 batch_size=64, num_workers=4,
                 train_prop=0.6, test_prop=0.2, val_prop=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = val_prop
        self.transform = transform
        self.pre_transform = pre_transform


    def setup(self, stage=None):
        self.dataset = MoleculeNet(self.data_dir, name="HIV",
                                 pre_transform=self.pre_transform,
                                 transform=self.transform)
        self.dataset = self.dataset.shuffle()

    def train_dataloader(self):
        return DataLoader(self.dataset[:self.train_prop], self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset[self.train_prop:self.train_prop+self.val_prop], self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset[self.train_prop+self.val_prop:], self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


class MoleculeHIV_herero_multi(InMemoryDataset):
    def __init__(self, root, concepts, transform=None, pre_transform=None):
        self.concepts = concepts
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    def raw_file_names(self):
        return ['data.pth']

    def processed_file_names(self):
        return ['transformed_dataset.pth']


    def download(self):
        dataset = MoleculeNet("./dataset/STEP/", name="HIV",
                              pre_transform=squeeze_y,)
        transformed_dataset = [AddSupernodesHeteroMulti(self.concepts)(data) for data in dataset]
        torch.save(transformed_dataset, f'{self.raw_dir}/data.pth')

    def process(self):
        print(self.processed_paths[0])
        data_list = torch.load(f'{self.raw_dir}/data.pth')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, f'{self.processed_dir}/transformed_dataset.pth')



class MoleculeHIV_hetero_multi_NetDataModule(L.LightningDataModule):
    def __init__(self, data_dir, concept_list, transform=None, pre_transform=None,
                 batch_size=64, num_workers=4,
                 train_prop=0.6, test_prop=0.2, val_prop=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = val_prop
        self.transform = transform
        self.pre_transform = pre_transform
        self.concept_list = concept_list


    def setup(self, stage=None):
        self.dataset = MoleculeHIV_herero_multi(self.data_dir, self.concept_list,
                                                pre_transform=self.pre_transform,
                                                transform=self.transform)
        self.dataset = self.dataset.shuffle()

    def train_dataloader(self):
        return DataLoader(self.dataset[:self.train_prop], self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset[self.train_prop:self.train_prop+self.val_prop], self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset[self.train_prop+self.val_prop:], self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

class TUDProteinsDataModule(L.LightningDataModule):
    def __init__(self, data_dir, transform=None, pre_transform=None,
                 batch_size=64, num_workers=4,
                 train_prop=0.6, test_prop=0.2, val_prop=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = val_prop
        self.transform = transform
        self.pre_transform = pre_transform


    def setup(self, stage=None):
        self.dataset = TUDataset(self.data_dir, name="PROTEINS",
                                 pre_transform=self.pre_transform,
                                 transform=self.transform)
        self.dataset = self.dataset.shuffle()

    def train_dataloader(self):
        return DataLoader(self.dataset[:self.train_prop], self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset[self.train_prop:self.train_prop+self.val_prop], self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset[self.train_prop+self.val_prop:], self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
