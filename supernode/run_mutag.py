import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from models.GIN import GIN
from models.GCN import GCN
from models.procedure import train, test
import torch_geometric.transforms as T
from concepts.concepts import *
from concepts.transformations import AddSupernodes
import os.path as osp

# MUTAG IMDB-BINARY, REDDIT-BINARY
dataset_name = 'IMDB-BINARY'
path = osp.join('./dataset/', dataset_name)
pathT = osp.join('./dataset/', dataset_name + 'T')
batch_size = 64

print(dataset_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset = TUDataset(root=path, name=dataset_name)

dataset = dataset.shuffle()

train_loader = DataLoader(dataset[:0.9], batch_size, shuffle=True)
test_loader = DataLoader(dataset[0.9:], batch_size)

model = GCN(num_node_features=dataset.num_node_features,
            hidden_channels=64,
            num_classes=dataset.num_classes).to(device)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

sum_acc = 0
loop = 20
#for i in range(0, loop):
#    test_acc = test(model, test_loader, device)
##    print(f'Test Acc: {test_acc:.4f}')
#    for epoch in range(1, 150):
#        train(model, criterion, optimizer, train_loader, device)
#        train_acc = test(model, train_loader, device)
##        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
#    test_acc = test(model, test_loader, device)
#    sum_acc += test_acc
#    print(f'[{i}]Test Acc: {test_acc:.4f}')
#    for layer in model.children():
#        if hasattr(layer, 'reset_parameters'):
#            layer.reset_parameters()
#print(f'Average Test Acc: {sum_acc/loop:.4f}')

############################
### ADD SUPERNODES
############################

concepts_list_ex = [
       {"name": "GCB", "fun": cycle_basis, "args": []},
       {"name": "GMC", "fun": max_cliques, "args": []},
       {"name": "GLP2", "fun": line_paths, "args": [2]}
    ]

#dataset.transform = AddSupernodes(concepts_list_ex)
dataset = TUDataset(root=pathT, name=dataset_name,
                    pre_transform=AddSupernodes(concepts_list_ex)
                    )


dataset = dataset.shuffle()

train_loader = DataLoader(dataset[:0.9], batch_size, shuffle=True)
test_loader = DataLoader(dataset[0.9:], batch_size)

model = GCN(num_node_features=dataset.num_node_features,
            hidden_channels=64,
            num_classes=dataset.num_classes).to(device)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

sum_acc = 0
for i in range(0, 20):
    test_acc = test(model, test_loader, device, supernode=True)
#    print(f'Test Acc: {test_acc:.4f}')
    for epoch in range(1, 150):
        train(model, criterion, optimizer, train_loader, device, supernode=True)
        train_acc = test(model, train_loader, device, supernode=True)
#        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
    test_acc = test(model, test_loader, device, supernode=True)
    sum_acc += test_acc
    print(f'[{i}]Test Acc: {test_acc:.4f}')
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
print(f'Average Test Acc: {sum_acc/loop:.4f}')
