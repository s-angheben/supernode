import torch
from data.dataset import Dataset_tree_cycle, Dataset_tree_cycle_Memory
from data.dataloader import create_dataloader
from models.GCN import GCN
from models.GIN import GIN
from models.GraphConv import GNN
from models.procedure import train, test
import torch_geometric.transforms as T
from concepts.concepts import *
from concepts.transformations import AddSupernodes
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = Dataset_tree_cycle_Memory("./dataset/d1Mem", "/home/sam/Documents/network/project/dataset/d1")

train_loader, test_loader, val_loader = create_dataloader(
        dataset, batch_size=60,
        train_prop=0.6, test_prop=0.2, val_prop=0.2
        )

#model = GCN(num_node_features=dataset.num_node_features,
#            hidden_channels=32,
#            num_classes=dataset.num_classes).to(device)

model = GNN(num_node_features=dataset.num_node_features,
            hidden_channels=32,
            num_classes=dataset.num_classes).to(device)

#model = GIN(
#    in_channels=dataset.num_features,
#    hidden_channels=32,
#    num_classes=dataset.num_classes,
#    num_layers=3,
#).to(device)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

test_acc = test(model, test_loader, device)
print(f'Test Acc: {test_acc:.4f}')
for epoch in range(1, 10):
    train(model, criterion, optimizer, train_loader, device)
    train_acc = test(model, train_loader, device)
    val_acc = test(model, val_loader, device)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
test_acc = test(model, test_loader, device)
print(f'Test Acc: {test_acc:.4f}')


############################
### ADD SUPERNODES
############################

concepts_list_ex = [
       {"name": "GCB", "fun": cycle_basis, "args": []},
       {"name": "GMC", "fun": max_cliques, "args": []},
       {"name": "GLP2", "fun": line_paths, "args": [2]}
    ]

#dataset.transform = AddSupernodes(concepts_list_ex)
dataset = Dataset_tree_cycle_Memory(root="./dataset/d1MemT",
                                    dataset_path="/home/sam/Documents/network/project/dataset/d1",
                                    pre_transform=AddSupernodes(concepts_list_ex)
                                    )
print(dataset.size)

train_loader, test_loader, val_loader = create_dataloader(
        dataset, batch_size=60,
        train_prop=0.6, test_prop=0.2, val_prop=0.2
        )

#model = GCN(num_node_features=dataset.num_node_features,
#            hidden_channels=32,
#            num_classes=dataset.num_classes).to(device)

model = GIN(
    in_channels=dataset.num_features,
    hidden_channels=32,
    num_classes=dataset.num_classes,
    num_layers=3,
).to(device)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

test_acc = test(model, test_loader, device, supernode=True)
print(f'Test Acc: {test_acc:.4f}')
for epoch in range(1, 10):
    train(model, criterion, optimizer, train_loader, device, supernode=True)
    train_acc = test(model, train_loader, device, supernode=True)
    val_acc = test(model, val_loader, device, supernode=True)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
test_acc = test(model, test_loader, device, supernode=True)
print(f'Test Acc: {test_acc:.4f}')
