from torch_geometric.loader import DataLoader

# keeps the original dataset tree/cycle proportion
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
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader


