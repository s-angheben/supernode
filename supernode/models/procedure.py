def train(model, criterion, optimizer, train_loader, device, supernode=False):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)

        supernode_mask = None
        edge_mask = None
        if supernode:
            supernode_mask = data.S > 0
            edge_mask = data.edge_S > 0

        out = model(data.x, data.edge_index, supernode_mask, edge_mask, data.batch)  # Perform a single forward pass.

        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(model, loader, device, supernode=False):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)

        supernode_mask = None
        edge_mask = None
        if supernode:
            supernode_mask = data.S > 0
            edge_mask = data.edge_S > 0

        out = model(data.x, data.edge_index, supernode_mask, edge_mask, data.batch)  # Perform a single forward pass.

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def printDataInfo(dataset):
    data = dataset[0]  # Get the first graph object.

    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
