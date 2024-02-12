from lightning.pytorch.loggers import logger
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import Dataset_tree_cycle_module

from models.S1_GNNs import *
from models.S2_GNNs import *
from models.S_GNNs import *
from models.S_module import *

from concepts.concepts import *
from concepts.transformations import AddSupernodes

import hashlib

############################
### SUPERNODES
############################

## hyperparameters
MAX_EPOCH = 20
BATCH_SIZE = 60


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

concepts_list_ex = [
       {"name": "GCB", "fun": cycle_basis, "args": []},
       {"name": "GMC", "fun": max_cliques, "args": []},
       {"name": "GLP2", "fun": line_paths, "args": []}
    ]

path_name = ''.join(map(lambda x: x['name'] + str(x['args']), concepts_list_ex))
hash_name = hashlib.sha256(path_name.encode('utf-8')).hexdigest()
dm = Dataset_tree_cycle_module("./dataset/d1Mem-"+hash_name,
                               dataset_path="/home/sam/Documents/network/supernode/dataset/d1",
                               pre_transform=AddSupernodes(concepts_list_ex),
                               batch_size=60, train_prop=0.6, test_prop=0.2, val_prop=0.2)
dm.setup()

in_channels = dm.dataset.num_node_features
out_channels = dm.dataset.num_classes
hidden_channels = 64

############################
### models
############################

#model, model_log = get_GCNS1_model(in_channels=dm.dataset.num_node_features,
#             out_channels=dm.dataset.num_classes)
#model, model_log = get_GINS1_model(in_channels=dm.dataset.num_node_features,
#             out_channels=dm.dataset.num_classes)
#model, model_log = get_GCNS2_model(in_channels=dm.dataset.num_node_features,
#             out_channels=dm.dataset.num_classes)
model, model_log = get_GINS2_model(in_channels=dm.dataset.num_node_features,
             out_channels=dm.dataset.num_classes)

model = model.to(device)
print(model)


logger = TensorBoardLogger("tb_logs/" + "tree_cycle", model_log["model"])
logger.log_hyperparams(dict({"device": device,
                        "dataset": "tree_cycle_transformed",
                        "concepts": str(concepts_list_ex),
                        "epochs": MAX_EPOCH}, **model_log))

trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCH,
                    logger=logger)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)

