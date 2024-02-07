from lightning.pytorch.loggers import logger
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import MoleculeHIVNetDataModule
from models.H_GNNs import get_HTEST_model

from concepts.concepts import *
from concepts.transformations import AddSupernodesHetero

import hashlib

############################
### SUPERNODES
############################

## hyperparameters
MAX_EPOCH = 300
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
dm = MoleculeHIVNetDataModule("./dataset/HIVH1"+hash_name,
                     transform=AddSupernodesHetero(concepts_list_ex),
                     batch_size=20, train_prop=0.6, test_prop=0.2, val_prop=0.2)
dm.setup()

out_channels = dm.dataset.num_classes
hidden_channels = 64


############################
### models
############################

model, model_log = get_HTEST_model(out_channels, hidden_channels=hidden_channels)

model = model.to(device)
print(model)

logger = TensorBoardLogger("tb_logs/" + "HIV", model_log)

trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCH,
                    logger=logger)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)

