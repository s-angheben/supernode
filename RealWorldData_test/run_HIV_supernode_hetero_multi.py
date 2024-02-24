from lightning.pytorch.loggers import logger
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
import torch_geometric.transforms as T

from data.dataset import squeeze_y, MoleculeHIVNetDataModule
from data.concepts import *
from data.transformation import AddSupernodesHeteroMulti
from models.supernode_hetero_multi_GNNs import get_SHGIN_multi

from torch_geometric.seed import seed_everything
import argparse

seed_everything(42)

NUM_NODE_FEATURES = 9
NUM_CLASSES = 2

## hyperparameters
MAX_EPOCH = 150
BATCH_SIZE = 100

parser = argparse.ArgumentParser(description="BREC Test")
parser.add_argument("--EPOCH", type=int, default=MAX_EPOCH)
parser.add_argument("--BATCH_SIZE", type=int, default=BATCH_SIZE)
parser.add_argument("--concepts", type=str, default="cycle_basis")

args = parser.parse_args()
EPOCH = args.EPOCH
BATCH_SIZE = args.BATCH_SIZE
concept_list_name = args.concepts
concept_list = get_concept_list(args.concepts)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

supnodes_name = [concept['name'] for concept in concept_list]

dm = MoleculeHIVNetDataModule("./dataset/HIV_multi", batch_size=BATCH_SIZE,
                              train_prop=0.6, test_prop=0.2, val_prop=0.2,
                              pre_transform=squeeze_y,
                              transform=AddSupernodesHeteroMulti(concept_list),
                              num_workers=8
                              )

dm.setup()
model, model_log = get_SHGIN_multi(in_channels=NUM_NODE_FEATURES,
                                   out_channels=NUM_CLASSES,
                                   supnodes_name=supnodes_name,)
model = model.to(device)
print(model)

logger = TensorBoardLogger("tb_logs/" + "HIV_multi", model_log["model"], concept_list_name)
logger.log_hyperparams(dict({"device": device, "concepts": concept_list_name,
                        "dataset": "HIV", "epochs": MAX_EPOCH}, **model_log))

trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCH,
                    logger=logger)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)
