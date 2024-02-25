from lightning.pytorch.loggers import logger
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
import torch_geometric.transforms as T

from data.dataset import TUDProteinsDataModule
from data.transformation import AddSupernodes
from data.concepts import *
from models.supernode_homogeneous_GNNs import get_SGIN_model

from torch_geometric.seed import seed_everything
import hashlib
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
parser.add_argument("--concepts", type=str, default="cyclebasis")

args = parser.parse_args()
EPOCH = args.EPOCH
BATCH_SIZE = args.BATCH_SIZE
concept_list_name = args.concepts
concept_list = get_concept_list(args.concepts)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

path_name = ''.join(map(lambda x: x['name'] + str(x['args']), concept_list))
hash_name = hashlib.sha256(path_name.encode('utf-8')).hexdigest()
dataset_name = f"PROTEINS_supernode_homogenous_{hash_name}"

transformation = T.Compose([AddSupernodes(concept_list)])

dm = TUDProteinsDataModule(f'./dataset/{dataset_name}', batch_size=BATCH_SIZE,
                              train_prop=0.6, test_prop=0.2, val_prop=0.2,
                              pre_transform=transformation,
                              num_workers=4
                              )
dm.setup()
model, model_log = get_SGIN_model(in_channels=NUM_NODE_FEATURES,
                                  out_channels=NUM_CLASSES)
model = model.to(device)
print(model)

logger = TensorBoardLogger("tb_logs/" + "PROTEINS_hom", model_log["model"], concept_list_name)
logger.log_hyperparams(dict({"device": device, "concepts": concept_list_name,
                        "dataset": "PROTEINS", "epochs": MAX_EPOCH}, **model_log))

trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCH,
                    logger=logger)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)
