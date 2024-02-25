from lightning.pytorch.loggers import logger
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
import torch_geometric.transforms as T

from data.dataset import TUDProteinsDataModule
from data.concepts import *
from data.transformation import AddSupernodesHeteroMulti
from models.supernode_hetero_multi_GNNs import get_SHGIN_multi

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

supnodes_name = [concept['name'] for concept in concept_list]

#path_name = ''.join(map(lambda x: x['name'] + str(x['args']), concepts_list))
#hash_name = hashlib.sha256(path_name.encode('utf-8')).hexdigest()
#dataset_name = f"PROTEIN_supernode_hetero_multi_{hash_name}"

#dm = MoleculeHIV_hetero_multi_NetDataModule(f'./dataset/{dataset_name}',
#                              concept_list=concepts_list, batch_size=BATCH_SIZE,
#                              train_prop=0.6, test_prop=0.2, val_prop=0.2,
#                              num_workers=16
#                              )
#dm = MoleculeHIVNetDataModule("./dataset/Molecule_normaleee111222", batch_size=1,
#                              train_prop=0.6, test_prop=0.2, val_prop=0.2,
#                              pre_transform=T.Compose([squeeze_y, AddSupernodesHeteroMulti(concepts_list)]),
#                              num_workers=16
#                              )
#dm = MoleculeHIVNetDataModule("./dataset/uuu", batch_size=BATCH_SIZE,
#                              train_prop=0.6, test_prop=0.2, val_prop=0.2,
#                              num_workers=0,
#                              transform=T.Compose([squeeze_y, AddSupernodesHeteroMulti(concepts_list)])
#                     )
dm = TUDProteinsDataModule("./dataset/PROTEINS_multi", batch_size=BATCH_SIZE,
                              train_prop=0.6, test_prop=0.2, val_prop=0.2,
                              transform=AddSupernodesHeteroMulti(concept_list),
                              num_workers=8
                              )

dm.setup()
model, model_log = get_SHGIN_multi(in_channels=NUM_NODE_FEATURES,
                                   out_channels=NUM_CLASSES,
                                   supnodes_name=supnodes_name,)
model = model.to(device)
print(model)

logger = TensorBoardLogger("tb_logs/" + "PROTEINS_multi", model_log["model"], concept_list_name)
logger.log_hyperparams(dict({"device": device, "concepts": concept_list_name,
                        "dataset": "PROTEINS", "epochs": MAX_EPOCH}, **model_log))

trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCH,
                    logger=logger)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)
