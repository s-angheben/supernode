from lightning.pytorch.loggers import logger
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import MoleculeHIVNetDataModule, squeeze_y

from models.normal_GNNs import *

from torch_geometric.seed import seed_everything
seed_everything(42)

## hyperparameters
MAX_EPOCH = 150
BATCH_SIZE = 100
NUM_NODE_FEATURES = 9
NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dm = MoleculeHIVNetDataModule("./dataset/HIV_normal", batch_size=BATCH_SIZE,
                              train_prop=0.6, test_prop=0.2, val_prop=0.2,
                              pre_transform=squeeze_y,
                              )
dm.setup()
model, model_log = get_GIN_model(in_channels=NUM_NODE_FEATURES,
                                 out_channels=NUM_CLASSES)
model = model.to(device)
print(model)

logger = TensorBoardLogger("tb_logs/" + "HIV_normal", model_log["model"])
logger.log_hyperparams(dict({"device": device,
                        "dataset": "HIV", "epochs": MAX_EPOCH}, **model_log))

trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCH,
                    logger=logger)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)
