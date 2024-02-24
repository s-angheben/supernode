from lightning.pytorch.loggers import logger
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import TUDProteinsDataModule

from models.normal_GNNs import *

from torch_geometric.seed import seed_everything

seed_everything(42)

## hyperparameters
MAX_EPOCH = 150
BATCH_SIZE = 100
NUM_NODE_FEATURES = 3
NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dm = TUDProteinsDataModule("./dataset/Proteins_normal", batch_size=BATCH_SIZE,
                              train_prop=0.6, test_prop=0.2, val_prop=0.2,
                              )
dm.setup()
model, model_log = get_GIN_model(in_channels=NUM_NODE_FEATURES,
                                 out_channels=NUM_CLASSES)
model = model.to(device)
print(model)

logger = TensorBoardLogger("tb_logs/" + "PROTEINS_normal", model_log["model"])
logger.log_hyperparams(dict({"device": device,
                        "dataset": "PROTEINS", "epochs": MAX_EPOCH}, **model_log))

trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCH,
                    logger=logger)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)
