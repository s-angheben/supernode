from lightning.pytorch.loggers import logger
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import MutagDataModule

from models.normal_GNNs import *

## hyperparameters
MAX_EPOCH = 300
BATCH_SIZE = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dm = MutagDataModule("./dataset/Mutag", batch_size=20, train_prop=0.6, test_prop=0.2, val_prop=0.2)
dm.setup()
#model, model_log = get_GCNN_model(in_channels=dm.dataset.num_node_features,
#             out_channels=dm.dataset.num_classes)
model, model_log = get_GINN_model(in_channels=dm.dataset.num_node_features,
             out_channels=dm.dataset.num_classes)
model = model.to(device)
print(model)

logger = TensorBoardLogger("tb_logs/" + "MUTAG", model_log["model"])
logger.log_hyperparams(dict({"device": device,
                        "dataset": "MUTAG", "epochs": MAX_EPOCH}, **model_log))

trainer = L.Trainer(accelerator="gpu", devices=[0], max_epochs=MAX_EPOCH,
                    logger=logger)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)

