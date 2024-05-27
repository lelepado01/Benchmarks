
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
import lightning as L

from run_configs import RunConfigs
from trainer import CustomTrainer
from view import get_images, save_losses
from pytorch_lightning.loggers import WandbLogger
import prov4ml

configs = RunConfigs.default()

PROV_SAVE_DIR = "/lustre/orion/cli900/world-shared/users/gabrielepadovani/prov"
MLFLOW_SAVE_DIR = "/lustre/orion/cli900/world-shared/users/gabrielepadovani/test_runs"
MODEL_NAME = configs.get_experiment_name()

prov4ml.start_run(
    prov_user_namespace="www.example.org", 
    experiment_name=MODEL_NAME,
    provenance_save_dir=PROV_SAVE_DIR,
    mlflow_save_dir=MLFLOW_SAVE_DIR,
) 

torch.set_float32_matmul_precision('high') # or 'medium'
    
vit_trainer = CustomTrainer(configs)

trainer = L.Trainer(
    enable_checkpointing=False,
    accelerator="gpu",
    strategy=configs.STRATEGY,
    max_epochs=configs.EPOCHS,
    num_nodes=configs.NUM_NODES,
    devices=torch.cuda.device_count(),
    # logger=WandbLogger(offline=True), 
    logger=prov4ml.get_mlflow_logger(),
)

prov4ml.log_model(model=vit_trainer, model_name=MODEL_NAME)
trainer.fit(vit_trainer)

trainer.test(vit_trainer)

# trainer.save_checkpoint(MODEL_NAME + ".ckpt")
# save_losses(vit_trainer, MODEL_NAME)
# get_images(vit_trainer.model, vit_trainer.dataset_train, number_of_images=10)

prov4ml.end_run()

# join_from_dir(PROV_SAVE_DIR)
