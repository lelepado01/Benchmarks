
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import lightning as L

from models.builder import build_model
import prov4ml
from datasets.masked import ModisMaskedPatchesDataset
from run_configs import RunConfigs
from losses.channel_reconstruction_loss import ChannelReconstructionLoss

class CustomTrainer(L.LightningModule):
    def __init__(self, configs : RunConfigs):
        super(CustomTrainer, self).__init__()
        self.model = build_model(configs)
        self.configs = configs

        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # self.loss_fn = ChannelReconstructionLoss(layer=0)
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        
        prov4ml.log_param("dataset transformation", self.image_transforms)


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        self.train_losses.append(loss.item())
        self.log("train_loss", loss)
        prov4ml.log_metric("train_loss",float(loss),prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_flops_per_batch("train_flops", self.model, batch, prov4ml.Context.TRAINING,step=self.current_epoch)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        self.val_losses.append(loss.item())
        self.log("val_loss", loss)

        prov4ml.log_metric("val_loss",loss,prov4ml.Context.VALIDATION,step=self.current_epoch)
        prov4ml.log_flops_per_batch("val_flops", self.model, batch, prov4ml.Context.VALIDATION,step=self.current_epoch)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        self.log("test_loss", loss) # necessary for output
        prov4ml.log_metric("test_loss",loss,prov4ml.Context.EVALUATION,step=self.current_epoch)

        return loss

    def on_train_epoch_end(self) -> None:
        prov4ml.log_metric("epoch",self.current_epoch,prov4ml.Context.TRAINING)
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_current_execution_time("train_step", prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self) -> None:
        prov4ml.log_metric("epoch",self.current_epoch,prov4ml.Context.VALIDATION)
        prov4ml.log_system_metrics(prov4ml.Context.VALIDATION,step=self.current_epoch)
        prov4ml.log_current_execution_time("val_step", prov4ml.Context.VALIDATION,step=self.current_epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.VALIDATION,step=self.current_epoch)
        return super().on_validation_epoch_end()
    
    def on_test_epoch_end(self) -> None:
        prov4ml.log_metric("epoch",self.current_epoch,prov4ml.Context.EVALUATION)
        prov4ml.log_system_metrics(prov4ml.Context.EVALUATION,step=self.current_epoch)
        prov4ml.log_current_execution_time("test_step", prov4ml.Context.EVALUATION,step=self.current_epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.EVALUATION,step=self.current_epoch)
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.configs.LEARNING_RATE)

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.dataset_train = ModisMaskedPatchesDataset(split="train")
            self.dataset_val = ModisMaskedPatchesDataset(split="val")
            if self.configs.train_samples is not None:
                self.dataset_train = Subset(self.dataset_train, range(self.configs.train_samples))
                self.dataset_val = Subset(self.dataset_val, range(self.configs.val_samples))

        if stage == "test" or stage is None:
            self.dataset_test = ModisMaskedPatchesDataset(split="test")
            if self.configs.test_samples is not None:
                self.dataset_test = Subset(self.dataset_test, range(self.configs.test_samples))
            
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.configs.BATCH_SIZE)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.configs.BATCH_SIZE)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.configs.BATCH_SIZE)
    
