import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics

import timm
from timm.layers import create_classifier

from copy import deepcopy


class XceptionBase(pl.LightningModule):

    def __init__(self,
                 lr=0.001,
                 loss='mse',
                 sch_factor=0.2,
                 sch_patience=3,
                 sch_min_lr=1e-6,
                 sch_monitor='val_mse',
                 **kwargs):
        super().__init__()

        self.lr = lr
        self.loss = loss
        self.num_features = 2048
        self.loss_dict = {'mse': nn.MSELoss,
                          'mape': torchmetrics.MeanAbsolutePercentageError}

        self.model = timm.create_model('xception', features_only=True,
                                       pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.global_pool, _ = create_classifier(self.num_features, 1,
                                                pool_type='avg')

        self.fc_block = None

        self.define_loss(self.loss)
        self.define_scheduller(sch_factor,
                               sch_patience,
                               sch_min_lr,
                               sch_monitor)

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.train_mape = torchmetrics.MeanAbsolutePercentageError()
        self.val_mape = torchmetrics.MeanAbsolutePercentageError()
        self.test_mape = torchmetrics.MeanAbsolutePercentageError()

    def unfreeze_n_layers(self, n_unfreeze):
        self.counter = 0
        self.n_unfreeze = n_unfreeze
        self._recursive_step(self.model)

    def _recursive_step(self, block):
        if self.counter == self.n_unfreeze:
            return
        children = list(block.children())[::-1]
        parameters = list(block.parameters())
        if children:
            for child in children:
                self._recursive_step(child)
                if self.counter == self.n_unfreeze:
                    return
        elif parameters:
            for param in block.parameters():
                param.requires_grad = True
            self.counter += 1

        if self.counter == self.n_unfreeze:
            return

    def forward(self, x):
        x = self.model(x)[-1]
        x = self.global_pool(x)
        preds = self.fc_block(x)
        return preds

    def define_loss(self, loss_name):
        self.loss_fn = self.loss_dict[loss_name]()

    def define_scheduller(self, sch_factor=0.2,
                          sch_patience=3,
                          sch_min_lr=1e-6,
                          sch_monitor='val_mse'):

        self.sch_factor = sch_factor
        self.sch_patience = sch_patience
        self.sch_min_lr = sch_min_lr
        self.sch_monitor = sch_monitor

    def training_step(self, batch, batch_idx):

        x, y = batch
        y = y.view(-1, 1)

        preds = self.forward(x)

        loss = self.loss_fn(preds, y)
        self.train_mse(preds, y.view(-1, 1))
        self.train_mae(preds, y.view(-1, 1))
        self.train_mape(preds, y.view(-1, 1))

        self.log('train_mse', self.train_mse,
                 on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('train_mae', self.train_mae,
                 on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('train_mape', self.train_mape,
                 on_epoch=True, prog_bar=True,
                 logger=True)

        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y = y.view(-1, 1)

        preds = self.forward(x)

        self.val_mse(preds, y.view(-1, 1))
        self.val_mae(preds, y.view(-1, 1))
        self.val_mape(preds, y.view(-1, 1))

        self.log('val_mse', self.val_mse,
                 on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('val_mae', self.val_mae,
                 on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('val_mape', self.val_mape,
                 on_epoch=True, prog_bar=True,
                 logger=True)

    def test_step(self, batch, batch_idx):

        x, y = batch
        y = y.view(-1, 1)
        preds = self.forward(x)

        self.test_mse(preds, y.view(-1, 1))
        self.test_mae(preds, y.view(-1, 1))
        self.test_mape(preds, y.view(-1, 1))

        self.log('test_mse', self.test_mse,
                 on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('test_mae', self.test_mae,
                 on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('test_mape', self.test_mape,
                 on_epoch=True, prog_bar=True,
                 logger=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler":
                                 ReduceLROnPlateau(optimizer,
                                                   factor=self.sch_factor,
                                                   patience=self.sch_patience,
                                                   min_lr=self.sch_min_lr),
                                 "monitor": self.sch_monitor}}


class Xception(XceptionBase):

    def __init__(self,
                 lr=0.001,
                 loss='mse',
                 sch_factor=0.2,
                 sch_patience=3,
                 sch_min_lr=1e-6,
                 sch_monitor='val_mse',
                 **kwargs):
        super().__init__(lr,
                         loss,
                         sch_factor,
                         sch_patience,
                         sch_min_lr,
                         sch_monitor)

        self.fc_block = nn.Sequential(
            nn.Linear(2048, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 1)
        )


class MetricsCallback(pl.callbacks.Callback):
    """Recording all metrics"""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)
