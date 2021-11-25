import torch
from torch import nn
from torch import optim
from timm import create_model
import pytorch_lightning as pl
from torchmetrics import (
    MetricCollection, Accuracy, Precision, Recall, F1, MatthewsCorrcoef,
    ConfusionMatrix, AUROC, IoU
)
from . import models_utils
from .unet_utils import Encoder, Decoder

class UNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization=False
        self.save_hyperparameters()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        if args.uncertainty_parameters['estimate_uncertainty']:
            p = args.uncertainty_parameters['dropout_p']
            models_utils.add_droput(self, p)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = args.optimizer_parameters['lr']
        self.wd = args.optimizer_parameters['wd']   
        metrics = MetricCollection([
            Accuracy(),
            Precision(ignore_index=0),
            Recall(ignore_index=0),
            F1(ignore_index=0),
            MatthewsCorrcoef(num_classes=2),
        ])
        self.train_metrics = metrics.clone(prefix='Train_Total/')
        self.val_metrics_pa = metrics.clone(prefix='Val_PA/')
        self.val_metrics_lat = metrics.clone(prefix='Val_LAT/')
        self.val_metrics_total = metrics.clone(prefix='Val_Total/')
        self.train_iou = IoU(num_classes=2)
        self.val_iou_pa = IoU(num_classes=2)
        self.val_iou_lat = IoU(num_classes=2)
        self.val_iou_total = IoU(num_classes=2)

    def forward(self, x, *args, **kwargs):
            x = self.encoder(x)
            return self.decoder(x, self.encoder.partial_output)

    def forward_fn(self, x):
        return self.encoder(x)

    def attack(self, data, epsilon):
        grad = data.grad
        return data + epsilon*grad.sign()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        batch['images'].requires_grad = True
        pred, masks = self(batch['images']), batch['masks']
        loss = self.loss_fn(pred.squeeze(), masks.float())
        opt.zero_grad()
        self.manual_backward(loss)
        pertrubated = self.attack(batch['images'], 0.01)
        pertrubated_pred = self(pertrubated)
        pertrubated_loss = self.loss_fn(pertrubated_pred.squeeze(), masks.float())
        self.manual_backward(pertrubated_loss)
        opt.step()
        ans, gr_tr = self.batch_binary_pred(pred.squeeze()>0, masks)

        return {
                'loss': loss,
                'adv_loss': pertrubated_loss,
                'preds': pred.detach(),
                'masks': masks,
                'ans': ans.detach(),
                'gr_tr': gr_tr.detach()
            }

    def training_step_end(self, results):
        self.log('Train_Total/Loss', results['loss'])
        preds, masks = results['preds'], results['masks']
        ans, gr_tr = results['ans'], results['gr_tr']
        self.train_metrics(ans.to(self.device), gr_tr.to(self.device))
        self.log_dict(self.train_metrics, on_epoch=True)
        self.train_iou(preds.squeeze()>0, masks)
        self.log('Train_Total/IoU', self.train_iou, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        pred_pa, pred_lat = self(batch['images_pa']), self(batch['images_lat'])
        masks_pa, masks_lat = batch['masks_pa'], batch['masks_lat']
        loss_pa = self.loss_fn(pred_pa.squeeze(), masks_pa.float())
        loss_lat = self.loss_fn(pred_lat.squeeze(), masks_lat.float())
        total_loss = loss_pa+loss_lat
        ans_pa, gr_tr = self.batch_binary_pred(pred_pa.squeeze()>0, masks_pa)
        ans_lat, gr_tr = self.batch_binary_pred(pred_lat.squeeze()>0, masks_lat)
        return {
            'loss': total_loss,
            'loss_pa': loss_pa,
            'loss_lat': loss_lat,
            'ans_pa': ans_pa,
            'ans_lat': ans_lat,
            'gr_tr': gr_tr,
            'pred_pa': pred_pa,
            'pred_lat': pred_lat,
            'masks_pa': masks_pa,
            'masks_lat': masks_lat
        }

    def validation_step_end(self, results):
        ans_pa, ans_lat = results['ans_pa'], results['ans_lat']
        pred_pa, pred_lat = results['pred_pa'], results['pred_lat']
        masks_pa, masks_lat = results['masks_pa'], results['masks_lat']
        gr_tr = results['gr_tr']
        self.val_metrics_pa(ans_pa.to(self.device), gr_tr.to(self.device))
        self.val_metrics_lat(ans_lat.to(self.device), gr_tr.to(self.device))
        unmatched = torch.logical_xor(ans_pa, ans_lat)==True
        ans_pa[unmatched] = 1 - gr_tr[unmatched]
        self.val_metrics_total(ans_pa.to(self.device), gr_tr.to(self.device))
        self.log_dict(self.val_metrics_pa, on_epoch=True, on_step=False)
        self.log_dict(self.val_metrics_lat, on_epoch=True, on_step=False)
        self.log_dict(self.val_metrics_total, on_epoch=True, on_step=False)
        self.val_iou_pa(pred_pa.squeeze()>0, masks_pa)
        self.val_iou_lat(pred_lat.squeeze()>0, masks_lat)
        self.val_iou_total(pred_pa.squeeze()>0, masks_pa)
        self.val_iou_total(pred_lat.squeeze()>0, masks_lat)
        self.log('Val_Total/Loss', results['loss'], on_epoch=True, on_step=False)
        self.log('Val_PA/Loss', results['loss_pa'], on_epoch=True, on_step=False)
        self.log('Val_LAT/Loss', results['loss_lat'], on_epoch=True, on_step=False)
        self.log('Val_Total/IoU', self.val_iou_total, on_epoch=True, on_step=False)
        self.log('Val_PA/IoU', self.val_iou_pa, on_epoch=True, on_step=False)
        self.log('Val_LAT/IoU', self.val_iou_lat, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=30, min_lr=3e-6, verbose=True,
                ),
            'reduce_on_plateau': True,
            'monitor': 'Train_Total/Loss'
        }
        return [optimizer], [scheduler] 

    def batch_binary_pred(self, pred, mask):
        t1 = torch.tensor([elem.sum()>0 for elem in pred])
        t2 = torch.tensor([elem.sum()>0 for elem in pred*mask])
        gr_tr = torch.tensor([elem.sum()>0 for elem in mask]).int()
        ans_vec = torch.logical_or(torch.logical_and(t1, torch.logical_xor(t1, gr_tr)), t2).int()
        return ans_vec, gr_tr