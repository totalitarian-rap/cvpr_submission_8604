from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.loader import JointLoader, Loader
from src.unet import UNet
from src.parameters import Parameters as args

train_dataset = Loader('PA_LAT', 'train', augment=True, experts='123')
model = UNet(args)
log_name = "baseline_mcdo"

val_dataset = JointLoader('val', augment=False, experts='123')
train_loader = DataLoader(train_dataset, batch_size=24, num_workers=32, collate_fn=train_dataset.collate, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=32, collate_fn=val_dataset.collate, shuffle=False)

# logger = TensorBoardLogger(save_dir='/ayb/vol1/datasets/chest/to_paper/finetuner/', name=log_name)

checkpoint_callbacks = [
    ModelCheckpoint(
        monitor="Val_Total/"+metric_name,
        # dirpath=logger.log_dir+'/checkpoints',
        filename="sample-model-{epoch:02d}-"+metric_name.lower()+"{_val:.4f}",
        save_top_k=3,
        mode="max",
) for metric_name in ['IoU', 'MatthewsCorrcoef', 'Accuracy']]

trainer = pl.Trainer(
    num_processes=16, 
    gpus=[2,3], 
    precision=16, 
    accelerator='ddp',
    default_root_dir='/ayb/vol1/datasets/chest/to_paper/finetuner/',
    # logger=logger,
    callbacks=checkpoint_callbacks,
    max_epochs=500
    )

trainer.fit(model, train_loader, val_loader)