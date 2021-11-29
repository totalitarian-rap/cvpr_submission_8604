from argparse import ArgumentParser
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.loader import JointLoader, Loader
from src.unet import UNet
from src.parameters import Parameters as model_args

def main(args):
    epochs = args.epochs
    gpu = args.gpus
    precision = args.precision
    num_workers = args.num_workers
    experts = args.experts
    train_bs = args.train_bs
    val_bs = args.val_bs
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = '/logs'
    data_dir = args.data_dir


    train_dataset = Loader('PA_LAT', 'train', metadata_prefix=data_dir, augment=True, experts=experts)
    model = UNet(model_args)
    log_name = "baseline_mcdo"

    val_dataset = JointLoader('val', metadata_prefix=data_dir, augment=False, experts=experts)
    train_loader = DataLoader(train_dataset, batch_size=train_bs, collate_fn=train_dataset.collate, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_bs, collate_fn=val_dataset.collate, shuffle=False, num_workers=num_workers)


    logger = TensorBoardLogger(save_dir=log_dir, name=log_name)

    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="Val_Total/"+metric_name,
            dirpath=logger.log_dir+'/checkpoints',
            filename="sample-model-{epoch:02d}-"+metric_name.lower()+"{_val:.4f}",
            save_top_k=3,
            mode="max",
    ) for metric_name in ['IoU', 'MatthewsCorrcoef', 'Accuracy']]

    trainer = pl.Trainer(
        num_processes=num_workers, 
        gpus=[1,3], 
        precision=precision, 
        accelerator='ddp',
        default_root_dir=log_dir,
        logger=logger,
        callbacks=checkpoint_callbacks,
        max_epochs=500,
        num_sanity_val_steps=0
        )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--experts', type=str, default='123')
    parser.add_argument('--gpus', default=3)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--train_bs', type=int, default=16)
    parser.add_argument('--val_bs', type=int, default=32)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
