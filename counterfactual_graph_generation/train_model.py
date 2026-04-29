import logging

logger = logging.getLogger(__name__)

import hydra
import omegaconf
from lightning.pytorch.loggers import WandbLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from counterfactual_graph_generation.data.make_dataset import CounterfactualGraphDataModule
from counterfactual_graph_generation.models.model import ModelFactory


@hydra.main(config_path="../config", config_name="config.yaml", version_base="1.2")
def main(cfg):
    # Validate config
    if cfg['model']['pl_config']['batch_size'] != cfg['dataset']['batch_size']:
        batch_size = cfg['model']['pl_config']['batch_size']
        logger.info(f'Batchsize defined in data module set to model defined batchsize of: {batch_size}')
        cfg['dataset']['batch_size'] = cfg['model']['pl_config']['batch_size']

    # Setup data module
    logger.info("Load data module, prepare- and setup data...")
    cfg['dataset']['dense_data_representation'] = cfg['model']['pl_config']['requires_dense']
    seed_everything(seed=cfg['dataset']['seed'], workers=True)
    data_module = CounterfactualGraphDataModule(**cfg['dataset'])
    data_module.prepare_data()
    data_module.setup()
    train_dataloader, val_dataloader, test_dataloader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

    # Initialize model
    logger.info("Initialize and setup model from defualt configurations...")
    model_factory = ModelFactory(**cfg['model'])
    model = model_factory.setup_model()
    logger.info("Model was loaded and setup succcesfully.")

    # Setup logger
    logger.info("Setup model training...")
    wandb_logger = WandbLogger(project=cfg['model']['model_name'] + "_" + cfg['dataset']['dataset_name'], log_model="all", save_dir='./logging')
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_logger.experiment.config.update(cfg_dict) # Ensures that hydra configs are stored as part of the wandb logs

    # Train model
    checkpoint_best_callback = ModelCheckpoint(**cfg['trainer']['checkpoint_best'])
    checkpoint_last_callback = ModelCheckpoint(**cfg['trainer']['checkpoint_last'])
    checkpoint_best_reconstruction_callback = [ModelCheckpoint(**cfg['trainer']['checkpoint_best_reconstruction'])] if cfg['model']['model_name'] in ['PEGVAE', 'PretrainedPEGVAE'] else []
    callbacks = [checkpoint_best_callback, checkpoint_last_callback] + checkpoint_best_reconstruction_callback
    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **cfg['trainer']['trainer'])

    # Start training
    logger.info("Begin training...")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=cfg['model']['resume_training_from_ckpt']['path'])
    logger.info("Training ended succesfully...")

    # Test:
    logger.info("Test model... Dataloaders are displayed in the order: Train, validation and test.")
    trainer.test(ckpt_path="best", dataloaders=[train_dataloader, val_dataloader, test_dataloader])
    logger.info("Testing ending succesfully")

if __name__ == '__main__':
    main()
