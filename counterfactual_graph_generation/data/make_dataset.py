import logging

logger = logging.getLogger(__name__)

import os
import hydra
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

import counterfactual_graph_generation.data.datasets as ds
import counterfactual_graph_generation.data.dataset_utils as cgf_utils


class CounterfactualGraphDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            graph_size_limit: int,
            download_data: bool,
            dense_data_representation: bool = False,
            seed: int = 12345,
            batch_size: int = 16,
            num_cpu_workers: int = 1,
            use_split_idx: bool = True
        ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def load_dataset(self):
        dataset_name = self.hparams.dataset_name
        if dataset_name == 'mutagenicity':
            dataset = ds.Mutagenicity('data/mutagenicity', graph_size_limit=self.hparams.graph_size_limit, dense_data_representation=self.hparams.dense_data_representation)
        elif dataset_name == 'aids':
            dataset = ds.AIDS('data/aids', graph_size_limit=self.hparams.graph_size_limit, dense_data_representation=self.hparams.dense_data_representation)
        elif dataset_name == 'nci1':
            dataset = ds.NCI1('data/nci1', graph_size_limit=self.hparams.graph_size_limit, dense_data_representation=self.hparams.dense_data_representation)
        elif dataset_name == 'proteins':
            dataset = ds.PROTEINS('data/proteins', graph_size_limit=self.hparams.graph_size_limit, dense_data_representation=self.hparams.dense_data_representation)
        elif dataset_name == 'qm9':
            dataset = ds.custom_QM9('data/qm9', graph_size_limit=self.hparams.graph_size_limit, dense_data_representation=self.hparams.dense_data_representation)
        elif dataset_name == "ogbg-molhiv":
            dataset = ds.OgbMolHiv('data/ogb-molhiv', graph_size_limit=self.hparams.graph_size_limit, dense_data_representation=self.hparams.dense_data_representation)
        elif dataset_name == "ogbg-molpcba":
            raise ValueError(f'Dataset {dataset_name} not implemented yet. ')
        else:
            raise ValueError(f'Dataset {dataset_name} not supported. ')
        return dataset

    def setup(self): # set the dataset to the correct fold.
        """ Splits the dataset in train, val and test and initializes and returns Dataset objects """
        self.dataset = self.load_dataset()
        split_idx = self.dataset.get_idx_split()
        if split_idx is not None and self.hparams.use_split_idx:
            self.data_train, self.data_val, self.data_test = Subset(self.dataset, split_idx['train']), Subset(self.dataset, split_idx['valid']), Subset(self.dataset, split_idx['test'])
        else:
            dataset_train, dataset_val_and_test = train_test_split(self.dataset, test_size=0.2, random_state=self.hparams.seed, shuffle=True)
            dataset_val, dataset_test = train_test_split(dataset_val_and_test, test_size=0.5, random_state=self.hparams.seed, shuffle=False)
            self.data_train, self.data_val, self.data_test = dataset_train, dataset_val, dataset_test

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_cpu_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_cpu_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_cpu_workers, shuffle=False) #len(self.data_test)

    def prepare_data(self):
        """ Download data to RAW and prepare data in the associated repositories. If the data has already been downloaded, then only process the data and save in """
        pass


@hydra.main(config_path="../../config/dataset", config_name="ogbg-molhiv.yaml", version_base="1.2")
def main(cfg):
    logger.info("Load data module, prepare- and setup data...")
    seed_everything(seed=cfg['seed'], workers=True)
    data_module = CounterfactualGraphDataModule(**cfg)
    data_module.prepare_data()
    data_module.setup()
    train_dataloader = data_module.train_dataloader
    generator = iter(train_dataloader())
    item = next(generator)
    logger.info("Print first data object:")
    print(item)
    return

if __name__ == '__main__':
    main()
