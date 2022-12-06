import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from datasets.contrastive_dataset import ContrastiveDataset
from hesiod import hcfg

def worker_init_fn(worker_id):                                                          
    np.random.seed(torch.randint(0, 1000, ()))

def collate_fn(list_data):
    cooridnates_augmented = torch.tensor([d["cooridnates_augmented"] for d in list_data], dtype=torch.float32)
    labels = torch.tensor([d["labels"] for d in list_data], dtype=torch.int64)
    original_coordinates = torch.tensor([d["original_coordinates"] for d in list_data], dtype=torch.float32)
    paths = np.array([d["paths"] for d in list_data])

    # true_lbl = torch.tensor([d["true_labels"] for d in list_data], dtype=torch.int64)
    # masks = torch.tensor([d["mask"] for d in list_data], dtype=torch.bool)
    return {
        "original_coordinates": original_coordinates,
        "cooridnates_augmented": cooridnates_augmented,
        "labels": labels,
        "paths": paths,
        # "true_labels": true_lbl,
        # "masks": masks
    }

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.files = []

        self.train_ds = ContrastiveDataset(
            hcfg("dataset_source.name"),
            None,
            "train",
        )

        self.val_ds_source = ContrastiveDataset(
            hcfg("dataset_source.name"),
            None,
            "test"
        )

        self.val_ds_target = ContrastiveDataset(
            hcfg("dataset_target.name"),
            None,
            "test"
        )

        self.collate = collate_fn

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=hcfg("train_batch_size_source"),
            shuffle=True,
            num_workers=hcfg("num_workers"),
            collate_fn=self.collate,
            # pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
        return train_dl

    def val_dataloader(self):
        val_dl_source = DataLoader(
            self.val_ds_source,
            batch_size=hcfg("val_batch_size"),
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate,
            # pin_memory=True,
            # drop_last=True,
        )
        val_dl_target = DataLoader(
            self.val_ds_target,
            batch_size=hcfg("val_batch_size"),
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate,
            # pin_memory=True,
            # drop_last=True,
        )
        return [val_dl_source, val_dl_target]

    def test_dataloader(self):
        test_dl_target = DataLoader(
            self.val_ds_target,
            batch_size=hcfg("val_batch_size"),
            shuffle=False,
            num_workers=hcfg("num_workers"),
            collate_fn=self.collate,
            pin_memory=True,
        )
        return test_dl_target
        
    def get_val_samples(self):
        train = self.train_dataloader()
        source, target = self.val_dataloader()
        return next(iter(train)), next(iter(source)), next(iter(target)) 

    def __len__(self):
        dl_temp = self.train_dataloader()
        return len(dl_temp)
