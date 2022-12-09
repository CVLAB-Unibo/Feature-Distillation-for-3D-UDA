import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from datasets.classification_dataset import ClassificationDataset
from hesiod import hcfg

def worker_init_fn(worker_id):                                                          
    np.random.seed(torch.randint(0, 1000, ()))

def collate_fn(list_data):
    original_coordinates = torch.tensor([d["original_coordinates"] for d in list_data], dtype=torch.float32)
    strongly_augmented = torch.tensor([d["stronly_augmented"] for d in list_data], dtype=torch.float32)
    weakly_augmented = torch.tensor([d["weakly_augmented"] for d in list_data], dtype=torch.float32)
    labels = torch.tensor([d["labels"] for d in list_data], dtype=torch.int64)
    lbl_init = torch.tensor([d["labels_init"] for d in list_data], dtype=torch.int64)
    paths = np.array([d["paths"] for d in list_data])
    true_lbl = torch.tensor([d["true_labels"] for d in list_data], dtype=torch.int64)
    
    return {
        "original_coordinates": original_coordinates,
        "weakly_aug": weakly_augmented,
        "strongly_aug": strongly_augmented,
        "labels": labels,
        "lbl_init": lbl_init,
        "paths": paths,
        "true_labels": true_lbl,
    }

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.files = []

        self.train_ds_target = ClassificationDataset(
            hcfg("dataset_target.name"),
            None,
            "train",
            target_domain=True
        )

        self.train_ds = ClassificationDataset(
            hcfg("dataset_source.name"),
            None,
            "train",
            occlusions= hcfg("occlusions")
        )

        self.val_ds_source = ClassificationDataset(
            hcfg("dataset_source.name"),
            None,
            "test",
            val_split=True
        )

        self.val_ds_target = ClassificationDataset(
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
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
        return train_dl

    def train_dataloader_target(self):
        train_dl = DataLoader(
            self.train_ds_target,
            batch_size=hcfg("train_batch_size_target"),
            shuffle=True,
            num_workers=hcfg("num_workers"),
            collate_fn=self.collate,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            # drop_last=True,
        )
        return train_dl        

    def val_dataloader(self):
        val_dl_source = DataLoader(
            self.val_ds_source,
            batch_size=hcfg("val_batch_size"),
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate,
            pin_memory=False,
        )
        val_dl_target = DataLoader(
            self.val_ds_target,
            batch_size=hcfg("val_batch_size"),
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate,
            pin_memory=False,
        )
        return [val_dl_source, val_dl_target]

    def test_dataloader(self):
        test_dl_target = DataLoader(
            self.val_ds_target,
            batch_size=hcfg("val_batch_size"),
            shuffle=False,
            num_workers=hcfg("num_workers"),
            collate_fn=self.collate,
            pin_memory=False,
        )
        return test_dl_target
        
    def get_val_samples(self):
        train = self.train_dataloader()
        source, target = self.val_dataloader()
        return next(iter(train)), next(iter(source)), next(iter(target)) 

    def __len__(self):
        dl_temp = self.train_dataloader()
        return len(dl_temp)
