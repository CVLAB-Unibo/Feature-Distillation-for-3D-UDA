import pytorch_lightning as pl
import torch
import wandb
import numpy as np
from hesiod import hcfg
import torch.nn.functional as F

class PCPredictionLogger(pl.Callback):
    def __init__(self, dataModule):
        super().__init__()
        self.dataModule = dataModule

    def log_point_clouds(self, trainer, pl_module, batch, split):
        
        if split=="train":
            pcs = batch["strongly_aug"].squeeze().cpu().numpy()[:6, ...]
        else:
            pcs = batch["weakly_aug"].squeeze().cpu().numpy()[:6, ...]
        
        trainer.logger.experiment.log(
                {"point_clouds_" + split: [wandb.Object3D(pc) for i, pc in enumerate(pcs)]},
        )

        if hasattr(pl_module, 'log_prediction'):
            _, pcs = pl_module(batch["weakly_aug"].to(pl_module.device))
            pcs = pcs[:6, ...].cpu().numpy()
            trainer.logger.experiment.log(
                {"point_clouds_REC" + split: [wandb.Object3D(pc) for i, pc in enumerate(pcs)]})

    def log_errors(self, trainer, pl_module, val_dataloader):
        columns=["id", "image", "guess", "truth"]
        for digit in range(10):
            columns.append("score_" + str(digit))
        test_table = wandb.Table(columns=columns)
        _id = 0

        for i, batch in enumerate(val_dataloader):
                if i>20:
                    break
                coords_b = batch["weakly_aug"].to(pl_module.device)
                labels_b = batch["labels"]
                    
                with torch.no_grad():
                    _, out_t = pl_module(coords_b[:, :1024, :])
                    logits = F.softmax(out_t, dim=1).cpu()
                    probs_b, predictions_b = torch.max(logits, dim=1)
                    probs_b = torch.argsort(probs_b)[:100]
                
                for i, l, p, s in zip(coords_b[probs_b], labels_b[probs_b], predictions_b[probs_b], logits[probs_b]):
                    # add required info to data table:
                    # id, image pixels, model's guess, true label, scores for all classes
                    img_id = str(_id) + "_"
                    test_table.add_data(img_id, wandb.Object3D(i.cpu().numpy()), p, l, *s.numpy())
                    _id += 1
        trainer.logger.experiment.log({"test_predictions" : test_table})        

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 0 and trainer.global_step>0:
            samples_train, samples_source, samples_target = self.dataModule.get_val_samples()
            self.log_point_clouds(trainer, pl_module, samples_train, "train")
            self.log_point_clouds(trainer, pl_module, samples_source, "source")
            self.log_point_clouds(trainer, pl_module, samples_target, "target")

        # if trainer.current_epoch == trainer.max_epochs-1:
        #     _, val_dataloader = self.dataModule.val_dataloader()
        #     self.log_errors(trainer, pl_module, val_dataloader)