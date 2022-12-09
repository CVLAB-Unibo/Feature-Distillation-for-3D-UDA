import torch
from torch import nn
import pytorch_lightning as pl
from networks.factory import get_model
from utils.losses import get_loss_fn, Distillation_Loss
from hesiod import hcfg
from hesiod import get_cfg_copy
from utils.optimizers import get_optimizer
import numpy as np
import wandb
import torch.nn.functional as F
import torchmetrics

from torch.utils.tensorboard import SummaryWriter

class Classifier(pl.LightningModule):
    def __init__(self, dm, device):
        super().__init__()

        self.net = get_model(device)
        self.ema = get_model(device)
        self.dm = dm
        self.best_accuracy_source = 0
        self.best_accuracy_target = 0
        self.loss_fn = get_loss_fn()
        self.train_acc = torchmetrics.Accuracy('multiclass', num_classes=hcfg("num_classes"))
        self.valid_acc_source = torchmetrics.Accuracy('multiclass', num_classes=hcfg("num_classes"))
        self.valid_acc_target = torchmetrics.Accuracy('multiclass', num_classes=hcfg("num_classes"))
        self.alpha_teacher = 0.999
        self.automatic_optimization = False
        self.save_hyperparameters(get_cfg_copy())
        self.target_dl = iter(dm.train_dataloader_target())
        self.ssl_loss = Distillation_Loss(self.hparams.ssl_classes, teacher_temp=self.hparams.teacher_temp, student_temp=self.hparams.student_temp)
        self.scaler = torch.cuda.amp.GradScaler()

    def create_ema_model(self):
        for param in self.ema.parameters():
            param.detach_()
        mp = list(self.net.parameters())
        mcp = list(self.ema.parameters())
        n = len(mp)
        for i in range(0, n):
            mcp[i].data[:] = mp[i].data[:].clone()

    def update_ema_variables(self):
        alpha_teacher = min(1 - 1 / (self.global_step + 1), self.alpha_teacher)
        for ema_param, param in zip(self.ema.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)

        for t, s in zip(self.ema.buffers(), self.net.buffers()):
            if not t.dtype == torch.int64:
                t.data.mul_(alpha_teacher).add_(s.data, alpha=1 - alpha_teacher)
    
    def on_train_start(self):
        self.writer = SummaryWriter(wandb.run.dir)
        self.create_ema_model()

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, weight, length):
        return weight * self.sigmoid_rampup(self.current_epoch, length)

    def forward(self, x, use_ema=False):
        if use_ema:
            embeddings, output = self.ema(x, embeddings=True)
        else:
            embeddings, output = self.net(x, embeddings=True)

        return embeddings, output

    def loss(self, xb, xb_weak, yb, student_target=None, teacher_target=None):

        with torch.cuda.amp.autocast():
            embeddings, output = self.net(xb, embeddings=True)
            loss = self.loss_fn(output, yb)

            ce_mt = 0
            consistency_weight = 0
            dino_loss = 0
            if student_target is not None:
                self.ema.eval()
                embeddings_std, _= self.net(student_target, embeddings=True)
                embeddings_std_src_tgt = torch.cat([embeddings, embeddings_std], dim=0)
                
                embeddings_tch, _ = self.ema( torch.cat([xb_weak, teacher_target]),  embeddings=True)

                consistency_weight = self.get_current_consistency_weight(self.hparams.max_weight, self.hparams.epochs)
                dino_loss = self.ssl_loss(embeddings_std_src_tgt, embeddings_tch)
                ce_mt += dino_loss*consistency_weight
            
        return embeddings, output, loss+ce_mt, dino_loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr_scheduler = self.lr_schedulers()
        opt.zero_grad()
        weight = self.get_current_consistency_weight(self.hparams.max_weight, length=self.hparams.epochs)
        coords = batch["strongly_aug"]
        coords_weak = batch["weakly_aug"]
        labels = batch["labels"]

        try:
            target_batch = next(self.target_dl)
        except:
            self.target_dl = iter(self.dm.train_dataloader_target())
            target_batch = next(self.target_dl)

        student_target = target_batch["strongly_aug"].to(self.device)
        teacher_target = target_batch["weakly_aug"].to(self.device)

        _, predictions, loss, dino_loss = self.loss(coords, coords_weak, labels, student_target, teacher_target)
        self.scaler.scale(loss).backward()
        self.scaler.step(opt)
        self.scaler.update()
        lr_scheduler.step()

        self.train_acc(F.softmax(predictions, dim=1), labels)

        if self.global_step % 100 == 0 and self.global_step != 0:
            self.logger.experiment.log(
                {
                    "train/loss": loss.item(),
                    "train/weight": weight,
                    "train/dino_loss": dino_loss.item(),
                }
            )

        self.update_ema_variables()
        return loss

    def training_epoch_end(self, outputs):

        self.logger.experiment.log(
            {
                "train/accuracy": self.train_acc.compute(),
            },
            commit=True,
            step=self.global_step,
        )
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx):
        coords = batch["weakly_aug"]
        labels = batch["labels"]

        with torch.no_grad():
            if dataloader_idx == 0:
                embeddings, predictions, loss, _ = self.loss(coords, None, labels)
                self.valid_acc_source(F.softmax(predictions, dim=1), labels)
            if dataloader_idx == 1:
                embeddings, predictions, loss, _ = self.loss(coords, None, labels)
                self.valid_acc_target(F.softmax(predictions, dim=1), labels)

        predictions = (predictions.argmax(dim=1), labels)
        return {"loss": loss, "predictions": predictions,
                "embeddings": embeddings, "labels": labels
        }

    def validation_epoch_end(self, validation_step_outputs):
        # print("\n")

        valid_acc_source = self.valid_acc_source.compute().item()
        print("SOURCE:", valid_acc_source)
        source_data = validation_step_outputs[0]
        predictions_source = np.concatenate(
            [x["predictions"][0].cpu().numpy() for x in source_data]
        )
        label_source = np.concatenate(
            [x["predictions"][1].cpu().numpy() for x in source_data]
        )
        avg_loss_source = np.mean([x["loss"].item() for x in source_data])

        valid_acc_target = self.valid_acc_target.compute().item()
        print("TARGET:", valid_acc_target)
        target_data = validation_step_outputs[1]
        predictions_target = np.concatenate(
            [x["predictions"][0].cpu().numpy() for x in target_data]
        )
        label_target = np.concatenate(
            [x["predictions"][1].cpu().numpy() for x in target_data]
        )
        avg_loss = np.mean([x["loss"].item() for x in target_data])

        if valid_acc_target >= self.best_accuracy_target and self.global_step != 0:
            self.logger.log_metrics({"best_accuracy": valid_acc_target})
            self.best_accuracy_target = valid_acc_target

        # take best model according to source test set
        if valid_acc_source >= self.best_accuracy_source and self.global_step != 0:
            self.logger.log_metrics({
                "final_accuracy": valid_acc_target,
                "best_accuracy_source": valid_acc_source
            }
            )
            self.best_accuracy_source = valid_acc_source

        self.logger.experiment.log(
            {
                "valid/source_loss": avg_loss_source,
                "valid/target_loss": avg_loss,
                "valid/target_accuracy": valid_acc_target,
            },
            commit=False,
        )
        self.log("valid/source_accuracy", valid_acc_source)
        self.valid_acc_target.reset()
        self.valid_acc_source.reset()

        # write embeddings on last epochs for source and target validation data
        if self.current_epoch==hcfg("epochs")-1:
            self.logger.experiment.log(
            {
                "confusion_source": wandb.plot.confusion_matrix(
                    preds=predictions_source,
                    y_true=label_source,
                    class_names=self.dm.train_ds.categories,
                ),
                "confusion_target": wandb.plot.confusion_matrix(
                    preds=predictions_target,
                    y_true=label_target,
                    class_names=self.dm.train_ds.categories,
                ),
            },
            commit=False,
            )

            source_data = validation_step_outputs[0]
            source_embeddings = np.concatenate(
                [x["embeddings"].squeeze().cpu().numpy() for x in source_data]
            )
            source_labels = np.concatenate(
                [x["labels"].squeeze().cpu().numpy() for x in source_data]
            )
            self.writer.add_embedding(source_embeddings, metadata=source_labels, global_step=self.global_step, tag="source")

            target_data = validation_step_outputs[1]
            target_embeddings = np.concatenate(
                [x["embeddings"].squeeze().cpu().numpy() for x in target_data]
            )
            target_labels = np.concatenate(
                [x["labels"].squeeze().cpu().numpy() for x in target_data]
            )
            self.writer.add_embedding(target_embeddings, metadata=target_labels, global_step=self.global_step, tag="target")

    def on_train_end(self) -> None:
        self.writer.close()
        return super().on_train_end()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, 1)

    def test_epoch_end(self, outputs):
        target_accuracy = self.valid_acc_target.compute().item()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_accuracy_target"] = self.best_accuracy_target
        checkpoint["best_accuracy_source"] = self.best_accuracy_source
        checkpoint["net"] = self.net.state_dict()
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpointed_state):
        self.best_accuracy_target = checkpointed_state["best_accuracy_target"]
        self.best_accuracy_source = checkpointed_state["best_accuracy_source"]
        self.ema.load_state_dict(checkpointed_state["ema"])
        self.net.load_state_dict(checkpointed_state["net"])

    def configure_optimizers(self):
        opt = get_optimizer(self.net.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hcfg("lr"), epochs=self.hparams.epochs, steps_per_epoch=len(self.dm))
        return [opt], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]