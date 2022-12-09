from pickle import TRUE
import torch
from torch import nn
import pytorch_lightning as pl
from torch._C import device
from networks.factory import get_model, GCN
from utils.losses import get_loss_fn, Distillation_Loss
from hesiod import hcfg
from hesiod import get_cfg_copy
from utils.optimizers import get_optimizer
import numpy as np
import wandb
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm

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
        self.gnn = GCN(num_features=1024, num_classes=10)
        self.optimizer_gnn = torch.optim.Adam(self.gnn.parameters(), lr=0.001, weight_decay=5e-4)
        self.criterion_gnn = torch.nn.CrossEntropyLoss()
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
        # Use the "true" average until the exponential average is more correct
        alpha_teacher = min(1 - 1 / (self.global_step + 1), self.alpha_teacher)
        for ema_param, param in zip(self.ema.encoder.parameters(), self.net.encoder.parameters()):
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

    def reinitialize(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)

        self.gnn.apply(weights_init)

    def loss(self, xb, xb_weak, yb, train_mask=None, student_target=None, teacher_target=None):

        with torch.cuda.amp.autocast():
            embeddings, output = self.net(xb, embeddings=True)

            if train_mask is not None:
                loss = (self.loss_fn(output, yb)*train_mask).mean()
            else:
                loss = self.loss_fn(output, yb).mean()
            ce_mt = 0
            
            consistency_weight = 0
            dino_loss = 0
            
            if xb_weak is not None:
                self.ema.eval()
                embeddings_std, _= self.net(student_target, embeddings=True)
                embeddings_std_src_tgt = torch.cat([embeddings, embeddings_std], dim=0)

                embeddings_tch, _ = self.ema( torch.cat([xb_weak, teacher_target]),  embeddings=True)
                consistency_weight = self.get_current_consistency_weight(self.hparams.max_weight, self.hparams.epochs)
                dino_loss = self.ssl_loss(embeddings_std_src_tgt, embeddings_tch)
                ce_mt += dino_loss*consistency_weight
            
            # return embeddings_std, output, loss+ce_mt, dino_loss
        # else:
        return embeddings, output, loss+ce_mt, dino_loss

    def training_step(self, batch, batch_idx):
        if batch_idx ==0:
            self.train_graph = torch.zeros((0, 1024)).to(self.device)
            self.train_scores = torch.zeros((0, 10)).to(self.device)
            self.train_scores_ema = torch.zeros((0, 10)).to(self.device)
            self.train_pl = torch.zeros((0)).to(self.device)
            self.train_pl_init = torch.zeros((0)).to(self.device)
            self.train_pl_ema = torch.zeros((0)).to(self.device)
            self.true_labels = torch.zeros((0)).to(self.device)
            self.pl_paths = np.empty((0),  dtype='<U44')
            if self.current_epoch == 0:
                self.confident_paths = []

        opt = self.optimizers()
        lr_scheduler = self.lr_schedulers()
        opt.zero_grad()
        weight = self.get_current_consistency_weight(self.hparams.max_weight, length=self.hparams.epochs)
        coords = batch["strongly_aug"]
        coords_weak = batch["weakly_aug"]
        labels = batch["labels"]
        lbl_init = batch["lbl_init"]
        true_labels = batch["true_labels"]
        paths = batch["paths"]

        pl_indeces = np.flatnonzero(np.core.defchararray.find(paths,"_st.ply")!=-1)
        train_mask = torch.ones_like(labels).float()
        for p_idx in pl_indeces:
            if paths[p_idx] not in self.confident_paths:
                train_mask[p_idx] = hcfg("lambda")
        try:
            target_batch = next(self.target_dl)
        except:
            self.target_dl = iter(self.dm.train_dataloader_target())
            target_batch = next(self.target_dl)

        student_target = target_batch["strongly_aug"].to(self.device)
        teacher_target = target_batch["weakly_aug"].to(self.device)

        embeddings, predictions, loss, dino_loss = self.loss(coords, coords_weak, labels, train_mask, student_target, teacher_target)

        if len(pl_indeces)>1:
            pl_labels = labels[pl_indeces]
            pl_labels_init = lbl_init[pl_indeces]
            pl_paths = paths[pl_indeces]
            true_labels = true_labels[pl_indeces]
            self.true_labels = torch.cat([self.true_labels, true_labels], dim=0)

            pl_embeddings = embeddings[pl_indeces].detach().clone()
            pl_prediction = predictions.argmax(-1)[pl_indeces].detach().clone()

            scores = F.softmax(predictions[pl_indeces], dim=-1).detach().clone()
            self.pl_paths = np.concatenate((self.pl_paths, pl_paths))

            self.train_pl = torch.cat([self.train_pl, pl_prediction], dim=0)
            self.train_pl_init = torch.cat([self.train_pl_init, pl_labels_init], dim=0)
            self.train_graph = torch.cat((self.train_graph, pl_embeddings), dim=0)
            self.train_scores = torch.cat((self.train_scores, scores), dim=0)

        self.train_acc(F.softmax(predictions, dim=1), labels)

        if self.global_step % 100 == 0 and self.global_step != 0:
            self.logger.experiment.log(
                {
                    "train/loss": loss.item(),
                    "train/weight": weight,
                    "train/dino_loss": dino_loss.item(),
                }
            )

        interval = 5
        if batch_idx == len(self.dm)-1 and self.current_epoch%interval==0 and self.current_epoch!=0:
            mean_nn = []
            self.gnn.train()
            nodes = len(self.train_graph)
            tau = 0.95
            clip = 10
            corr_matrix = np.corrcoef(self.train_graph.cpu())
            corr_matrix_bin = corr_matrix.copy()

            corr_matrix = torch.from_numpy(corr_matrix).to(self.device)
            corr_matrix_bin[corr_matrix_bin<=tau] = 0
            corr_matrix_bin[corr_matrix_bin>tau] = 1
            corr_matrix_bin = torch.from_numpy(corr_matrix_bin).bool().to(self.device)

            edges = []
            for i in range(nodes):
                indexes = torch.argsort(corr_matrix[i], descending=True)                
                tau_mask = corr_matrix_bin[i]==1
                tau_mask = tau_mask[indexes]
                indexes = indexes[tau_mask][:clip]
                mean_nn.append(len(indexes))
                for index in set(indexes):
                    edges.append((i, index))
            edge_index = torch.tensor(edges).transpose(1, 0).long().to(self.device)
            print(np.array(mean_nn).mean())
            print(" ----------- TRAINING 2 STARTS -----------")
            self.reinitialize()
            self.optimizer_gnn = torch.optim.Adam(self.gnn.parameters(), lr=0.001, weight_decay=5e-4)
            for _ in tqdm(range(1000)):
                self.optimizer_gnn.zero_grad()  # Clear gradients.
                column_vec = torch.rand(self.train_graph.shape[0]).reshape(-1, 1)<0.2
                column_vec = column_vec.repeat(1, 10)
                pl_features = self.train_scores.clone()
                pl_features[column_vec] = 0
                _, out = self.gnn(self.train_graph, pl_features, edge_index) # Perform a singùle forward pass.
                loss_gnn = self.criterion_gnn(out, self.train_pl.long())  # Compute the loss solely based on the training nodes.
                self.manual_backward(loss_gnn)
                self.optimizer_gnn.step()  # Update parameters based on gradients.
            
            self.class_scores_gnn, self.scores_gnn, self.predictions_gnn = self.test_gnn(self.gnn, self.train_graph, self.train_scores, edge_index)
            new_labels = {}

            new_confident_gnn = self.filter_data(self.predictions_gnn.cpu(), self.scores_gnn.cpu(), 1-self.current_epoch/100, 10)                     #(1-weight per selezionare più sample target)
            for path, new_label in zip(self.pl_paths[new_confident_gnn], self.predictions_gnn[new_confident_gnn]):
                new_labels[path] = new_label.item()
                if path not in self.confident_paths:
                    self.confident_paths.append(path) 
            self.dm.train_ds.update_labels(new_labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(opt)
        self.scaler.update()
        lr_scheduler.step()
        self.update_ema_variables()

        return {"loss": loss}

    def filter_data(self, predictions, probs, p, num_classes):
        thres = []
        for i in range(num_classes):
            x = probs[predictions==i]
            if len(x) == 0:
                thres.append(0)
                continue        
            x, _ = torch.sort(x)
            take = int(round(len(x)*p))
            if take == len(x):
                take -= 1
            thres.append(x[take])
        thres = torch.tensor(thres)
        selected = torch.ones_like(probs, dtype=torch.bool)

        for i in range(len(predictions)):
            for c in range(num_classes):
                if probs[i]<thres[c]*(predictions[i]==c):
                    selected[i] = False
        return selected

    def training_epoch_end(self, outputs):

        self.logger.experiment.log(
            {
                "train/accuracy": self.train_acc.compute(),
            },
            commit=True,
            step=self.global_step,
        )
        self.train_acc.reset()

    def test_gnn(self, gnn, graph, train_scores, edge_index):
        gnn.eval()
        with torch.no_grad():
            _, out = gnn(graph, train_scores, edge_index)
            out = F.softmax(out, dim=1)
            scores, pred = out.max(dim=1)  # Use the class with highest probability.
        return out, scores, pred

    def validation_step(self, batch, batch_idx, dataloader_idx):
        coords = batch["weakly_aug"]
        labels = batch["labels"]
        with torch.no_grad():
            if dataloader_idx == 0:
                embeddings, predictions, loss, _ = self.loss(coords, None, labels)
                self.valid_acc_source(F.softmax(predictions, dim=1), labels)
            if dataloader_idx == 1:
                embeddings, predictions, loss, _ = self.loss(coords, None,labels)
                self.valid_acc_target(F.softmax(predictions, dim=1), labels)

        predictions = (predictions.argmax(dim=1), labels)
        return {"loss": loss, "predictions": predictions,
                "embeddings": embeddings.cpu().numpy(), "labels": labels
        }

    def validation_epoch_end(self, validation_step_outputs):

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
                [x["embeddings"].squeeze() for x in source_data]
            )
            source_labels = np.concatenate(
                [x["labels"].squeeze().cpu().numpy() for x in source_data]
            )
            self.writer.add_embedding(source_embeddings, metadata=source_labels, global_step=self.global_step, tag="source")

            target_data = validation_step_outputs[1]
            target_embeddings = np.concatenate(
                [x["embeddings"].squeeze() for x in target_data]
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
        self.log("final_accuracy", target_accuracy)

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
        opt = get_optimizer(list(self.net.parameters()))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hcfg("lr"), epochs=self.hparams.epochs, steps_per_epoch=len(self.dm))
        return [opt], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]