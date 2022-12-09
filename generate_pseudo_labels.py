#%%
import sys
from hesiod.core import set_cfg
sys.path.append(".")
import os
os.environ['WANDB_SILENT']="true"

from sklearn.metrics import confusion_matrix, classification_report
import pickle 
import torch
from hesiod import hcfg, hmain
from pathlib import Path
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import torch_geometric.utils as utils
import open3d as o3d

pl.seed_everything(42)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
#%%

source_domain = sys.argv[1]
target_domain = sys.argv[2]

if source_domain=="modelnet":
    s = "m"
elif source_domain=="shapenet":
    s = "s"
elif source_domain=="scannet":
    s = "sc"

if target_domain=="modelnet":
    t = "m"
    tau = 0.95
    min_conf = 0.5

elif target_domain=="shapenet":
    t = "s"
    tau = 0.95
    min_conf = 0.6

elif target_domain=="scannet":
    t = "sc"
    tau = 0.95
    min_conf = 0.5

#modify based on architecture
dataset_name = source_domain+"_"+ target_domain + "_step1_dgcnn"
proj = s+"2"+t
root = Path(f"logs/{proj}/step1_DGCNN")
ckpt_path = root / "checkpoint/best.ckpt"
run_file_path = root / "run.yaml"
device = "cuda"
save_source_data = True
load_wandb = True

#%%

def print_scores(predictions, labels, target_names=None, verbose=False):
    cm = confusion_matrix(labels.cpu(), predictions.cpu())
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_scores = cmn.diagonal()
    print(cm_scores*100)
    print(f"accuracy: {cm.diagonal().sum()/cm.sum()*100:0.2f}")
    print(f"AVG accuracy: {(cm_scores*100).mean():0.2f}")
    if verbose:
        print(classification_report(labels.cpu(), predictions.cpu(), target_names=target_names, labels=np.arange(10)))

def filter_data(predictions, probs, p, num_classes):
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
    print(thres)   
    selected = torch.ones_like(probs, dtype=torch.bool)

    for i in range(len(predictions)):
        for c in range(num_classes):
            if probs[i]<thres[c]*(predictions[i]==c):
                selected[i] = False
    return selected

@hmain(base_cfg_dir=Path("cfg"), run_cfg_file=Path(run_file_path), create_out_dir=False, parse_cmd_line=False)
def main():

    run = wandb.init(
                job_type="pseudo_labels",
                project=hcfg("project_name"),
                name=target_domain,
                entity="cvpr",
                save_code=False,
    )
    from datamodules.classification_datamodule import DataModule
    from trainers.classification_trainerMT import Classifier

    set_cfg("aug", False)
    set_cfg("dataset_source.rotate", False)
    set_cfg("val_batch_size", 32)

    # set_cfg("dataset_source.name", source_domain)
    # dm = DataModule()
    # dataloader_train_source = dm.train_dataloader()
    
    set_cfg("dataset_source.name", target_domain)
    dm = DataModule()
    dataloader_train = dm.train_dataloader()
    _, dataloader_val = dm.val_dataloader()

    categories = dm.train_ds.categories
    model = Classifier.load_from_checkpoint(checkpoint_path=str(ckpt_path), device=device, dm=dm)
    
    model.to(device).eval()

    #get target data

    labels = torch.zeros((0))
    pc = torch.zeros((0, 2048, 3))
    predictions = torch.zeros((0))
    embeddings_matrix = torch.zeros((0, 1024))
    score_matrix = torch.zeros((0, 10))
    paths = []
    confidence = torch.zeros((0))

    for batch in tqdm(dataloader_train):
        coords_b = batch["weakly_aug"].to(device)
        coords_b_original = batch["original_coordinates"]
        labels_b = batch["labels"]
        paths_b = batch["paths"]
            
        with torch.no_grad():
            _, out_t = model.net(coords_b[:, :1024, :], embeddings=True)
            # feature_t = reconstructor(coords_b[:, :1024, :], embeddings=True)
            logits = F.softmax(out_t, dim=1)
            probs_b, predictions_b = torch.max(logits, dim=1)

            predictions = torch.cat([predictions, predictions_b.cpu()], dim=0).long()
            pc = torch.cat([pc, coords_b_original], dim=0)
        
            labels = torch.cat([labels, labels_b], dim=0)
            # embeddings_matrix = torch.cat((embeddings_matrix, feature_t.squeeze().cpu()), dim=0)
            score_matrix = torch.cat((score_matrix, logits.cpu()), dim=0)
            paths.extend(paths_b)
            confidence = torch.cat((confidence, probs_b.cpu()), dim=0)

    paths_st = []
    new_dataset_root = Path("data") / Path(dataset_name) 
    os.makedirs(new_dataset_root)
    classes = ["bathtub", "bed", "bookshelf", "cabinet", "chair", "lamp", "monitor", "plant", "sofa", "table"]
    for i, c in tqdm(enumerate(classes)):
        dirname = f"{new_dataset_root}/{c}/train"
        os.makedirs(dirname)
        coords_class_c = pc[predictions==i]
        for j, pc_class_c in enumerate(coords_class_c):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_class_c)
            o3d.io.write_point_cloud( dirname + f"/{target_domain}_{j}_st.ply", pcd)
            paths_st.append(f"/{c}/train/{target_domain}_{j}_st.ply")

    labels_source_train = torch.zeros(0)
    pcs_source_train = torch.zeros((0, 2048, 3))
    paths_source_train = []
    labels_source_val = torch.zeros(0)
    pcs_source_val = torch.zeros((0, 2048, 3))
    paths_source_val = []

    if save_source_data:
        for j, c in tqdm(enumerate(classes)):
            for phase in ["test", "train"]:

                os.makedirs(f"{new_dataset_root}/{c}/{phase}", exist_ok=True)
                for f in (new_dataset_root.parent / Path(source_domain) / Path(c) / Path(phase)).glob("*.ply"):
                    pcd = o3d.io.read_point_cloud(str(f), )
                    pc_class_c = np.array(pcd.points)
                    name = f.name
                    pcd.points = o3d.utility.Vector3dVector(pc_class_c)
                    o3d.io.write_point_cloud(f"{new_dataset_root}/{c}/{phase}/{name}", pcd )
                    if phase == "test":
                        labels_source_val = torch.cat([labels_source_val, torch.tensor(j).unsqueeze(0)], dim=0)
                        pcs_source_val = torch.cat([pcs_source_val, torch.from_numpy(pc_class_c).unsqueeze(0)], dim=0)
                        paths_source_val.append(f"/{c}/{phase}/{name}")
                    else:
                        labels_source_train = torch.cat([labels_source_train, torch.tensor(j).unsqueeze(0)], dim=0)
                        pcs_source_train = torch.cat([pcs_source_train, torch.from_numpy(pc_class_c).unsqueeze(0)], dim=0)
                        paths_source_train.append(f"/{c}/{phase}/{name}")

    names = ["test", "train"]
    dataset_test = [
        pcs_source_val,
        labels_source_val,
        labels_source_val,
        paths_source_val
    ]

    dataset_train = [
        torch.cat([pc, pcs_source_train], dim=0),
        torch.cat([predictions, labels_source_train], dim=0),
        torch.cat([labels, labels_source_train], dim=0),
        np.array(paths_st + paths_source_train)
    ]

    datasets = [
        dataset_test,
        dataset_train
    ]

    print("loading")
    if load_wandb:
        raw_data = wandb.Artifact(
            dataset_name, type="dataset",
            description=dataset_name + " PL",
            metadata={
                        "ckpt": str(ckpt_path),
                        "sizes": [len(dataset[0]) for dataset in datasets]})

        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".npz", mode="wb") as file:
                np.savez(file, x=data[0], y=data[1], realy=data[2], path=data[3])

        run.log_artifact(raw_data)

# %%

main()
# %%
