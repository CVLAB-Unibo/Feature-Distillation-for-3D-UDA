import os, sys
os.environ['WANDB_SILENT']="true"
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch.nn.functional as F
import open3d as o3d
import wandb
import glob

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
pl.seed_everything(42)
#%%

dataset_name = sys.argv[1]
root = "data/" + dataset_name
project_name=sys.argv[3]
entity=sys.argv[2]

categories = glob.glob(os.path.join(root, "*"))
categories = [c.split(os.path.sep)[-1] for c in categories]
categories = sorted(categories)

def create_dataset(pts_list):
  pc_list = []
  lbl_list = []
  pc_path = []
  for elem in pts_list:
      pc = o3d.io.read_point_cloud(elem)
      pc = np.array(pc.points).astype(np.float32)
      pc_list.append(pc)
      pc_path.append(elem.replace(root, ""))
      lbl_list.append(categories.index(elem.split("/")[-3]))
  return np.stack(pc_list), np.stack(lbl_list), pc_path

with wandb.init(project=project_name, job_type="upload_data", save_code=False, entity=entity) as run:

  pts_list_train = glob.glob(os.path.join(root, "*", "train", "*.ply"))
  pts_list_test = glob.glob(os.path.join(root, "*", "test", "*.ply"))

  train = create_dataset(pts_list_train)
  datasets = [train]
  names = ["train"]

  if len(pts_list_test)!=0:
    test = create_dataset(pts_list_test)
    datasets.append(test)
    names.append("test")
  
  raw_data = wandb.Artifact(
      dataset_name, 
      type="dataset",
      description=dataset_name + " PL",
      metadata={
        "sizes": [len(dataset[0]) for dataset in datasets]})

  for name, data in zip(names, datasets):
      with raw_data.new_file(name + ".npz", mode="wb") as file:
          np.savez(file, x=data[0], y=data[1], path=data[2])

  run.log_artifact(raw_data)