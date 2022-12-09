from hesiod import hcfg
import torch
import wandb
import glob
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

def get_model(device, tgt_decoder=False):

    if hcfg("task") == "classification":
        if hcfg("net.name") == "pointnet":
            from networks.pointnet import Pointnet
            model = Pointnet(num_class=hcfg("num_classes"), device=device, feat_dims=hcfg("feat_dims"))
        else:
            from networks.dgcnn import DGCNN
            model = DGCNN(num_class=hcfg("num_classes"), device=device, feat_dims=hcfg("feat_dims"))

    if hcfg("restore_weights") != "null":
        model_artifact = wandb.run.use_artifact(hcfg("restore_weights")+ ":latest", type='model')
        model_dir = model_artifact.download()
        model_paths = [path for path in glob.glob(model_dir+"/*.ckpt")] 
        saved_state_dict = torch.load(model_paths[0])
        
        if "state_dict" in saved_state_dict:
            saved_state_dict = saved_state_dict["state_dict"]
            new_params = model.state_dict().copy()
            if hcfg("test"):
                start_from = 1
            else:
                start_from = 2
            for it, i in enumerate(saved_state_dict):
                i_parts = i.split('.')
                if '.'.join(i_parts[start_from:]) in new_params.keys() and i_parts[0]!="decoder" and i_parts[0]!="head":
                    new_params['.'.join(i_parts[start_from:])] = saved_state_dict[i]
                    if it ==0:
                        print("####################### loading from" + model_paths[0] + " #######################")
            model.load_state_dict(new_params)
        else:        
            new_params = model.state_dict().copy()
            for it, i in enumerate(saved_state_dict):
                i_parts = i.split('.')
                if '.'.join(i_parts[1:]) in new_params.keys() and i_parts[0]!="decoder" and i_parts[0]!="head":
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                    if it ==0:
                        print("####################### loading from" + model_paths[0] + " #######################")
            model.load_state_dict(new_params)
    return model

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_features, num_features)
        self.conv2 = GCNConv(num_features, num_features)
        self.conv3 = GCNConv(num_features, num_classes)
        self.linear1 = nn.Linear(10, num_features)
    
    def forward(self, x, pseudo_y, edge_index, egde_values=None):

        y = self.linear1(pseudo_y)
        x = self.conv1(x+y, edge_index, egde_values)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        f = self.conv2(x, edge_index, egde_values)
        x = f.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index, egde_values)
        return f, x
   