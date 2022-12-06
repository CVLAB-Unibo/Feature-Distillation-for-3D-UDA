
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from networks.pointnet import transform_net, conv_2d, fc_layer
# from PyTorchEMD.emd import earth_mover_distance
from utils.losses import ChamferLoss

class Pointnet_encoder(nn.Module):
    def __init__(self, device="cpu", feat_dims=512):
        super(Pointnet_encoder, self).__init__()
        norm_layer=nn.BatchNorm2d
        self.trans_net1 = transform_net(3,3, device, norm_layer=norm_layer)
        self.trans_net2 = transform_net(64,64, device, norm_layer=norm_layer)
        self.conv1 = conv_2d(3, 64, 1, norm_layer=norm_layer)
        self.conv2 = conv_2d(64, 64, 1, norm_layer=norm_layer)
        self.conv3 = conv_2d(64, 64, 1, norm_layer=norm_layer)
        self.conv4 = conv_2d(64, 128, 1, norm_layer=norm_layer)
        self.conv5 = conv_2d(128, feat_dims, 1, norm_layer=norm_layer)

    def forward(self, x, embeddings=False):
        x = x.permute(0, 2, 1).unsqueeze(dim=3) #B, C, N, 1 
        
        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        return x

class Simple_Decoder(nn.Module):
    def __init__(self, feat_dims=512):
        super(Simple_Decoder, self).__init__()
        self.m = 2048  # 45 * 45.
        self.folding1 = nn.Sequential(
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, self.m*3, 1),
        )
        
    def forward(self, x):
        x = self.folding1(x)           # (batch_size, 3, num_points)
        x = x.reshape(-1, 2048, 3)          # (batch_size, num_points ,3)
        return x


class ReconstructionNet(nn.Module):
    def __init__(self, device, feat_dims=1024):
        super(ReconstructionNet, self).__init__()
        self.encoder = Pointnet_encoder(device, feat_dims)
        self.decoder = Simple_Decoder(feat_dims)

    def forward(self, input, embeddings=False):
        feature = self.encoder(input)
        if embeddings:
            return feature
        else:
            output = self.decoder(feature)
            return output

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())