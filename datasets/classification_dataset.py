import torch
import torch.utils.data as data
import os
import numpy as np
import glob
from .transformations import *
from hesiod import hcfg
import scipy
import scipy.ndimage
import scipy.interpolate

import random
import open3d as o3d
import wandb
# import torch_points3d.core.data_transform as T3D
from torch_geometric.data import Data
from torch_geometric.transforms import RandomScale, RandomTranslate

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class ElasticDistortion:
    """Apply elastic distortion on sparse coordinate space. First projects the position onto a 
    voxel grid and then apply the distortion to the voxel grid.
    Parameters
    ----------
    granularity: List[float]
        Granularity of the noise in meters
    magnitude:List[float]
        Noise multiplier in meters
    Returns
    -------
    data: Data
        Returns the same data object with distorted grid
    """

    def __init__(
        self, apply_distorsion= True, granularity = [0.2, 0.8], magnitude=[0.4, 1.6],
    ):
        assert len(magnitude) == len(granularity)
        self._apply_distorsion = apply_distorsion
        self._granularity = granularity
        self._magnitude = magnitude

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        coords = coords.numpy()
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords = coords + interp(coords) * magnitude
        return torch.tensor(coords).float()

    def __call__(self, data):
        # coords = data.pos / self._spatial_resolution
        if self._apply_distorsion:
            if random.random() < 0.95:
                for i in range(len(self._granularity)):
                    data.pos = ElasticDistortion.elastic_distortion(data.pos, self._granularity[i], self._magnitude[i],)
        return data

    def __repr__(self):
        return "{}(apply_distorsion={}, granularity={}, magnitude={})".format(
            self.__class__.__name__, self._apply_distorsion, self._granularity, self._magnitude,
        )


def farthest_point_sample_np(xyz, npoint):
    """
    Input:`
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape  # 1, 3, 2048
    centroids = np.zeros((B, npoint), dtype=np.int64)  # 1, 6
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(
            B, 3, 1
        )  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        dist = np.sum(
            (xyz - centroid) ** 2, 1
        )  # euclidean distance of points from the current centroid
        mask = (
            dist < distance
        )  # save index of all point that are closer than the current max distance
        distance[mask] = dist[
            mask
        ]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(
            distance, axis=1
        )  # get the index of the point farthest away
    return centroids, centroids_vals


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)

def remove(points, p_keep=0.7):
    rand_xyz = uniform_2_sphere()
    centroid = np.mean(points[:, :3], axis=0)
    points_centered = points[:, :3] - centroid

    dist_from_plane = np.dot(points_centered, rand_xyz)
    if p_keep == 0.5:
        mask = dist_from_plane > 0
    else:
        mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

    return points[mask, :]

class ClassificationDataset(data.Dataset):
    def __init__(self, name, root, split, val_split=False, occlusions=False, target_domain=False):
        super().__init__()

        self.split = split
        self.root = root
        self.pc_list = []
        self.pc_mask = []
        self.lbl_list = []
        self.pc_path = []
        self.pc_input_num = hcfg("pc_input_num")
        self.aug = hcfg("aug")
        self.name = name
        print("using", self.name + ":latest")
        data_artifact = wandb.run.use_artifact(self.name + ":latest")
        dataset = data_artifact.download()
        
        filename = split + ".npz"
        data = np.load(os.path.join(dataset, filename))
        self.pc_list = data["x"]
        self.lbl_list = data["y"]
        self.lbl_list_init = data["y"]
        self.pc_path = data["path"]
        self.target_domain = target_domain

        if "realy" in data:
            self.true_lbl_list = data["realy"]
        else:
            self.true_lbl_list = np.ones_like(self.lbl_list)*-1
        
        if val_split:
            print("shortening val set for SOURCE #################")
            selected = np.zeros_like(self.lbl_list)
            for c in range(hcfg("num_classes")):
                elems_class_c = np.where(self.lbl_list==c)[0][:400]
                selected[elems_class_c] = True
            selected = selected.astype(np.bool)
            self.pc_list = self.pc_list[selected]
            self.lbl_list = self.lbl_list[selected]
            self.true_lbl_list = self.true_lbl_list[selected]
            self.pc_path = self.pc_path[selected]
            print(len(self.pc_list))

        if len(self.pc_path)>1:
            self.categories = [c.split(os.path.sep)[-3] for c in self.pc_path]
            self.categories = sorted(set(self.categories))
            n_samples = []
            for c in range(10):
                n_samples.append(np.count_nonzero(self.lbl_list==c))

        print(f"{split} data num: {len(self.pc_list)}")
    
    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        lbl_init = self.lbl_list_init[idx]

        true_lbl = self.true_lbl_list[idx]
        pc = self.pc_list[idx]
        weakly_pc = pc.copy()
        pc_original = pc.copy()
        pc_path = self.pc_path[idx]

        # source domain 
        if not self.target_domain:
            # augment only for training data 
            if self.aug and self.split == "train":
                
                # strongly augmented pc for student
                pc = random_rotate_one_axis(pc, axis="z")
                dist = ElasticDistortion(apply_distorsion=True, granularity=[0.2, 0.8], magnitude=[0.4, 1.6])
                pc_data = Data(x=torch.tensor(pc), pos=torch.tensor(pc))
                pc_data = dist(pc_data)
                scale = RandomScale((0.7, 1.5))                
                pc_data = scale(pc_data)           
                pc = pc_data.pos.numpy()
                pc = jitter_point_cloud(pc)

                # weakly augmented pc for mean teacher
                weakly_pc = jitter_point_cloud(weakly_pc)

                if hcfg("occlusions") and ("shapenet" in self.name or "modelnet" in self.name) and not self.target_domain: 
                    pc = remove(pc)
        # target domain
        else:
                # strongly augmented pc for student
                dist = ElasticDistortion(apply_distorsion=True, granularity=[0.2, 0.8], magnitude=[0.4, 1.6])
                pc_data = Data(x=torch.tensor(pc), pos=torch.tensor(pc))
                pc_data = dist(pc_data)
                scale = RandomScale((0.7, 1.5))                
                pc_data = scale(pc_data)             
                pc = pc_data.pos.numpy()
                pc = jitter_point_cloud(pc)
                        
                # weakly augmented pc for mean teacher
                weakly_pc = jitter_point_cloud(weakly_pc)

        if pc.shape[0] > self.pc_input_num:
            if hcfg("sampling") == "fps":
                # apply Further Point Sampling
                pc = np.swapaxes(np.expand_dims(pc, 0), 1, 2)
                _, pc = farthest_point_sample_np(pc, self.pc_input_num)
                pc = np.swapaxes(pc.squeeze(), 1, 0).astype(np.float32)
            elif hcfg("sampling") == "uniform":
                ids = np.random.choice(
                    pc.shape[0], size=self.pc_input_num, replace=False
                )
                pc = pc[ids]
                ids = np.random.choice(
                    weakly_pc.shape[0], size=self.pc_input_num, replace=False
                )
                weakly_pc = weakly_pc[ids]
            else:
                pc = pc[: self.pc_input_num]

        return {
            "original_coordinates": pc_original,
            "stronly_augmented": pc,
            "weakly_augmented": weakly_pc,
            "labels": lbl,
            "labels_init": lbl_init,
            "true_labels": true_lbl,
            "paths": pc_path,
        }

    def __len__(self):
        return len(self.pc_list)

    def update_labels(self, dic_path_labels):
        for path, new_label in dic_path_labels.items():
            idx = np.nonzero(self.pc_path == path)[0][0]
            self.lbl_list[idx] = new_label

    def rotate_pc(self, name, pointcloud, lbl):
        if not (name == "shapenet" and self.categories[lbl] == "plant"):
            pointcloud = rotate_shape(pointcloud, "x", -np.pi / 2)
        return pointcloud
