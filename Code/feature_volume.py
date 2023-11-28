import numpy as np
import torch
from torch import nn
from einops import (rearrange, reduce, repeat)

from .utils.grid_sample import grid_sample_2d
from .utils.cnn3d import VolumeRegularization
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D


def tensor_to_point_cloud(tensor, color):
    x_coords, y_coords, z_coords = torch.nonzero(tensor, as_tuple=True)
    points = np.vstack((x_coords.cpu().numpy(), y_coords.cpu().numpy(), z_coords.cpu().numpy())).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))

    return point_cloud


class FeatureVolume(nn.Module):
    """
    Create the coarse feature volume in a MVS-like way
    """

    def __init__(self, volume_reso):
        """
        Set up the volume grid given resolution
        """
        super().__init__()

        self.volume_reso = volume_reso
        self.volume_regularization = VolumeRegularization()

        # the volume is a cube, so we only need to define the x, y, z
        x_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * 2 / (self.volume_reso - 1) - 1  # [-1, 1]
        y_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * 2 / (self.volume_reso - 1) - 1
        z_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * 2 / (self.volume_reso - 1) - 1

        # create the volume grid
        self.x, self.y, self.z = np.meshgrid(x_line, y_line, z_line, indexing='ij')
        self.xyz = np.stack([self.x, self.y, self.z])

        self.linear = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8)
        )

    def forward(self, feats, batch):
        """
        feats: [B NV C H W], NV: number of views
        batch: to get the poses for homography
        """
        source_poses = batch['source_poses']
        B, NV, _, _ = source_poses.shape

        # ---- step 1: projection -----------------------------------------------
        volume_xyz = torch.tensor(self.xyz).type_as(source_poses)
        volume_xyz = volume_xyz.reshape([3, -1])
        volume_xyz_homo = torch.cat([volume_xyz, torch.ones_like(volume_xyz[0:1])], axis=0)  # [4,XYZ]

        volume_xyz_homo_NV = repeat(volume_xyz_homo, "Num4 XYZ -> B NV Num4 XYZ", B=B, NV=NV)

        # volume project into views
        volume_xyz_pixel_homo = source_poses @ volume_xyz_homo_NV  # B NV 4 4 @ B NV 4 XYZ
        volume_xyz_pixel_homo = volume_xyz_pixel_homo[:, :, :3]
        mask_valid_depth = volume_xyz_pixel_homo[:, :, 2] > 0  # B NV XYZ
        mask_valid_depth = mask_valid_depth.float()
        mask_valid_depth = rearrange(mask_valid_depth, "B NV XYZ -> (B NV) XYZ")

        volume_xyz_pixel = volume_xyz_pixel_homo / volume_xyz_pixel_homo[:, :, 2:3]
        volume_xyz_pixel = volume_xyz_pixel[:, :, :2]
        volume_xyz_pixel = rearrange(volume_xyz_pixel, "B NV Dim2 XYZ -> (B NV) XYZ Dim2")
        volume_xyz_pixel = volume_xyz_pixel.unsqueeze(2)

        # projection: project all x * y * z points to NV images and sample features

        # grid sample 2D
        # prior depth map: (B, NV, H, W)
        depth_prior = batch['depths_prior_h']  # .unsqueeze(2) #(B, NV, H, W)
        # get valid prior depth
        # near = batch['near_fars'][:, 0, 0:1].unsqueeze(-1).unsqueeze(-1)
        # far = batch['near_fars'][:, 0, 1:2].unsqueeze(-1).unsqueeze(-1)
        mask_depth = (depth_prior > 0) #& (depth_prior >= near) & (depth_prior <= far)
        # min_val = torch.min(depth_prior)
        # max_val = torch.max(depth_prior)
        # print(max_val, min_val)
        depth_prior = depth_prior * mask_depth
        pixel_depth, pixel_depth_mask = grid_sample_2d(
            rearrange(depth_prior.unsqueeze(2), "B NV C H W -> (B NV) C H W"),
            volume_xyz_pixel.double())  # (B NV) C XYZ 1, (B NV XYZ 1)
        pixel_depth_mask_multi = pixel_depth_mask.unsqueeze(1)  # .repeat(1, NV, 1, 1)#.unsqueeze(2)
        pixel_depth = pixel_depth * pixel_depth_mask_multi

        pixel_depth = pixel_depth.squeeze(1).squeeze(-1)
        pixel_depth = rearrange(pixel_depth, "(B NV) XYZ -> B NV XYZ", B=B, NV=NV)
        # get valid projected depth
        projected_depth = volume_xyz_pixel_homo[:, :, 2]  # (B, NV, XYZ)
        # near = near.squeeze(-1)
        # far = far.squeeze(-1)
        projected_depth_mask = (projected_depth > 0) #& (projected_depth >= near) & (projected_depth <= far)
        projected_depth = projected_depth * projected_depth_mask
        # get unit depth
        unit_depth = (projected_depth[:, :, -1] - projected_depth[:, :, 0]) / (self.volume_reso - 1)
        unit_depth = unit_depth.unsqueeze(-1).repeat(1, 1, projected_depth.shape[-1])
        # print('unit_depth', unit_depth.shape)
        depth_difference = pixel_depth - projected_depth
        # min_val = torch.min(depth_difference)
        # max_val = torch.max(depth_difference)
        # print(max_val, min_val)
        delta_mask = torch.abs(depth_difference) < 50 * unit_depth
        torch.set_printoptions(threshold=float('inf'))
        # print(delta_mask)
        # mutiply teo valid masks and delta mask
        pixel_depth_mask_multi = pixel_depth_mask_multi.squeeze(1).squeeze(-1)  # (B N XYZ)
        pixel_depth_mask_multi = rearrange(pixel_depth_mask_multi, "(B NV) XYZ -> B NV XYZ", B=B, NV=NV)

        delta_mask = delta_mask * pixel_depth_mask_multi * projected_depth_mask
        # min_val = torch.min(pixel_depth)
        # max_val = torch.max(pixel_depth)
        # min_val1 = torch.min(projected_depth)
        # max_val1 = torch.max(projected_depth)
        # print(max_val, min_val, min_val1, max_val1)
        # for debugging & visulization
        # pixel_depth = rearrange(delta_mask, "B NV (X Y Z) -> B NV X Y Z", X=48, Y=48, Z=48)
        # vis_mask1 = pixel_depth[0, 0]
        # projected_depth = rearrange(projected_depth, "B NV (X Y Z) -> B NV X Y Z", X=48, Y=48, Z=48)
        # vis_mask2 = projected_depth[0, 0]
        # point_cloud1 = tensor_to_point_cloud(vis_mask1, (1, 0, 0))  # 红色
        # # point_cloud2 = tensor_to_point_cloud(vis_mask2, (0, 0, 1))  # 蓝色
        # o3d.visualization.draw_geometries([point_cloud1])
        #
        # exit()

        delta_mask = rearrange(delta_mask, "B NV XYZ -> (B NV) XYZ")
        depth_mask = delta_mask

        volume_feature, mask = grid_sample_2d(rearrange(feats, "B NV C H W -> (B NV) C H W"),
                                              volume_xyz_pixel)  # (B NV) C XYZ 1, (B NV XYZ 1)

        volume_feature = volume_feature.squeeze(-1)
        mask = mask.squeeze(-1)  # (B NV XYZ)
        mask = mask * mask_valid_depth  # * delta_mask

        volume_feature = rearrange(volume_feature, "(B NV) C (NumX NumY NumZ) -> B NV NumX NumY NumZ C", B=B, NV=NV,
                                   NumX=self.volume_reso, NumY=self.volume_reso, NumZ=self.volume_reso)
        mask = rearrange(mask, "(B NV) (NumX NumY NumZ) -> B NV NumX NumY NumZ", B=B, NV=NV, NumX=self.volume_reso,
                         NumY=self.volume_reso, NumZ=self.volume_reso)

        weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
        weight = weight.unsqueeze(-1)  # B NV X Y Z 1

        # ---- step 2: compress ------------------------------------------------
        depth_mask = rearrange(depth_mask, "(B NV) (NumX NumY NumZ) -> B NV NumX NumY NumZ", B=B, NV=NV,
                               NumX=self.volume_reso, NumY=self.volume_reso, NumZ=self.volume_reso)
        num_true = torch.sum(depth_mask).item()
        num_false = depth_mask.numel() - num_true
        depth_weight = depth_mask / (torch.sum(depth_mask, dim=1, keepdim=True) + 1e-8)
        depth_weight = depth_weight.unsqueeze(-1)
        volume_feature = volume_feature * depth_weight
        volume_feature_compressed = self.linear(volume_feature)

        # ---- step 3: mean, var ------------------------------------------------
        mean = torch.sum(volume_feature_compressed * weight * depth_weight, dim=1, keepdim=True)  # B 1 X Y Z C
        var = torch.sum(weight * (volume_feature_compressed - mean) ** 2, dim=1, keepdim=True)  # B 1 X Y Z C
        mean = mean.squeeze(1)
        var = var.squeeze(1)

        volume_mean_var = torch.cat([mean, var], axis=-1)  # [B X Y Z C]
        volume_mean_var = volume_mean_var.permute(0, 4, 3, 2, 1)  # [B,C,Z,Y,X]

        # ---- step 4: 3D regularization ----------------------------------------
        volume_mean_var_reg = self.volume_regularization(volume_mean_var)

        return volume_mean_var_reg
