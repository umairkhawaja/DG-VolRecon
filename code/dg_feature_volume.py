import numpy as np
import torch
from torch import nn
from einops import rearrange, reduce, repeat

from .utils.grid_sample import grid_sample_2d
from .utils.cnn3d import VolumeRegularization
import torch.nn.functional as F


### UMAIR --> TSDF Volume + FeatVolume
### Creating V_d --> TSDF Volume
def project(xyz, poses, K, imsize):
    """
    xyz: b x (*spatial_dims) x 3
    poses: b x nviews x 4 x 4
    K: (b x nviews x 3 x 3)
    imsize: (imheight, imwidth)
    """

    device = xyz.device
    batch_size = xyz.shape[0]
    spatial_dims = xyz.shape[1:-1]
    n_views = poses.shape[1]

    xyz = xyz.view(batch_size, 1, -1, 3).transpose(3, 2)
    xyz = torch.cat((xyz, torch.ones_like(xyz[:, :, :1])), dim=2)

    with torch.autocast(enabled=False, device_type=device.type):
        xyz_cam = (torch.inverse(poses) @ xyz)[:, :, :3]
        uv = K @ xyz_cam

    z = uv[:, :, 2]
    uv = uv[:, :, :2] / uv[:, :, 2:]
    imheight, imwidth = imsize
    """
    assuming that these uv coordinates have
        (0, 0) = center of top left pixel
        (w - 1, h - 1) = center of bottom right pixel
    then we allow values between (-.5, w-.5) because they are inside the border pixel
    """
    valid = (
        (uv[:, :, 0] >= -0.5)
        & (uv[:, :, 1] >= -0.5)
        & (uv[:, :, 0] <= imwidth - 0.5)
        & (uv[:, :, 1] <= imheight - 0.5)
        & (z > 0)
    )
    uv = uv.transpose(2, 3)
    uv = uv.view(batch_size, n_views, *spatial_dims, 2)
    z = z.view(batch_size, n_views, *spatial_dims)
    valid = valid.view(batch_size, n_views, *spatial_dims)
    return uv, z, valid


def sample_posed_images(
    imgs, poses, K, xyz, mode="bilinear", padding_mode="zeros", return_z=False
):
    """
    imgs: b x nviews x C x H x W
    poses: b x nviews x 4 x 4
    K: (b x nviews x 3 x 3)
    xyz: b x (*spatial_dims) x 3
    """

    device = imgs.device
    batch_size, n_views, _, imheight, imwidth = imgs.shape
    spatial_dims = xyz.shape[1:-1]

    """
    assuming that these uv coordinates have
        (0, 0) = center of top left pixel
        (w - 1, h - 1) = center of bottom right pixel

    adjust because grid_sample(align_corners=False) assumes
        (0, 0) = top left corner of top left pixel
        (w, h) = bottom right corner of bottom right pixel
    """
    uv, z, valid = project(xyz, poses, K, (imheight, imwidth))
    imsize = torch.tensor([imwidth, imheight], device=device)
    # grid = (uv + 0.5) / imsize * 2 - 1
    grid = uv / (0.5 * imsize) + (1 / imsize - 1)
    vals = torch.nn.functional.grid_sample(
        imgs.view(batch_size * n_views, *imgs.shape[2:]),
        grid.view(batch_size * n_views, 1, -1, 2),
        align_corners=False,
        mode=mode,
        padding_mode=padding_mode,
    )
    vals = vals.view(batch_size, n_views, -1, *spatial_dims)
    if return_z:
        return vals, valid, z
    else:
        return vals, valid


def tsdf_fusion(pred_depth_imgs, poses, K_pred_depth, input_coords):
    depth, valid, z = sample_posed_images(
        pred_depth_imgs[:, :, None],
        poses,
        K_pred_depth,
        input_coords,
        mode="nearest",
        return_z=True,
    )
    depth = depth.squeeze(2)
    valid.masked_fill_(depth == 0, False)
    margin = 3 * 1  # self.config.voxel_size
    tsdf = torch.clamp(z - depth, -margin, margin) / margin
    valid &= tsdf < 0.999
    tsdf.masked_fill_(~valid, 0)
    tsdf = torch.sum(tsdf, dim=1)
    weight = torch.sum(valid, dim=1)
    tsdf /= weight
    return tsdf, weight


class DepthGuidedFeatureVolume(nn.Module):
    """
    Create the coarse feature volume in a MVS-like way
    """

    def __init__(self, volume_reso, num_views, concat_tsdf=True):
        """
        Set up the volume grid given resolution
        """
        super().__init__()

        self.volume_reso = volume_reso
        self.num_views = num_views
        self.concat_tsdf = concat_tsdf
        self.volume_regularization = VolumeRegularization(concat_tsdf=concat_tsdf)

        # the volume is a cube, so we only need to define the x, y, z
        x_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * 2 / (
            self.volume_reso - 1
        ) - 1  # [-1, 1]
        y_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * 2 / (
            self.volume_reso - 1
        ) - 1
        z_line = (np.linspace(0, self.volume_reso - 1, self.volume_reso)) * 2 / (
            self.volume_reso - 1
        ) - 1

        # create the volume grid
        self.x, self.y, self.z = np.meshgrid(x_line, y_line, z_line, indexing="ij")
        self.xyz = np.stack([self.x, self.y, self.z])
        linear_in_features = 32  # 33 if self.concat_tsdf else 32
        self.linear = nn.Sequential(
            nn.Linear(linear_in_features, linear_in_features),
            nn.ReLU(inplace=True),
            nn.Linear(linear_in_features, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
        )

    def tsdf_to_weights(self, tsdf_volume, epsilon=1e-3, alpha=1.0):
        """
        Convert TSDF values to weights. Higher weights near the surface (TSDF ~ 0),
        and lower weights away from it.
        """
        # Using an inverse exponential function for weighting
        # Weights will be higher near surfaces and decay away from the surface
        weights = torch.exp(-alpha * torch.abs(tsdf_volume) / epsilon)
        return weights

    def forward(self, feats, batch):
        """
        feats: [B NV C H W], NV: number of views
        batch: to get the poses for homography
        """
        source_poses = batch["source_poses"]
        B, NV, _, _ = source_poses.shape
        ## NOTE: Change priors HERE
        depth_prior_key = "source_depths_h"  # Using Ground-truth
        # depth_prior_key = "source_depths_prior_h"  # Using predicted
        depth_maps = batch[depth_prior_key]

        # ---- step 1: projection -----------------------------------------------
        volume_xyz = torch.tensor(self.xyz).type_as(source_poses)
        volume_xyz = volume_xyz.reshape([3, -1])
        volume_xyz_homo = torch.cat(
            [volume_xyz, torch.ones_like(volume_xyz[0:1])], axis=0
        )  # [4,XYZ]

        volume_xyz_homo_NV = repeat(
            volume_xyz_homo, "Num4 XYZ -> B NV Num4 XYZ", B=B, NV=NV
        )

        # volume project into views
        volume_xyz_pixel_homo = (
            source_poses @ volume_xyz_homo_NV
        )  # B NV 4 4 @ B NV 4 XYZ
        volume_xyz_pixel_homo = volume_xyz_pixel_homo[:, :, :3]
        mask_valid_depth = volume_xyz_pixel_homo[:, :, 2] > 0  # B NV XYZ
        mask_valid_depth = mask_valid_depth.float()
        mask_valid_depth = rearrange(mask_valid_depth, "B NV XYZ -> (B NV) XYZ")

        volume_xyz_pixel = volume_xyz_pixel_homo / volume_xyz_pixel_homo[:, :, 2:3]
        volume_xyz_pixel = volume_xyz_pixel[:, :, :2]
        volume_xyz_pixel = rearrange(
            volume_xyz_pixel, "B NV Dim2 XYZ -> (B NV) XYZ Dim2"
        )
        volume_xyz_pixel = volume_xyz_pixel.unsqueeze(2)

        # projection: project all x * y * z points to NV images and sample features

        # grid sample 2D
        volume_feature, mask = grid_sample_2d(
            rearrange(feats, "B NV C H W -> (B NV) C H W"), volume_xyz_pixel
        )  # (B NV) C XYZ 1, (B NV XYZ 1)

        volume_feature = volume_feature.squeeze(-1)
        mask = mask.squeeze(-1)  # (B NV XYZ)
        mask = mask * mask_valid_depth

        volume_feature = rearrange(
            volume_feature,
            "(B NV) C (NumX NumY NumZ) -> B NV NumX NumY NumZ C",
            B=B,
            NV=NV,
            NumX=self.volume_reso,
            NumY=self.volume_reso,
            NumZ=self.volume_reso,
        )
        mask = rearrange(
            mask,
            "(B NV) (NumX NumY NumZ) -> B NV NumX NumY NumZ",
            B=B,
            NV=NV,
            NumX=self.volume_reso,
            NumY=self.volume_reso,
            NumZ=self.volume_reso,
        )

        weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
        weight = weight.unsqueeze(-1)  # B NV X Y Z 1

        ### V_d
        ## Batchify the volume grid
        xyz = torch.stack([torch.tensor(self.xyz)] * B, axis=0)
        xyz = xyz.reshape(B, -1, 3)

        ## ScanNet: https://github.com/ScanNet/ScanNet/issues/24#issuecomment-514077850
        # FineRecon uses ScanNet which stores poses (cam2world) not extrinsics (world2cam)
        ## FineRecon project uses these poses
        tsdf, tsdf_weight = tsdf_fusion(
            batch[depth_prior_key].type_as(batch["source_c2ws"]),  # Skip ref views
            batch["source_c2ws"].to(batch[depth_prior_key].device),
            batch["source_intrinsics"].to(batch[depth_prior_key].device),
            xyz.type_as(batch["source_intrinsics"]).to(batch[depth_prior_key].device),
        )
        tsdf.masked_fill_(tsdf_weight == 0, 1)
        tsdf_volume = tsdf.reshape(
            B, self.volume_reso, self.volume_reso, self.volume_reso
        )

        # Convert TSDF to weights
        tsdf_weights = self.tsdf_to_weights(tsdf_volume)

        # Adjust dimensions of tsdf_weights to match volume_feature
        tsdf_weights_expanded = tsdf_weights.unsqueeze(1)  # Adding a dimension for NV
        tsdf_weights_expanded = tsdf_weights_expanded.expand(
            -1, NV, -1, -1, -1
        )  # Expanding to [B, NV, D, H, W]
        tsdf_weights_expanded = tsdf_weights_expanded.unsqueeze(
            -1
        )  # Adding a dimension for feature channels

        # Apply weights to the volume feature
        volume_feature_weighted = volume_feature * tsdf_weights_expanded

        ## TSDF CONCAT: Concatenate with volume feature BEFORE passing to compression ##
        # tsdf_volume_NV = torch.stack([tsdf_volume[:, :, :, :, None]] * NV, dim=1)
        # print(tsdf_volume_NV.shape)
        # print(volume_feature.shape)
        # v_D = torch.concat([volume_feature, tsdf_volume_NV], dim=-1)
        # print(v_D.shape)
        ######### END TSDF CONCAT ##########

        # ---- step 2: compress ------------------------------------------------
        volume_feature_compressed = self.linear(volume_feature_weighted)
        # volume_feature_compressed = self.linear(v_D)
        # print(volume_feature_compressed.shape)
        # ---- step 3: mean, var ------------------------------------------------
        mean = torch.sum(
            volume_feature_compressed * weight, dim=1, keepdim=True
        )  # B 1 X Y Z C
        var = torch.sum(
            weight * (volume_feature_compressed - mean) ** 2, dim=1, keepdim=True
        )  # B 1 X Y Z C
        mean = mean.squeeze(1)
        var = var.squeeze(1)

        volume_mean_var = torch.cat([mean, var], axis=-1)  # [B X Y Z C]
        volume_mean_var = volume_mean_var.permute(0, 4, 3, 2, 1)  # [B,C,Z,Y,X]

        # tsdf_volume_mean_var = torch.concat(
        #     [volume_mean_var, tsdf_volume[:, None]], dim=1
        # )
        # ---- step 4: 3D regularization ----------------------------------------
        volume_mean_var_reg = self.volume_regularization(volume_mean_var)
        # volume_mean_var_reg = self.volume_regularization(tsdf_volume_mean_var)

        return volume_mean_var_reg