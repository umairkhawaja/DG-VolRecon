import torch
from torch import nn
from einops import rearrange


class DepthGuidedFeatureVolume(nn.Module):
    def __init__(self, volume_reso, delta):
        """
        Initialize the Depth Guided Feature Volume module.

        Parameters:
        - volume_reso: Resolution of the feature volume (number of voxels along each axis).
        - delta: Threshold distance for feature inclusion in the volume.
        """
        super().__init__()
        self.volume_reso = volume_reso
        self.delta = delta

        # A linear layer to compress and process the features
        self.linear = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
        )

    def unproject(self, depth_map, intrinsic_matrix_inv, pose_matrix):
        """
        Unproject 2D depth map to 3D points in world coordinates.

        Parameters:
        - depth_map: The 2D depth map.
        - intrinsic_matrix_inv: The inverse of the camera's intrinsic matrix.
        - pose_matrix: The camera pose matrix.
        """
        H, W = depth_map.shape[-2], depth_map.shape[-1]
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

        # Normalize pixel coordinates to camera coordinates
        grid_x_normalized = (
            (grid_x.float() - intrinsic_matrix_inv[0, 2])
            * depth_map
            / intrinsic_matrix_inv[0, 0]
        )
        grid_y_normalized = (
            (grid_y.float() - intrinsic_matrix_inv[1, 2])
            * depth_map
            / intrinsic_matrix_inv[1, 1]
        )

        # Stack and transform to world coordinates
        points_3d = torch.stack(
            (
                grid_x_normalized,
                grid_y_normalized,
                depth_map,
                torch.ones_like(depth_map),
            ),
            dim=-1,
        )
        points_3d = torch.matmul(pose_matrix, points_3d.reshape(-1, 4).T).T.reshape(
            H, W, 4
        )

        return points_3d[:, :, :3]  # Return only the XYZ coordinates

    def forward(self, feats, depth_maps, source_poses, intrinsic_matrices):
        """
        Forward pass to create a depth-guided feature volume.

        Parameters:
        - feats: 2D feature maps from multiple views (B, NV, C, H, W).
        - depth_maps: Predicted depth maps for each view (B, NV, H, W).
        - source_poses: Camera poses for each view (B, NV, 4, 4).
        - intrinsic_matrices: Camera intrinsic matrices for each view (B, NV, 3, 3).
        """
        B, NV, C, H, W = feats.shape
        # Initialize an empty feature volume
        volume_features = torch.zeros(
            B,
            self.volume_reso,
            self.volume_reso,
            self.volume_reso,
            C,
            device=feats.device,
        )

        for b in range(B):
            for nv in range(NV):
                depth_map = depth_maps[b, nv]
                intrinsic_matrix_inv = torch.inverse(intrinsic_matrices[b, nv])
                pose_matrix = source_poses[b, nv]

                # Back-project the 2D features into 3D space
                points_3d = self.unproject(depth_map, intrinsic_matrix_inv, pose_matrix)

                # Check if the 3D points are within a certain distance from the depth surface
                for y in range(H):
                    for x in range(W):
                        point_3d = points_3d[y, x]
                        depth_at_pixel = depth_map[y, x]
                        distance_from_surface = torch.abs(
                            point_3d[2] - depth_at_pixel
                        )  # Z coordinate represents depth

                        # Only process points within the delta threshold
                        if distance_from_surface <= self.delta:
                            # Calculate voxel index and update feature volume
                            voxel = (point_3d / self.volume_reso).long()
                            if (
                                0 <= voxel[0] < self.volume_reso
                                and 0 <= voxel[1] < self.volume_reso
                                and 0 <= voxel[2] < self.volume_reso
                            ):
                                volume_features[
                                    b, voxel[0], voxel[1], voxel[2]
                                ] += feats[b, nv, :, y, x]

        # Normalize and compress features using the linear layer
        volume_features = self.linear(volume_features)
        return volume_features


import torch
from torch import nn
from einops import rearrange


class DGReconFeatureVolume(nn.Module):
    def __init__(self, volume_reso=48, delta=5):
        """
        Initialize the Depth Guided Feature Volume module.

        Parameters:
        - volume_reso: Resolution of the feature volume (number of voxels along each axis).
        - delta: Threshold distance for feature inclusion in the volume.
        """
        super().__init__()
        self.volume_reso = volume_reso
        self.delta = delta

        # A linear layer to compress and process the features
        self.linear = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
        )

    def unproject(self, depth_map, pose_matrix):
        """
        Unproject 2D depth map to 3D points in world coordinates.

        Parameters:
        - depth_map: The 2D depth map.
        - intrinsic_matrix_inv: The inverse of the camera's intrinsic matrix.
        - pose_matrix: The camera pose matrix.
        """
        H, W = depth_map.shape[-2], depth_map.shape[-1]
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

        # Normalize pixel coordinates to camera coordinates
        grid_x_normalized = (
            (grid_x.float() - intrinsic_matrix_inv[0, 2])
            * depth_map
            / intrinsic_matrix_inv[0, 0]
        )
        grid_y_normalized = (
            (grid_y.float() - intrinsic_matrix_inv[1, 2])
            * depth_map
            / intrinsic_matrix_inv[1, 1]
        )

        # Stack and transform to world coordinates
        points_3d = torch.stack(
            (
                grid_x_normalized,
                grid_y_normalized,
                depth_map,
                torch.ones_like(depth_map),
            ),
            dim=-1,
        )
        points_3d = torch.matmul(pose_matrix, points_3d.reshape(-1, 4).T).T.reshape(
            H, W, 4
        )

        return points_3d[:, :, :3]  # Return only the XYZ coordinates

    def forward(self, feats, depth_maps, source_poses, intrinsic_matrices):
        """
        Forward pass to create a depth-guided feature volume.

        Parameters:
        - feats: 2D feature maps from multiple views (B, NV, C, H, W).
        - depth_maps: Predicted depth maps for each view (B, NV, H, W).
        - source_poses: Camera poses for each view (B, NV, 4, 4).
        - intrinsic_matrices: Camera intrinsic matrices for each view (B, NV, 3, 3).
        """
        B, NV, C, H, W = feats.shape
        # Initialize an empty feature volume
        volume_features = torch.zeros(
            B,
            self.volume_reso,
            self.volume_reso,
            self.volume_reso,
            C,
            device=feats.device,
        )

        for b in range(B):
            for nv in range(NV):
                depth_map = depth_maps[b, nv]
                intrinsic_matrix_inv = torch.inverse(intrinsic_matrices[b, nv])
                pose_matrix = source_poses[b, nv]

                # Back-project the 2D features into 3D space
                points_3d = self.unproject(depth_map, intrinsic_matrix_inv, pose_matrix)

                # Check if the 3D points are within a certain distance from the depth surface
                for y in range(H):
                    for x in range(W):
                        point_3d = points_3d[y, x]
                        depth_at_pixel = depth_map[y, x]
                        distance_from_surface = torch.abs(
                            point_3d[2] - depth_at_pixel
                        )  # Z coordinate represents depth

                        # Only process points within the delta threshold
                        if distance_from_surface <= self.delta:
                            # Calculate voxel index and update feature volume
                            voxel = (point_3d / self.volume_reso).long()
                            if (
                                0 <= voxel[0] < self.volume_reso
                                and 0 <= voxel[1] < self.volume_reso
                                and 0 <= voxel[2] < self.volume_reso
                            ):
                                volume_features[
                                    b, voxel[0], voxel[1], voxel[2]
                                ] += feats[b, nv, :, y, x]

        # Normalize and compress features using the linear layer
        volume_features = self.linear(volume_features)
        return volume_features
