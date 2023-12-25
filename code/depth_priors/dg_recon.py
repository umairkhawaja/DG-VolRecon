import torch

"""
    Depth-Guided Backprojection from DG-Recon

    1. Predict the depth map for a keyframe.
    2. Back-project the 2D features into 3D space using the predicted depth map.
    3. Apply a thresholding operation to only keep the features within a fixed distance Δ from the estimated depth surface.
    
"""

def unproject(depth_map, intrinsic_matrix_inv, pose_matrix):
    # Create a grid of coordinates corresponding to the depth map pixels
    H, W = depth_map.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    # Normalize pixel coordinates (u, v) to camera coordinates (x, y)
    grid_x_normalized = (grid_x.float() - intrinsic_matrix_inv[0, 2]) * depth_map / intrinsic_matrix_inv[0, 0]
    grid_y_normalized = (grid_y.float() - intrinsic_matrix_inv[1, 2]) * depth_map / intrinsic_matrix_inv[1, 1]

    # Unproject from camera to world space
    points_3d = torch.stack((grid_x_normalized, grid_y_normalized, depth_map, torch.ones_like(depth_map)), dim=-1)
    points_3d = torch.matmul(pose_matrix, points_3d.reshape(-1, 4).T).T.reshape(H, W, 4)

    return points_3d[:, :, :3]  # Discard the homogeneous coordinate

def back_project_features(depth_map, features_2d, intrinsic_matrix, pose_matrix, delta):
    # Compute the inverse of the intrinsic matrix
    intrinsic_matrix_inv = torch.inverse(intrinsic_matrix)

    # Unproject the 2D depth map to 3D points
    points_3d = unproject(depth_map, intrinsic_matrix_inv, pose_matrix)

    # Calculate the voxel indices for the 3D points
    voxel_indices = (points_3d / delta).long()

    # Initialize 3D feature volume (size dependent on the scene and delta)
    feature_volume_size = torch.max(voxel_indices, dim=[0, 1])[0] + 1  # This assumes all indices are positive
    feature_volume = torch.zeros(*feature_volume_size, features_2d.shape[-1], device=depth_map.device)

    # Iterate over each pixel and assign the 2D features to the 3D feature volume
    for y in range(H):
        for x in range(W):
            voxel = voxel_indices[y, x]
            feature_volume[voxel[0], voxel[1], voxel[2]] = features_2d[y, x]

    return feature_volume

#### Usage
# Example inputs
# H, W, C = 240, 320, 64  # Example dimensions for image and feature map
# depth_map = torch.rand(H, W) * 10  # Dummy depth map with values in some unit, e.g., meters
# features_2d = torch.rand(H, W, C)  # Dummy 2D features
# intrinsic_matrix = torch.tensor([[500, 0, W/2], [0, 500, H/2], [0, 0, 1]])  # Example intrinsic matrix
# pose_matrix = torch.eye(4)  # Example pose (identity matrix implies no transformation)
# delta = 0.05  # Example threshold in the same unit as depth_map

# # Convert the intrinsic matrix to a tensor and invert it
# intrinsic_matrix_inv = torch.inverse(intrinsic_matrix)

# # Calculate the back-projected feature volume
# back_projected_features = back_project_features(
#     depth_map, features_2d, intrinsic_matrix_inv, pose_matrix, delta
# )


"""
The provided excerpt describes a process for occupancy mapping using a depth prior. The occupancy probability is updated recursively using a log-odds representation, which is common in robotics for representing map certainty. The probability of a voxel being occupied, given depth observations from ( k ) different views, is computed using a Gaussian distribution centered at the depth estimate with a standard deviation equal to ( \Delta ), the back projection margin.

To implement this in PyTorch, you'd need to:

Initialize a 3D grid to store the log-odds values.
Update the grid based on new depth observations.
Convert the depth measurements into a Gaussian probability distribution.
Update the log-odds values using the Gaussian probability and the previous log-odds value.
"""

