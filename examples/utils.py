import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image
import os
from typing import Tuple, Optional


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


def save_image_numpy(image_array: np.ndarray, filename: str):
    """
    Saves a [1, 3, H, W] NumPy array as an image file.

    Args:
        image_array (np.ndarray): The input array with shape [1, 3, H, W].
                                  Values are expected to be in the [0, 1] range.
        filename (str): The path to save the image to (e.g., 'output.png').
    """
    # 1. Remove the batch dimension -> shape becomes [3, H, W]
    image = image_array.squeeze(0)
    
    # 2. Transpose dimensions to [H, W, 3] for PIL
    image = np.transpose(image, (1, 2, 0))
    
    # 3. Scale values from [0, 1] to [0, 255] and convert to 8-bit integers
    image = (image * 255).astype(np.uint8)
    
    # 4. Create a PIL Image from the NumPy array
    pil_image = Image.fromarray(image)
    
    # 5. Save the image
    pil_image.save(filename)
    print(f"✅ Image saved successfully as '{filename}'")

def save_image_tensor(image_tensor: torch.Tensor, filename: str):
    """
    Saves a [1, 3, H, W] tensor as an image file.

    Args:
        image_tensor (torch.Tensor): The input tensor with shape [1, 3, H, W].
                                     Values are expected to be in the [0, 1] range.
        filename (str): The path to save the image to (e.g., 'output.png').
    """
    # Ensure tensor is on the CPU
    image_tensor = image_tensor.cpu()
    
    # 1. Remove the batch dimension -> shape becomes [3, H, W]
    image = image_tensor.squeeze(0)
    
    # 2. Permute the dimensions to [H, W, 3] for PIL
    image = image.permute(1, 2, 0)
    
    # 3. Scale values from [0, 1] to [0, 255] and convert to 8-bit integers
    image = (image * 255).byte()
    
    # 4. Convert to a NumPy array and create a PIL Image
    pil_image = Image.fromarray(image.numpy())
    
    # 5. Save the image
    pil_image.save(filename)
    print(f"✅ Image saved successfully as '{filename}'")

def sample_normals_from_map(means2d, normal_map, image_height, image_width):
    """
    Sample normal vectors from a normal map using 2D projected coordinates.
    
    Args:
        means2d: [1, N, 2] - 2D projected coordinates of Gaussians
        normal_map: [1, H, W, 3] - Normal map in HWC format
        image_height, image_width: Image dimensions
    
    Returns:
        sampled_normals: [1, N, 3] - Normal vectors for each Gaussian
    """
    
    # Move normal_map to the same device as means2d
    normal_map = normal_map.to(means2d.device)
    
    # Handle the format: [1, H, W, 3] -> [1, 3, H, W] for grid_sample
    if normal_map.shape[-1] == 3:  # If it's [1, H, W, 3]
        normal_map = normal_map.permute(0, 3, 1, 2)  # Convert to [1, 3, H, W]
    
    # Normalize coordinates to [-1, 1] for grid_sample
    normalized_coords = means2d.clone()
    normalized_coords[..., 0] = (normalized_coords[..., 0] / (image_width - 1)) * 2.0 - 1.0
    normalized_coords[..., 1] = (normalized_coords[..., 1] / (image_height - 1)) * 2.0 - 1.0
    
    # Add dummy dimension for grid_sample: [1, N, 2] -> [1, N, 1, 2]
    grid_coords = normalized_coords.unsqueeze(2)  # [1, N, 1, 2]
    
    # Sample normals using grid_sample
    sampled_normals = F.grid_sample(
        normal_map,      # [1, 3, H, W]
        grid_coords,     # [1, N, 1, 2]
        mode='bilinear', 
        padding_mode='border',
        align_corners=True
    )  # Output: [1, 3, N, 1]
    
    # Reshape to [1, N, 3]
    sampled_normals = sampled_normals.squeeze(3).permute(0, 2, 1)  # [1, 3, N, 1] -> [1, 3, N] -> [1, N, 3]
    
    # Convert from [0,1] to [-1,1] if needed and normalize
    # Check if normals are in [0,1] range (common for normal map textures)
    if sampled_normals.min() >= 0 and sampled_normals.max() <= 1:
        sampled_normals = sampled_normals * 2.0 - 1.0
    
    # Normalize to unit vectors
    sampled_normals = F.normalize(sampled_normals, dim=-1)
    
    return sampled_normals

def create_normal_map_from_gaussians(means2d, sampled_normals, image_height, image_width, 
                                   method='nearest', background_normal=None):
    """
    Create a normal map by projecting Gaussian normals back to image space.
    
    Args:
        means2d: [1, N, 2] - 2D projected coordinates of Gaussians
        sampled_normals: [1, N, 3] - Normal vectors for each Gaussian
        image_height, image_width: Target image dimensions
        method: 'nearest', 'weighted', or 'splatting' - how to handle overlapping Gaussians
        background_normal: [3] - Default normal for empty pixels (default: [0, 0, 1])
    
    Returns:
        normal_map: [3, H, W] - Reconstructed normal map
        coverage_map: [H, W] - Shows which pixels have Gaussian contributions
    """
    
    device = means2d.device
    batch_size, num_gaussians, _ = means2d.shape
    
    if background_normal is None:
        background_normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    # Initialize output maps
    normal_map = background_normal.view(3, 1, 1).expand(3, image_height, image_width).clone()
    coverage_map = torch.zeros(image_height, image_width, device=device)
    
    # Get valid Gaussians (within image bounds)
    valid_mask = (
        (means2d[0, :, 0] >= 0) & (means2d[0, :, 0] < image_width) &
        (means2d[0, :, 1] >= 0) & (means2d[0, :, 1] < image_height)
    )
    
    if not valid_mask.any():
        return normal_map, coverage_map
    
    valid_coords = means2d[0, valid_mask]  # [N_valid, 2]
    valid_normals = sampled_normals[0, valid_mask]  # [N_valid, 3]
    norms = _create_normal_map_nearest(valid_coords, valid_normals, normal_map, coverage_map)
    save_normal_map_as_image(normal_map, "debug_normal_map.png")
    # raise ValueError(f"Unknown method: {method}")


def _create_normal_map_nearest(coords, normals, normal_map, coverage_map):
    """Nearest neighbor assignment - each Gaussian affects its closest pixel."""
    
    # Round coordinates to nearest pixel
    pixel_coords = torch.round(coords).long()  # [N_valid, 2]
    
    # Clamp to image bounds (safety check)
    pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, normal_map.shape[2] - 1)
    pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, normal_map.shape[1] - 1)
    
    # Assign normals to pixels
    for i in range(len(pixel_coords)):
        x, y = pixel_coords[i]
        normal_map[:, y, x] = normals[i]
        coverage_map[y, x] = 1.0
    
    return normal_map, coverage_map


# Helper function to save normal maps for visual inspection
def save_normal_map_as_image(normal_map, filename, convert_to_color=True):
    """Save normal map as an image file for visual inspection."""
    import numpy as np
    from PIL import Image
    
    if convert_to_color:
        # Convert from [-1,1] to [0,255] for visualization
        normal_vis = (normal_map + 1.0) * 0.5
        normal_vis = torch.clamp(normal_vis, 0, 1)
        normal_vis = (normal_vis * 255).byte().permute(1, 2, 0).cpu().numpy()
    else:
        normal_vis = (normal_map * 255).byte().permute(1, 2, 0).cpu().numpy()
    
    Image.fromarray(normal_vis).save(filename)


def calculate_gaussian_splat_normal(rotation_matrix, scale_matrix, temperature=0.01):
    """
    Calculate the normal vector of a 3D Gaussian splat given rotation and scale matrices.
    
    Args:
        rotation_matrix: Shape [N, 4] - quaternion representation (w, x, y, z)
                                    or [N, 3, 3] - rotation matrices
        scale_matrix: Shape [N, 3] - scale factors for each axis
    
    Returns:
        np.ndarray: Shape [1, N, 3] - normal vectors for each Gaussian splat
    """
    
    # Convert quaternions to rotation matrices if input is quaternions
    if rotation_matrix.shape[-1] == 4:
        rotation_matrix = quaternion_to_rotation_matrix(rotation_matrix)
    
    N = rotation_matrix.shape[0]
    device = rotation_matrix.device
    
    # # Find indices of minimum scale for each Gaussian (vectorized)
    # min_scale_indices = torch.argmin(scale_matrix, dim=1)
    
    # # Extract the corresponding columns from rotation matrices
    # row_indices = torch.arange(N, device=device)
    # normals = rotation_matrix[row_indices, :, min_scale_indices]
    
    # # Normalize the normal vectors
    # normals = torch.nn.functional.normalize(normals, p=2, dim=1)
    
    # return normals.unsqueeze(0)

    weights = F.softmax(-scale_matrix / temperature, dim=1)

    axes = rotation_matrix.transpose(1, 2)

    normals = torch.sum(weights.unsqueeze(-1) * axes, dim=1)

    normals = F.normalize(normals, p=2, dim=1)
    
    return normals.unsqueeze(0) 

def quaternion_to_rotation_matrix(quaternions):
    """
    Convert quaternions to rotation matrices using PyTorch.
    
    Args:
        quaternions (torch.Tensor): Shape [N, 4] - quaternions (w, x, y, z)
    
    Returns:
        torch.Tensor: Shape [N, 3, 3] - rotation matrices
    """
    # Normalize quaternions
    quaternions = torch.nn.functional.normalize(quaternions, p=2, dim=1)
    
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Compute rotation matrix elements
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # Build rotation matrices
    rotation_matrices = torch.stack([
        torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=1),
        torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=1),
        torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=1)
    ], dim=1)
    
    return rotation_matrices


# ==== claude suggestions =====

def compute_normal_loss(gaussian_normals, sampled_normals, loss_type='combined'):
    """
    Enhanced normal loss with multiple components.
    """
    gaussian_normals = F.normalize(gaussian_normals, dim=-1)
    sampled_normals = F.normalize(sampled_normals, dim=-1)
    
    if loss_type == 'cosine':
        return (1.0 - F.cosine_similarity(gaussian_normals, sampled_normals, dim=-1)).mean()
    
    elif loss_type == 'angular':
        # Angular loss - more sensitive to small deviations
        cos_sim = F.cosine_similarity(gaussian_normals, sampled_normals, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)  # Numerical stability
        angular_loss = torch.acos(torch.abs(cos_sim)).mean()
        return angular_loss
    
    elif loss_type == 'combined':
        # Combine cosine similarity and L2 loss
        cosine_loss = (1.0 - F.cosine_similarity(gaussian_normals, sampled_normals, dim=-1)).mean()
        l2_loss = F.mse_loss(gaussian_normals, sampled_normals)
        return 0.7 * cosine_loss + 0.3 * l2_loss
    
    elif loss_type == 'robust':
        # Robust loss that handles outliers better
        diff = gaussian_normals - sampled_normals
        loss = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8).mean()
        return loss

def compute_progressive_normal_loss(gaussian_normals, sampled_normals, means2d, 
                                  iteration, max_iterations, confidence=None):
    """
    Progressive normal loss that starts with coarse guidance and adds fine details.
    """
    progress = iteration / max_iterations

    loss = compute_normal_loss(gaussian_normals, sampled_normals, 'cosine')
    weight = 2.0

    return loss * weight
    
    # Stage 1 (0-30%): Focus on major surface orientations
    if progress < 0.3:
        # Use heavily smoothed normals for global structure
        # smoothed_normals = smooth_normals(sampled_normals, kernel_size=15)
        loss = compute_normal_loss(gaussian_normals, sampled_normals, 'cosine')
        weight = 2.0
        
    # Stage 2 (30-70%): Gradually introduce finer details  
    elif progress < 0.7:
        # Blend between smoothed and original normals
        blend_factor = (progress - 0.3) / 0.4
        smoothed_normals = smooth_normals(sampled_normals, kernel_size=7)
        blended_normals = (1 - blend_factor) * smoothed_normals + blend_factor * sampled_normals
        loss = compute_normal_loss(gaussian_normals, blended_normals, 'combined')
        weight = 1.5
        
    # Stage 3 (70-100%): Full detail with confidence weighting
    else:
        loss = compute_normal_loss(gaussian_normals, sampled_normals, 'robust')
        if confidence is not None:
            # Weight loss by confidence
            loss = (loss * confidence).mean() / (confidence.mean() + 1e-8)
        weight = 1.0
    
    return loss * weight

def smooth_normals(normals, kernel_size=5):
    """Apply Gaussian smoothing to normal map."""
    sigma = kernel_size / 6.0
    kernel_tensor = torch.exp(-torch.linspace(-3, 3, kernel_size, device=normals.device)**2 / (2 * sigma**2))
    kernel_tensor = kernel_tensor / kernel_tensor.sum()
    kernel = kernel_tensor.view(1, 1, -1)
    
    padding = kernel_size // 2
    
    # --- FIX: Process each dimension and store in a list ---
    processed_slices = []
    
    for dim in range(3):
        # Isolate the slice from the *original*, unmodified normals tensor
        normal_slice = normals[..., dim]
        
        # 1. Horizontal pass
        horizontal_pass = F.conv1d(
            normal_slice.unsqueeze(1),
            kernel,
            padding=padding
        ).squeeze(1)
        
        # 2. Vertical pass
        vertical_pass = F.conv1d(
            horizontal_pass.transpose(-2, -1).unsqueeze(1),
            kernel,
            padding=padding
        ).squeeze(1).transpose(-2, -1)
        
        # 3. Add the processed slice to our list
        processed_slices.append(vertical_pass.unsqueeze(-1))
        
    # 4. Concatenate the results into a new tensor after the loop
    normals_smooth = torch.cat(processed_slices, dim=-1)

    return F.normalize(normals_smooth, dim=-1)


def cluster_gaussians_by_surface(means3d, normals, cluster_threshold=0.1):
    """
    Group nearby Gaussians that should represent the same surface.
    """
    from cuml.cluster import DBSCAN
    
    features = torch.cat([means3d, normals], dim=-1)  # [N, 6]
    
    # Run DBSCAN directly on the GPU tensor (no .cpu().numpy() needed)
    # Note: cuML expects float32 data
    clustering = DBSCAN(eps=cluster_threshold, min_samples=5).fit(features.detach().float())
    
    # The labels are already a GPU tensor (CuPy array), convert to PyTorch
    clusters = torch.as_tensor(clustering.labels_, device=means3d.device)
    
    return clusters

def apply_surface_consistency_loss(gaussian_normals, means3d, weight=0.5):
    """
    Encourage nearby Gaussians to have consistent normals (vectorized).
    """
    # FIX: Ensure normals are 2D
    if gaussian_normals.dim() == 3:
        gaussian_normals = gaussian_normals.squeeze(0)

    # 1. Get clusters using the fast, GPU-native function
    clusters = cluster_gaussians_by_surface(means3d, gaussian_normals)
    
    # 2. Filter out noise points (cluster_id == -1) upfront
    valid_mask = (clusters != -1)
    valid_clusters = clusters[valid_mask]
    valid_normals = gaussian_normals[valid_mask]

    if valid_clusters.numel() == 0:
        return 0.0 # No valid clusters to compute loss on

    # 3. Get unique cluster IDs and their counts efficiently
    unique_cluster_ids, inverse_indices, counts = torch.unique(
        valid_clusters, return_inverse=True, return_counts=True
    )
    
    # 4. Compute the mean normal for every cluster at once
    # Create a tensor to hold the sum of normals for each unique cluster
    num_unique_clusters = unique_cluster_ids.shape[0]
    sum_normals = torch.zeros(
        (num_unique_clusters, 3), dtype=valid_normals.dtype, device=valid_normals.device
    )
    
    # Add normals to their corresponding cluster's slot using the inverse indices
    sum_normals.index_add_(0, inverse_indices, valid_normals)
    
    # Divide sum by count to get the mean
    mean_cluster_normals = F.normalize(sum_normals / counts.unsqueeze(-1), dim=-1)
    
    # 5. "Gather" the mean normal for each point using the inverse indices
    # This creates a tensor where each row is the mean normal of the cluster that point belongs to
    mean_normals_per_point = mean_cluster_normals[inverse_indices]
    
    # 6. Compute the loss for all points in one parallel operation
    # Compare each point's normal to the mean normal of its cluster
    consistency_loss = (1.0 - F.cosine_similarity(
        valid_normals, mean_normals_per_point, dim=-1
    )).mean()
        
    return consistency_loss * weight

def calculate_gaussian_splat_normal_adaptive(rotation_matrix, scale_matrix, 
                                           opacity=None, temperature=0.1):
    """
    Enhanced normal calculation with confidence-based weighting.
    """
    if rotation_matrix.shape[-1] == 4:
        rotation_matrix = quaternion_to_rotation_matrix(rotation_matrix)
    
    N = rotation_matrix.shape[0]
    device = rotation_matrix.device
    
    # Adaptive temperature based on scale anisotropy
    scale_ratios = scale_matrix.max(dim=1)[0] / (scale_matrix.min(dim=1)[0] + 1e-8)
    
    # Higher temperature for more anisotropic Gaussians (more confident about normal)
    adaptive_temp = temperature / torch.clamp(scale_ratios, 1.0, 10.0).unsqueeze(1)
    
    # Soft weighting based on inverse scales
    weights = F.softmax(-scale_matrix / adaptive_temp, dim=1)
    
    # Calculate confidence based on how anisotropic the Gaussian is
    confidence = torch.clamp(scale_ratios - 1.0, 0.0, 5.0) / 5.0
    
    # Weight by opacity if available
    if opacity is not None:
        confidence *= torch.sigmoid(opacity).squeeze()
    
    axes = rotation_matrix.transpose(1, 2)
    normals = torch.sum(weights.unsqueeze(-1) * axes, dim=1)
    normals = F.normalize(normals, p=2, dim=1)
    
    return normals.unsqueeze(0), confidence.unsqueeze(0)

def calculate_gaussian_splat_normal_differentiable(rotation_matrix, scale_matrix, 
                                                 opacity=None, temperature=0.01, 
                                                 sharpness=10.0):
    """
    Fully differentiable normal calculation that selects the axis with smallest scale.
    
    Args:
        rotation_matrix: [N, 3, 3] or [N, 4] quaternions
        scale_matrix: [N, 3] scale values for each axis
        opacity: [N, 1] optional opacity values  
        temperature: controls softmax sharpness for scale weighting
        sharpness: multiplier for making selection sharper
    
    Returns:
        normals: [1, N, 3] normal vectors
        confidence: [1, N] confidence scores
    """
    if rotation_matrix.shape[-1] == 4:
        rotation_matrix = quaternion_to_rotation_matrix(rotation_matrix)
    
    N = rotation_matrix.shape[0]
    device = rotation_matrix.device
    
    # Method 1: Sharp softmax on inverse scales
    # Higher values for smaller scales, with very sharp selection
    inv_scale_scores = sharpness / (scale_matrix + 1e-8)
    weights = F.softmax(inv_scale_scores / temperature, dim=1)
    
    # Extract rotation axes (each column is an axis)
    axes = rotation_matrix  # [N, 3, 3] where axes[i, :, j] is j-th axis
    
    # Weighted combination heavily favoring smallest scale axis
    normals = torch.sum(weights.unsqueeze(1) * axes, dim=2)  # [N, 3]
    normals = F.normalize(normals, p=2, dim=1)
    
    # Calculate confidence based on scale anisotropy
    max_scale = torch.max(scale_matrix, dim=1)[0]
    min_scale = torch.min(scale_matrix, dim=1)[0] 
    anisotropy_ratio = max_scale / (min_scale + 1e-8)
    confidence = torch.clamp((anisotropy_ratio - 1.0) / 10.0, 0.0, 1.0)
    
    # Weight by opacity if available
    if opacity is not None:
        opacity_weight = torch.sigmoid(opacity).squeeze(-1) if opacity.dim() > 1 else torch.sigmoid(opacity)
        confidence *= opacity_weight
    
    return normals.unsqueeze(0), confidence.unsqueeze(0)


# =======


def generate_image_from_normals(sampled_normals, means2d, image_height, image_width, 
                               rendering_mode='shaded', light_dir=None, colors=None,
                               gaussian_size=3.0, opacity=1.0):
    """
    Generate an image by rendering sampled normals at their 2D projected positions.
    
    Args:
        sampled_normals: [1, N, 3] - Normal vectors for each Gaussian
        means2d: [1, N, 2] - 2D projected coordinates of Gaussians
        image_height, image_width: Image dimensions
        rendering_mode: str - 'shaded', 'normals', 'depth', or 'colored'
        light_dir: [3] - Light direction for shading (default: [0, 0, 1])
        colors: [1, N, 3] - Optional colors for each Gaussian (for 'colored' mode)
        gaussian_size: float - Size of each Gaussian splat
        opacity: float - Opacity of each Gaussian
    
    Returns:
        rendered_image: [1, 3, H, W] - Generated image
    """
    
    device = sampled_normals.device
    batch_size, num_points, _ = sampled_normals.shape
    
    # Initialize output image
    rendered_image = torch.zeros(batch_size, 3, image_height, image_width, 
                                device=device, dtype=sampled_normals.dtype)
    weight_sum = torch.zeros(batch_size, 1, image_height, image_width, 
                           device=device, dtype=sampled_normals.dtype)
    
    # Create coordinate grids
    y_coords = torch.arange(image_height, device=device, dtype=sampled_normals.dtype)
    x_coords = torch.arange(image_width, device=device, dtype=sampled_normals.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    
    # Process each Gaussian
    for i in range(num_points):
        # Get position and normal for this Gaussian
        pos_2d = means2d[0, i]  # [2]
        normal = sampled_normals[0, i]  # [3]
        
        # Skip if position is outside image bounds
        if (pos_2d[0] < 0 or pos_2d[0] >= image_width or 
            pos_2d[1] < 0 or pos_2d[1] >= image_height):
            continue
        
        # Calculate distance from Gaussian center
        dist_sq = ((grid_coords - pos_2d.unsqueeze(0).unsqueeze(0))**2).sum(dim=-1)
        
        # Gaussian weight
        sigma_sq = gaussian_size ** 2
        weights = torch.exp(-dist_sq / (2 * sigma_sq)) * opacity
        
        # Generate color based on rendering mode
        if rendering_mode == 'normals':
            # Visualize normals as RGB (convert from [-1,1] to [0,1])
            color = (normal + 1.0) / 2.0
            
        elif rendering_mode == 'shaded':
            # Simple Lambertian shading
            if light_dir is None:
                light_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
            else:
                light_dir = torch.tensor(light_dir, device=device)
            light_dir = F.normalize(light_dir, dim=0)
            
            # Calculate lighting
            lambertian = torch.clamp(torch.dot(normal, light_dir), 0.0, 1.0)
            color = torch.ones(3, device=device) * lambertian
            
        elif rendering_mode == 'depth':
            # Use Z component as depth (assuming normals point towards camera)
            depth_val = (normal[2] + 1.0) / 2.0  # Convert to [0,1]
            color = torch.ones(3, device=device) * depth_val
            
        elif rendering_mode == 'colored':
            # Use provided colors with normal-based shading
            if colors is not None:
                base_color = colors[0, i]  # [3]
            else:
                base_color = torch.ones(3, device=device)
            
            if light_dir is None:
                light_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
            else:
                light_dir = torch.tensor(light_dir, device=device)
            light_dir = F.normalize(light_dir, dim=0)
            
            lambertian = torch.clamp(torch.dot(normal, light_dir), 0.1, 1.0)
            color = base_color * lambertian
            
        else:
            raise ValueError(f"Unknown rendering mode: {rendering_mode}")
        
        # Apply Gaussian splatting
        for c in range(3):
            rendered_image[0, c] += weights * color[c]
        weight_sum[0, 0] += weights
    
    # Normalize by weights to avoid over-brightening
    weight_sum = torch.clamp(weight_sum, min=1e-6)
    rendered_image = rendered_image / weight_sum
    
    return rendered_image


def visualize_normal_sampling(means2d, normal_map, sampled_normals, image_height, image_width,
                             save_path=None, show_comparison=True):
    """
    Create a visualization comparing the original normal map with the sampled normals.
    
    Args:
        means2d: [1, N, 2] - 2D projected coordinates
        normal_map: [1, H, W, 3] - Original normal map
        sampled_normals: [1, N, 3] - Sampled normals from the map
        image_height, image_width: Image dimensions
        save_path: Optional path to save the visualization
        show_comparison: Whether to show side-by-side comparison
    
    Returns:
        comparison_image: Combined visualization image
    """
    import matplotlib.pyplot as plt
    
    device = means2d.device
    
    # Generate image from sampled normals
    reconstructed = generate_image_from_normals(
        sampled_normals, means2d, image_height, image_width,
        rendering_mode='normals', gaussian_size=2.0
    )
    
    # Convert to numpy for visualization
    if normal_map.shape[-1] == 3:  # [1, H, W, 3]
        original_vis = normal_map[0].cpu().numpy()
    else:  # [1, 3, H, W]
        original_vis = normal_map[0].permute(1, 2, 0).cpu().numpy()
    
    reconstructed_vis = reconstructed[0].permute(1, 2, 0).cpu().numpy()
    
    # Ensure values are in [0, 1] for visualization
    original_vis = np.clip((original_vis + 1.0) / 2.0 if original_vis.min() < 0 else original_vis, 0, 1)
    reconstructed_vis = np.clip(reconstructed_vis, 0, 1)
    
    if show_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_vis)
        axes[0].set_title('Original Normal Map')
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed_vis)
        axes[1].set_title('Reconstructed from Sampled Normals')
        axes[1].axis('off')
        
        # Optionally overlay sampling points
        means_np = means2d[0].cpu().numpy()
        axes[1].scatter(means_np[:, 0], means_np[:, 1], 
                       c='red', s=1, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    else:
        return reconstructed_vis

def generate_image_from_normals_batch(sampled_normals, means2d, image_height, image_width, 
                               rendering_mode='shaded', light_dir=None, colors=None,
                               gaussian_size=3.0, opacity=1.0, chunk_size=1000, 
                               use_sparse_rendering=True):
    """
    Generate an image by rendering sampled normals at their 2D projected positions.
    Memory-optimized version with chunked processing and sparse rendering.
    
    Args:
        sampled_normals: [1, N, 3] - Normal vectors for each Gaussian
        means2d: [1, N, 2] - 2D projected coordinates of Gaussians
        image_height, image_width: Image dimensions
        rendering_mode: str - 'shaded', 'normals', 'depth', or 'colored'
        light_dir: [3] - Light direction for shading (default: [0, 0, 1])
        colors: [1, N, 3] - Optional colors for each Gaussian (for 'colored' mode)
        gaussian_size: float - Size of each Gaussian splat
        opacity: float - Opacity of each Gaussian
        chunk_size: int - Number of Gaussians to process at once
        use_sparse_rendering: bool - Only compute pixels within 3*gaussian_size of each point
    
    Returns:
        rendered_image: [1, 3, H, W] - Generated image
    """
    
    device = sampled_normals.device
    dtype = sampled_normals.dtype
    batch_size, num_points, _ = sampled_normals.shape
    
    # Initialize output image
    rendered_image = torch.zeros(batch_size, 3, image_height, image_width, 
                                device=device, dtype=dtype)
    weight_sum = torch.zeros(batch_size, 1, image_height, image_width, 
                           device=device, dtype=dtype)
    
    # Filter out points outside image bounds
    valid_mask = ((means2d[0, :, 0] >= 0) & (means2d[0, :, 0] < image_width) &
                  (means2d[0, :, 1] >= 0) & (means2d[0, :, 1] < image_height))
    
    if not valid_mask.any():
        return rendered_image
    
    valid_means = means2d[0][valid_mask]  # [N_valid, 2]
    valid_normals = sampled_normals[0][valid_mask]  # [N_valid, 3]
    valid_colors = colors[0][valid_mask] if colors is not None else None
    num_valid = valid_means.shape[0]
    
    # Pre-compute colors for all valid points
    all_colors = compute_colors_batch(valid_normals, rendering_mode, light_dir, 
                                     valid_colors, device)  # [N_valid, 3]
    
    # Process in chunks to manage memory
    for start_idx in range(0, num_valid, chunk_size):
        end_idx = min(start_idx + chunk_size, num_valid)
        chunk_means = valid_means[start_idx:end_idx]  # [chunk_size, 2]
        chunk_colors = all_colors[start_idx:end_idx]  # [chunk_size, 3]
        
        if use_sparse_rendering:
            # render_chunk_sparse(rendered_image[0], weight_sum[0], 
            #                   chunk_means, chunk_colors, gaussian_size, opacity,
            #                   image_height, image_width)

            render_chunk_sparse_vectorized(rendered_image[0], weight_sum[0], 
                               chunk_means, chunk_colors, gaussian_size, opacity,
                               image_height, image_width)
        else:
            render_chunk_dense(rendered_image[0], weight_sum[0], 
                              chunk_means, chunk_colors, gaussian_size, opacity,
                              image_height, image_width)
        
        # Clear cache periodically
        if start_idx % (chunk_size * 5) == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Normalize by weights
    weight_sum = torch.clamp(weight_sum, min=1e-6)
    rendered_image = rendered_image / weight_sum
    
    return rendered_image


def compute_colors_batch(normals, rendering_mode, light_dir, colors, device):
    """Compute colors for all normals at once."""
    num_normals = normals.shape[0]
    
    if rendering_mode == 'normals':
        # Visualize normals as RGB (convert from [-1,1] to [0,1])
        return (normals + 1.0) / 2.0
        
    elif rendering_mode == 'shaded':
        # Simple Lambertian shading
        if light_dir is None:
            light_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
        else:
            light_dir = torch.tensor(light_dir, device=device)
        light_dir = F.normalize(light_dir, dim=0)
        
        # Vectorized lighting calculation
        lambertian = torch.clamp(torch.sum(normals * light_dir, dim=1), 0.0, 1.0)  # [N]
        return lambertian.unsqueeze(1).expand(-1, 3)  # [N, 3]
        
    elif rendering_mode == 'depth':
        # Use Z component as depth
        depth_vals = (normals[:, 2] + 1.0) / 2.0  # [N]
        return depth_vals.unsqueeze(1).expand(-1, 3)  # [N, 3]
        
    elif rendering_mode == 'colored':
        # Use provided colors with normal-based shading
        if colors is None:
            colors = torch.ones(num_normals, 3, device=device)
        
        if light_dir is None:
            light_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
        else:
            light_dir = torch.tensor(light_dir, device=device)
        light_dir = F.normalize(light_dir, dim=0)
        
        lambertian = torch.clamp(torch.sum(normals * light_dir, dim=1), 0.1, 1.0)  # [N]
        return colors * lambertian.unsqueeze(1)  # [N, 3]
        
    else:
        raise ValueError(f"Unknown rendering mode: {rendering_mode}")


def render_chunk_sparse(rendered_image, weight_sum, means, colors, gaussian_size, 
                       opacity, image_height, image_width):
    """Sparse rendering - only compute pixels within gaussian radius."""
    radius = int(gaussian_size * 3)  # 3-sigma radius
    sigma_sq = gaussian_size ** 2
    
    for i in range(means.shape[0]):
        cx, cy = int(means[i, 0].item()), int(means[i, 1].item())
        color = colors[i]  # [3]
        
        # Define bounding box
        x_min = max(0, cx - radius)
        x_max = min(image_width, cx + radius + 1)
        y_min = max(0, cy - radius)
        y_max = min(image_height, cy + radius + 1)
        
        if x_min >= x_max or y_min >= y_max:
            continue
        
        # Create local coordinate grids
        y_local = torch.arange(y_min, y_max, device=means.device, dtype=means.dtype)
        x_local = torch.arange(x_min, x_max, device=means.device, dtype=means.dtype)
        yy, xx = torch.meshgrid(y_local, x_local, indexing='ij')
        
        # Calculate distances and weights
        dx = xx - means[i, 0]
        dy = yy - means[i, 1]
        dist_sq = dx * dx + dy * dy
        weights = torch.exp(-dist_sq / (2 * sigma_sq)) * opacity
        
        # Update image
        for c in range(3):
            rendered_image[c, y_min:y_max, x_min:x_max] += weights * color[c]
        weight_sum[0, y_min:y_max, x_min:x_max] += weights


def render_chunk_dense(rendered_image, weight_sum, means, colors, gaussian_size,
                      opacity, image_height, image_width):
    """Dense rendering - compute all pixels (more memory intensive)."""
    device = means.device
    dtype = means.dtype
    chunk_size = means.shape[0]
    
    # Create coordinate grids
    y_coords = torch.arange(image_height, device=device, dtype=dtype)
    x_coords = torch.arange(image_width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Vectorized distance computation
    # means: [chunk_size, 2], grid: [H, W, 2]
    grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    
    # Reshape for broadcasting: [1, 1, chunk_size, 2] - [H, W, 1, 2]
    means_expanded = means.unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_size, 2]
    grid_expanded = grid_coords.unsqueeze(2)  # [H, W, 1, 2]
    
    # Calculate squared distances
    dist_sq = ((grid_expanded - means_expanded) ** 2).sum(dim=-1)  # [H, W, chunk_size]
    
    # Gaussian weights
    sigma_sq = gaussian_size ** 2
    weights = torch.exp(-dist_sq / (2 * sigma_sq)) * opacity  # [H, W, chunk_size]
    
    # Accumulate colors
    colors_expanded = colors.unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_size, 3]
    weighted_colors = weights.unsqueeze(-1) * colors_expanded  # [H, W, chunk_size, 3]
    
    # Sum over gaussians and add to image
    color_contrib = weighted_colors.sum(dim=2)  # [H, W, 3]
    weight_contrib = weights.sum(dim=2)  # [H, W]
    
    rendered_image += color_contrib.permute(2, 0, 1)  # [3, H, W]
    weight_sum[0] += weight_contrib

def render_chunk_sparse_vectorized(rendered_image, weight_sum, means, colors, gaussian_size, 
                                 opacity, image_height, image_width):
    """Fully vectorized sparse rendering."""
    radius = int(gaussian_size * 3)
    sigma_sq = gaussian_size ** 2
    device = means.device
    
    # Get integer centers
    centers = means.round().long()  # [N, 2]
    
    # Create offset grids once
    offsets = torch.arange(-radius, radius + 1, device=device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')
    offset_coords = torch.stack([dx.flatten(), dy.flatten()], dim=1)  # [offset_count, 2]
    
    # Broadcast to get all pixel coordinates for all gaussians
    all_centers = centers.unsqueeze(1)  # [N, 1, 2]
    all_offsets = offset_coords.unsqueeze(0)  # [1, offset_count, 2]
    pixel_coords = all_centers + all_offsets  # [N, offset_count, 2]
    
    # Filter valid coordinates
    valid_mask = ((pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < image_width) &
                  (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < image_height))
    
    # Get valid pixels and their corresponding gaussian indices
    valid_pixels = pixel_coords[valid_mask]  # [valid_count, 2]
    gaussian_indices = torch.arange(means.shape[0], device=device).unsqueeze(1).expand(-1, offset_coords.shape[0])[valid_mask]
    
    # Compute weights for valid pixels
    pixel_float = valid_pixels.float()
    gaussian_centers = means[gaussian_indices]
    dist_sq = ((pixel_float - gaussian_centers) ** 2).sum(dim=1)
    weights = torch.exp(-dist_sq / (2 * sigma_sq)) * opacity
    
    # Use scatter_add for accumulation
    pixel_indices = valid_pixels[:, 1] * image_width + valid_pixels[:, 0]  # Flatten to 1D indices
    
    for c in range(3):
        color_values = colors[gaussian_indices, c] * weights
        rendered_image[c].view(-1).scatter_add_(0, pixel_indices, color_values)
    
    weight_sum[0].view(-1).scatter_add_(0, pixel_indices, weights)

def render_normals_simple(normals, means2d, image_height, image_width):
    """
    Simple normal rendering - just place RGB colors at pixel coordinates.
    
    Args:
        normals: [1, N, 3] - Normal vectors
        means2d: [1, N, 2] - Pixel coordinates
    Returns:
        image: [1, 3, H, W]
    """
    device = normals.device
    image = torch.zeros(1, 3, image_height, image_width, device=device)
    
    # Convert normals to RGB colors ([-1,1] -> [0,1])
    colors = (normals[0] + 1.0) / 2.0  # [N, 3]
    
    # Get integer pixel coordinates
    pixel_coords = means2d[0].round().long()  # [N, 2]
    
    # Filter valid coordinates
    valid_mask = ((pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_width) &
                  (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_height))
    
    valid_coords = pixel_coords[valid_mask]
    valid_colors = colors[valid_mask]
    
    # Place colors at pixel locations
    image[0, :, valid_coords[:, 1], valid_coords[:, 0]] = valid_colors.T
    
    return image

def render_normals_with_interpolation(normals, means2d, image_height, image_width, kernel_size=5):
    """
    Normal rendering with nearest neighbor interpolation and smoothing.
    
    Args:
        normals: [1, N, 3] - Normal vectors
        means2d: [1, N, 2] - Pixel coordinates
        kernel_size: Size of smoothing kernel
    Returns:
        image: [1, 3, H, W]
    """
    device = normals.device
    image = torch.zeros(1, 3, image_height, image_width, device=device)
    
    # Convert normals to RGB colors ([-1,1] -> [0,1])
    colors = (normals[0] + 1.0) / 2.0  # [N, 3]
    
    # Get integer pixel coordinates
    pixel_coords = means2d[0].round().long()  # [N, 2]
    
    # Filter valid coordinates
    valid_mask = ((pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_width) &
                  (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_height))
    
    valid_coords = pixel_coords[valid_mask]
    valid_colors = colors[valid_mask]
    
    # Place colors at pixel locations
    image[0, :, valid_coords[:, 1], valid_coords[:, 0]] = valid_colors.T
    
    # Apply Gaussian blur for smoothing
    padding = kernel_size // 2
    blur_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size**2)
    
    # Create a mask of valid pixels
    mask = torch.zeros(1, 1, image_height, image_width, device=device)
    mask[0, 0, valid_coords[:, 1], valid_coords[:, 0]] = 1.0
    
    # Dilate the mask to fill gaps
    dilated_mask = F.conv2d(mask, blur_kernel, padding=padding)
    dilated_mask = (dilated_mask > 0).float()
    
    # Apply smoothing to each color channel
    smoothed_image = torch.zeros_like(image)
    for c in range(3):
        channel = image[0:1, c:c+1]
        smoothed_channel = F.conv2d(channel, blur_kernel, padding=padding)
        
        # Normalize by the dilated mask to avoid darkening
        mask_sum = F.conv2d(mask, blur_kernel, padding=padding)
        smoothed_channel = torch.where(mask_sum > 0, smoothed_channel / mask_sum, smoothed_channel)
        
        smoothed_image[0, c] = smoothed_channel[0, 0]
    
    # Apply the dilated mask to only show smoothed regions near original points
    final_image = smoothed_image * dilated_mask
    
    return final_image

def add_spatial_smoothness_loss(gaussian_normals, means2d, means3d, k_neighbors=8):
    """
    Add smoothness constraint for spatially close Gaussians
    """
    print("Gaussian Normals Shape:", gaussian_normals.shape)
    # Find k-nearest neighbors in 2D screen space
    distances = torch.cdist(means2d.squeeze(0), means2d.squeeze(0))
    _, neighbor_indices = torch.topk(distances, k_neighbors + 1, largest=False)
    neighbor_indices = neighbor_indices[:, 1:]  # Exclude self
    
    # Calculate normal consistency loss
    neighbor_normals = gaussian_normals.squeeze(0)[neighbor_indices]  # [N, k, 3]
    current_normals = gaussian_normals.squeeze(0).unsqueeze(1)  # [N, 1, 3]
    
    # Weight by 3D distance (closer Gaussians should have more similar normals)
    spatial_weights = 1.0 / (torch.cdist(means3d.squeeze(0), means3d.squeeze(0)).gather(1, neighbor_indices) + 1e-6)
    spatial_weights = F.softmax(spatial_weights, dim=1)
    
    consistency_loss = (1 - F.cosine_similarity(current_normals, neighbor_normals, dim=2))
    weighted_loss = (consistency_loss * spatial_weights).mean()
    
    return weighted_loss

def save_disparity_image(
    disp_gt: torch.Tensor,
    output_path: str,
    points: torch.Tensor = None,
    height: int = None,
    width: int = None,
    colormap: str = 'viridis'
):
    """
    Saves a visualization of the ground truth disparity (disp_gt).

    This function handles both dense (2D) and sparse (1D) disparity tensors.
    For sparse data, it plots the points onto a black canvas. The resulting
    disparity map is normalized and color-mapped for clear visualization.

    Args:
        disp_gt (torch.Tensor): The ground truth disparity tensor.
                                Can be dense [1, H, W] or sparse [1, M].
        output_path (str): The path to save the output image (e.g., "output/disp.png").
        points (torch.Tensor, optional): The coordinates for sparse disparity.
                                         Shape [1, M, 2]. Required if disp_gt is sparse.
        height (int, optional): The height of the output image. Required if disp_gt is sparse.
        width (int, optional): The width of the output image. Required if disp_gt is sparse.
        colormap (str, optional): The matplotlib colormap to apply. Defaults to 'viridis'.
                                  Set to None for a grayscale image.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Move tensor to CPU and convert to numpy, removing batch dimension
    disp_gt_np = disp_gt.detach().cpu().numpy().squeeze()

    # --- Create 2D disparity map from dense or sparse data ---
    if disp_gt_np.ndim == 2:  # Dense case
        disp_map = disp_gt_np
    elif disp_gt_np.ndim == 1:  # Sparse case
        if points is None or height is None or width is None:
            raise ValueError("For sparse disparity, 'points', 'height', and 'width' must be provided.")
        
        # Create a blank (black) image
        disp_map = np.zeros((height, width), dtype=np.float32)
        
        # Get integer coordinates from the points tensor
        points_np = points.detach().cpu().numpy().squeeze()
        x_coords = points_np[:, 0].astype(int)
        y_coords = points_np[:, 1].astype(int)

        # Clamp coordinates to be within image bounds to prevent errors
        x_coords = np.clip(x_coords, 0, width - 1)
        y_coords = np.clip(y_coords, 0, height - 1)

        # Populate the disparity map with sparse values
        disp_map[y_coords, x_coords] = disp_gt_np
    else:
        raise ValueError(f"Unsupported disp_gt shape: {disp_gt.shape}")

    # --- Normalize and colormap the 2D disparity map for visualization ---
    valid_disp = disp_map[disp_map > 1e-6] # Use a small epsilon for floating point
    if valid_disp.size == 0:
        # If no valid points, save a black image
        normalized_disp = np.zeros(disp_map.shape, dtype=np.uint8)
    else:
        # Normalize valid disparity values to the [0, 1] range
        min_disp, max_disp = np.min(valid_disp), np.max(valid_disp)
        if max_disp - min_disp < 1e-6:
             # Handle case where all disparities are the same
            disp_map_scaled = (disp_map > 1e-6).astype(np.float32)
        else:
            disp_map_scaled = (disp_map - min_disp) / (max_disp - min_disp)
            # Mask out the zero values again after normalization
            disp_map_scaled[disp_map < 1e-6] = 0

        if colormap:
            # Use matplotlib to apply colormap
            cmap = plt.get_cmap(colormap)
            # Apply colormap, discard alpha channel, and convert to 8-bit RGB
            normalized_disp = (cmap(disp_map_scaled)[:, :, :3] * 255).astype(np.uint8)
        else:
            # For grayscale, just scale to 255
            normalized_disp = (disp_map_scaled * 255).astype(np.uint8)

    # --- Save the final image ---
    img = Image.fromarray(normalized_disp)
    img.save(output_path)
    print(f"✅ Saved disparity image to {output_path}")

# ------- Depth normal loss =======

def compute_depth_gradients(depth_map: torch.Tensor, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial gradients of depth map.
    
    Args:
        depth_map: Depth map [B, H, W] or [B, 1, H, W]
        normalize: Whether to normalize gradients
        
    Returns:
        Tuple of (grad_x, grad_y) each [B, H, W]
    """
    if depth_map.dim() == 4:
        depth_map = depth_map.squeeze(1)  # Remove channel dim if present
    
    # Sobel operators for gradient computation
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)
    
    # Add channel dimension and apply convolution
    depth_input = depth_map.unsqueeze(1)  # [B, 1, H, W]
    
    grad_x = F.conv2d(depth_input, sobel_x, padding=1)  # [B, 1, H, W]
    grad_y = F.conv2d(depth_input, sobel_y, padding=1)  # [B, 1, H, W]
    
    grad_x = grad_x.squeeze(1)  # [B, H, W]
    grad_y = grad_y.squeeze(1)  # [B, H, W]
    
    if normalize:
        # Normalize gradients to prevent exploding values
        grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        grad_x = grad_x / (grad_norm + 1e-8)
        grad_y = grad_y / (grad_norm + 1e-8)
    
    return grad_x, grad_y

def depth_to_normals_simple(
    depth_map: torch.Tensor,
    camera_K: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Convert depth map to surface normals using finite differences.
    
    Args:
        depth_map: Depth map [B, H, W] or [B, 1, H, W]
        camera_K: Camera intrinsics [3, 3] or [B, 3, 3]
        mask: Optional valid pixel mask [B, H, W]
        
    Returns:
        Surface normals [B, 3, H, W]
    """
    if depth_map.dim() == 4:
        depth_map = depth_map.squeeze(1)  # [B, H, W]
    
    B, H, W = depth_map.shape
    device = depth_map.device
    
    # Handle camera intrinsics
    if camera_K.dim() == 2:
        camera_K = camera_K.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 3]
    
    fx = camera_K[:, 0, 0]  # [B]
    fy = camera_K[:, 1, 1]  # [B]
    
    # Compute depth gradients
    grad_x, grad_y = compute_depth_gradients(depth_map, normalize=False)
    
    # Convert to normal vectors
    # Normal = [-fx * dZ/dx, -fy * dZ/dy, 1] (in camera space)
    normal_x = -fx.view(B, 1, 1) * grad_x  # [B, H, W]
    normal_y = -fy.view(B, 1, 1) * grad_y  # [B, H, W]
    normal_z = torch.ones_like(normal_x)    # [B, H, W]
    
    # Stack to form normals [B, 3, H, W]
    normals = torch.stack([normal_x, normal_y, normal_z], dim=1)
    
    # Normalize
    norm = torch.norm(normals, dim=1, keepdim=True)  # [B, 1, H, W]
    normals = normals / (norm + 1e-8)
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.unsqueeze(1)  # [B, 1, H, W]
        normals = normals * mask
    
    return normals

def compute_depth_smoothness_loss(
    depth_map: torch.Tensor,
    image: torch.Tensor,
    lambda_smooth: float = 0.1
) -> torch.Tensor:
    """
    Compute depth smoothness loss with edge-aware weighting.
    
    Args:
        depth_map: Rendered depth [B, H, W] or [B, 1, H, W]
        image: RGB image [B, H, W, 3] or [B, 3, H, W]
        lambda_smooth: Smoothness weight
        
    Returns:
        Smoothness loss scalar
    """
    if depth_map.dim() == 4:
        depth_map = depth_map.squeeze(1)  # [B, H, W]
    
    if image.dim() == 4 and image.shape[1] == 3:  # [B, 3, H, W]
        image = image.permute(0, 2, 3, 1)  # [B, H, W, 3]
    elif image.dim() == 4:  # [B, H, W, 3]
        pass
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Convert to grayscale for edge detection
    gray = torch.mean(image, dim=-1)  # [B, H, W]
    
    # Compute depth gradients
    depth_grad_x, depth_grad_y = compute_depth_gradients(depth_map, normalize=False)
    
    # Compute image gradients  
    image_grad_x, image_grad_y = compute_depth_gradients(gray, normalize=False)
    
    # Edge-aware weights (smaller gradients where image has edges)
    weight_x = torch.exp(-torch.abs(image_grad_x))  # [B, H, W]
    weight_y = torch.exp(-torch.abs(image_grad_y))  # [B, H, W]
    
    # Weighted depth smoothness
    smooth_loss_x = (weight_x * torch.abs(depth_grad_x)).mean()
    smooth_loss_y = (weight_y * torch.abs(depth_grad_y)).mean()
    
    total_smooth_loss = lambda_smooth * (smooth_loss_x + smooth_loss_y)
    
    return total_smooth_loss

def compute_normal_consistency_loss(
    depth_map: torch.Tensor,
    camera_K: torch.Tensor,
    lambda_normal: float = 0.01,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute normal consistency loss - encourages locally consistent normals.
    
    Args:
        depth_map: Rendered depth [B, H, W] or [B, 1, H, W]
        camera_K: Camera intrinsics [3, 3]
        lambda_normal: Normal consistency weight
        mask: Optional valid pixel mask
        
    Returns:
        Normal consistency loss
    """
    # Get normals from depth
    normals = depth_to_normals_simple(depth_map, camera_K, mask)  # [B, 3, H, W]
    
    # Compute normal gradients (measure of normal variation)
    normal_grad_x = torch.abs(normals[:, :, :, 1:] - normals[:, :, :, :-1])  # [B, 3, H, W-1]
    normal_grad_y = torch.abs(normals[:, :, 1:, :] - normals[:, :, :-1, :])  # [B, 3, H-1, W]
    
    # Total variation in normals
    consistency_loss = lambda_normal * (normal_grad_x.mean() + normal_grad_y.mean())
    
    return consistency_loss

def add_simplified_depth_normal_loss(
    renders: torch.Tensor,
    pixels: torch.Tensor,
    camera_K: torch.Tensor,
    cfg,
    current_loss: torch.Tensor,
    step: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Add simplified depth-normal regularization without requiring rasterizer changes.
    
    Args:
        renders: Rendered outputs [B, H, W, C] where last channel might be depth
        pixels: Ground truth RGB [B, H, W, 3]
        camera_K: Camera intrinsics [3, 3]
        cfg: Config object
        current_loss: Current accumulated loss
        step: Training step
        
    Returns:
        Tuple of (total_loss, smoothness_loss, normal_loss)
    """
    # Check if we have depth channel
    if renders.shape[-1] < 4:
        return current_loss, None, None
    
    # Extract depth from renders
    colors = renders[..., :3]  # [B, H, W, 3]
    depths = renders[..., 3]   # [B, H, W]
    
    # Only apply after warmup period
    if step < getattr(cfg, 'depth_normal_start_step', 1000):
        return current_loss, None, None
    
    # Create validity mask (exclude pixels with zero or very small depth)
    valid_mask = depths > 1e-3  # [B, H, W]
    
    if valid_mask.float().mean() < 0.1:  # Less than 10% valid pixels
        return current_loss, None, None
    
    smoothness_loss = None
    normal_loss = None
    
    try:
        # 1. Depth smoothness loss (edge-aware)
        if getattr(cfg, 'enable_depth_smoothness', True):
            smoothness_loss = compute_depth_smoothness_loss(
                depth_map=depths,
                image=pixels,
                lambda_smooth=getattr(cfg, 'lambda_depth_smooth', 0.1)
            )
            current_loss = current_loss + smoothness_loss
        
        # 2. Normal consistency loss
        if getattr(cfg, 'enable_normal_consistency', True):
            normal_loss = compute_normal_consistency_loss(
                depth_map=depths,
                camera_K=camera_K,
                lambda_normal=getattr(cfg, 'lambda_normal_consistency', 0.01),
                mask=valid_mask
            )
            current_loss = current_loss + normal_loss
            
    except Exception as e:
        print(f"Warning: Error in depth-normal loss computation: {e}")
        
    return current_loss, smoothness_loss, normal_loss

# ------- Depth normal loss end =======