
import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple

def save_depth_as_image(
    depth_map: torch.Tensor,
    save_path: str,
    colormap: str = 'spectral',
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    invalid_color: Tuple[int, int, int] = (0, 0, 0)
) -> None:
    """
    Save depth map as a colored image.
    
    Args:
        depth_map: Depth tensor [H, W] or [1, H, W] or [B, H, W]
        save_path: Path to save the image
        colormap: Matplotlib colormap name
        min_depth: Minimum depth for normalization (auto if None)
        max_depth: Maximum depth for normalization (auto if None)
        invalid_color: RGB color for invalid/zero depth pixels
    """
    # Ensure we have a 2D depth map
    if depth_map.dim() == 3:
        depth_map = depth_map[0]  # Take first batch/channel
    elif depth_map.dim() > 3:
        raise ValueError(f"Unsupported depth map shape: {depth_map.shape}")
    
    # Convert to numpy
    depth_np = depth_map.detach().cpu().numpy()
    
    # Create mask for valid depths
    valid_mask = depth_np > 1e-6
    
    if not valid_mask.any():
        # All depths are invalid, save black image
        Image.fromarray(np.zeros((*depth_np.shape, 3), dtype=np.uint8)).save(save_path)
        return
    
    # Get depth range for normalization
    if min_depth is None:
        min_depth = depth_np[valid_mask].min()
    if max_depth is None:
        max_depth = depth_np[valid_mask].max()
    
    # Avoid division by zero
    if max_depth - min_depth < 1e-6:
        max_depth = min_depth + 1.0
    
    # Normalize depth values
    depth_normalized = np.zeros_like(depth_np)
    depth_normalized[valid_mask] = (depth_np[valid_mask] - min_depth) / (max_depth - min_depth)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored_depth = cmap(depth_normalized)  # Returns RGBA
    
    # Convert to RGB
    rgb_image = (colored_depth[:, :, :3] * 255).astype(np.uint8)
    
    # Set invalid pixels to specified color
    rgb_image[~valid_mask] = invalid_color
    
    # Save image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(rgb_image).save(save_path)

def save_normals_as_image(
    normals: torch.Tensor,
    save_path: str,
    coordinate_system: str = 'camera',
    mask: Optional[torch.Tensor] = None
) -> None:
    """
    Save surface normals as RGB image where XYZ -> RGB.
    
    Args:
        normals: Normal vectors [3, H, W] or [B, 3, H, W]
        save_path: Path to save the image
        coordinate_system: 'camera' or 'world' for interpretation
        mask: Optional mask for valid pixels [H, W] or [B, H, W]
    """
    # Ensure we have [3, H, W] format
    if normals.dim() == 4:
        normals = normals[0]  # Take first batch
    if normals.shape[0] != 3:
        raise ValueError(f"Expected normals with shape [3, H, W], got {normals.shape}")
    
    # Convert to numpy and permute to [H, W, 3]
    normals_np = normals.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Handle mask
    if mask is not None:
        if mask.dim() == 3:
            mask = mask[0]  # Take first batch
        mask_np = mask.detach().cpu().numpy()
    else:
        # Create mask based on normal magnitude
        normal_magnitude = np.linalg.norm(normals_np, axis=2)
        mask_np = normal_magnitude > 1e-6
    
    # Normalize normals to [-1, 1] range
    normals_normalized = np.zeros_like(normals_np)
    normals_normalized[mask_np] = normals_np[mask_np] / (np.linalg.norm(normals_np[mask_np], axis=1, keepdims=True) + 1e-8)
    
    # Convert from [-1, 1] to [0, 255] for RGB visualization
    # X (right) -> Red, Y (up) -> Green, Z (forward) -> Blue
    rgb_normals = ((normals_normalized + 1.0) * 127.5).astype(np.uint8)
    
    # Set invalid pixels to black
    rgb_normals[~mask_np] = [0, 0, 0]
    
    # Save image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(rgb_normals).save(save_path)

def create_depth_normal_comparison(
    depth_map: torch.Tensor,
    normals: torch.Tensor,
    rgb_image: torch.Tensor,
    save_path: str,
    titles: Optional[list] = None
) -> None:
    """
    Create a side-by-side comparison of RGB, depth, and normals.
    
    Args:
        depth_map: Depth tensor [H, W] or similar
        normals: Normal vectors [3, H, W] or similar
        rgb_image: RGB image [H, W, 3] or [3, H, W]
        save_path: Path to save comparison image
        titles: Optional titles for subplots
    """
    if titles is None:
        titles = ['RGB Image', 'Depth Map', 'Surface Normals']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB Image
    if rgb_image.dim() == 3 and rgb_image.shape[0] == 3:
        rgb_np = rgb_image.permute(1, 2, 0).detach().cpu().numpy()
    else:
        rgb_np = rgb_image.detach().cpu().numpy()
    rgb_np = np.clip(rgb_np, 0, 1)
    axes[0].imshow(rgb_np)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    # Depth Map
    if depth_map.dim() > 2:
        depth_map = depth_map[0] if depth_map.dim() == 3 else depth_map[0, 0]
    depth_np = depth_map.detach().cpu().numpy()
    valid_mask = depth_np > 1e-6
    if valid_mask.any():
        depth_min, depth_max = depth_np[valid_mask].min(), depth_np[valid_mask].max()
        im1 = axes[1].imshow(depth_np, cmap='viridis', vmin=depth_min, vmax=depth_max)
        plt.colorbar(im1, ax=axes[1], shrink=0.8)
    else:
        axes[1].imshow(depth_np, cmap='gray')
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    # Surface Normals
    if normals.dim() == 4:
        normals = normals[0]
    normals_np = normals.detach().cpu().numpy().transpose(1, 2, 0)
    # Normalize and convert to RGB
    normal_magnitude = np.linalg.norm(normals_np, axis=2, keepdims=True)
    valid_normals = normal_magnitude[..., 0] > 1e-6
    normals_rgb = np.zeros_like(normals_np)
    if valid_normals.any():
        normals_rgb[valid_normals] = (normals_np[valid_normals] / normal_magnitude[valid_normals] + 1.0) * 0.5
    axes[2].imshow(normals_rgb)
    axes[2].set_title(titles[2])
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_depth_and_normals_during_training(
    renders: torch.Tensor,
    pixels: torch.Tensor,
    camera_K: torch.Tensor,
    step: int,
    save_dir: str,
    save_frequency: int = 500,
    world_rank: int = 0
) -> None:
    """
    Save depth maps and computed normals during training.
    
    Args:
        renders: Rendered outputs [B, H, W, C] where last channel might be depth
        pixels: Ground truth RGB [B, H, W, 3]
        camera_K: Camera intrinsics [3, 3]
        step: Current training step
        save_dir: Directory to save images
        save_frequency: Save every N steps
        world_rank: Current process rank (for multi-GPU)
    """
    # Only save on specified frequency and for main process
    if step % save_frequency != 0 or world_rank != 0:
        return
    
    # Check if we have depth channel
    if renders.shape[-1] < 4:
        print(f"Warning: No depth channel found in renders at step {step}")
        return
    
    try:
        # Extract components
        colors = renders[0, ..., :3]  # Take first batch [H, W, 3]
        depths = renders[0, ..., 3]   # Take first batch [H, W]
        gt_colors = pixels[0]         # Take first batch [H, W, 3]
        
        # Compute normals from depth
        from utils import depth_to_normals_simple
        normals = depth_to_normals_simple(
            depth_map=depths.unsqueeze(0),  # Add batch dim [1, H, W]
            camera_K=camera_K
        )[0]  # Remove batch dim -> [3, H, W]
        
        # Create save directory
        depth_dir = os.path.join(save_dir, "depth_maps")
        normal_dir = os.path.join(save_dir, "normal_maps") 
        comparison_dir = os.path.join(save_dir, "depth_normal_comparisons")
        
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Save individual components
        depth_path = os.path.join(depth_dir, f"depth_step_{step:06d}.png")
        normal_path = os.path.join(normal_dir, f"normals_step_{step:06d}.png")
        comparison_path = os.path.join(comparison_dir, f"comparison_step_{step:06d}.png")
        
        # Save depth map
        save_depth_as_image(depths, depth_path)
        
        # Save normals
        save_normals_as_image(normals, normal_path)
        
        # Save comparison
        create_depth_normal_comparison(
            depth_map=depths,
            normals=normals,
            rgb_image=gt_colors,
            save_path=comparison_path,
            titles=[f'GT RGB (Step {step})', f'Rendered Depth (Step {step})', f'Computed Normals (Step {step})']
        )
        
        # Also save rendered RGB for reference
        rendered_rgb_path = os.path.join(save_dir, "rendered_rgb", f"rgb_step_{step:06d}.png")
        os.makedirs(os.path.dirname(rendered_rgb_path), exist_ok=True)
        rgb_np = (colors.detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_np).save(rendered_rgb_path)
        
        print(f"Saved depth/normal visualizations for step {step}")
        
    except Exception as e:
        print(f"Error saving depth/normal visualizations at step {step}: {e}")

def add_depth_normal_visualization_to_training_loop(
    renders: torch.Tensor,
    pixels: torch.Tensor,
    camera_K: torch.Tensor,
    step: int,
    cfg,
    world_rank: int = 0
) -> None:
    """
    Convenience function to add to your training loop.
    
    Usage in training loop:
    ```python
    # After computing renders and before loss computation
    if cfg.save_depth_normal_vis:
        add_depth_normal_visualization_to_training_loop(
            renders=renders,
            pixels=pixels, 
            camera_K=Ks[0],
            step=step,
            cfg=cfg,
            world_rank=world_rank
        )
    ```
    """
    if not getattr(cfg, 'save_depth_normal_vis', False):
        return
    
    save_frequency = getattr(cfg, 'depth_normal_vis_frequency', 500)
    save_dir = getattr(cfg, 'result_dir', './visuals')
    
    save_depth_and_normals_during_training(
        renders=renders,
        pixels=pixels,
        camera_K=camera_K,
        step=step,
        save_dir=save_dir,
        save_frequency=save_frequency,
        world_rank=world_rank
    )