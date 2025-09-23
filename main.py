import sys
import os
# ...existing code...
# ensure 'examples' directory is on sys.path so `from datasets.colmap` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))

import numpy as np
import torch 
import torchvision.utils
from torch import Tensor
from typing import Optional
import warnings
import argparse
import time

from examples.simple_trainer import create_splats_with_optimizers, Config, Parser, Dataset, Runner
from gsplat.utils import load_ply, load_ply_milo

class GaussianModel:
    def __init__(self, config, device):
        """
        Initializes the GaussianModel.

        Args:
            config: An object or dict containing model and optimizer parameters
                    (e.g., sh_degree, means_lr, scales_lr, etc.).
        """
        self.splats = None
        self.optimizers = None
        self.config = config

    def setup_model_and_optimizers(self, parser):
        """Creates the initial splats and optimizers."""
        feature_dim = 32 if cfg.app_opt else None
        world_rank, world_size = 0, 1  # Assuming single GPU for simplicity
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=1.0,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model and optimizers created.")

    def _recreate_optimizers(self):
        """
        Re-initializes the optimizers to be linked to the current self.splats parameters.
        This is crucial after loading new splat data.
        """
        if self.splats is None:
            warnings.warn("Cannot create optimizers without splat parameters.")
            return

        # This logic is adapted from the end of `create_splats_with_optimizers`
        params_with_lr = [
            ("means", self.config.means_lr),
            ("scales", self.config.scales_lr),
            ("quats", self.config.quats_lr),
            ("opacities", self.config.opacities_lr),
            ("sh0", self.config.sh0_lr),
            ("shN", self.config.shN_lr),
        ]
        
        new_optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr, "name": name}],
                eps=1e-15
            )
            for name, lr in params_with_lr if name in self.splats
        }
        self.optimizers = new_optimizers
        print("Optimizers have been re-initialized.")


    def load_splats_from_ply(self, path: str):
        """
        Loads splats from a .ply file, pads SH coefficients to match model config,
        and replaces the current model parameters. This invalidates and recreates 
        the optimizers.
        """
        # 1. Load the raw data from the .ply file into a dictionary of tensors
        loaded_splats_dict = load_ply(path, device="cuda")

        # --- NEW: Check and Pad Spherical Harmonics ---
        sh0 = loaded_splats_dict.get("sh0")
        shN = loaded_splats_dict.get("shN")
        
        # Get the desired sh_degree from your configuration.
        # We default to 3 if it's not specified in self.config.
        sh_degree = getattr(self.config, 'sh_degree', 3) 

        if sh0 is not None and shN is not None:
            # The total number of SH coefficients needed is (sh_degree + 1)^2.
            # We subtract 1 because sh0 is the first coefficient.
            desired_shN_features = (sh_degree + 1)**2 - 1
            current_shN_features = shN.shape[1]

            # If the loaded data has fewer SH coefficients than desired...
            if current_shN_features < desired_shN_features:
                num_points = shN.shape[0]
                
                # Calculate the number of missing feature dimensions
                num_features_to_pad = desired_shN_features - current_shN_features
                
                print(f"INFO: Padding SH coefficients to support sh_degree={sh_degree}. "
                    f"Adding {num_features_to_pad} zeroed features to 'shN'.")

                # Create a tensor of zeros with the correct shape for padding
                padding_shape = (num_points, num_features_to_pad, 3)
                padding = torch.zeros(padding_shape, dtype=shN.dtype, device=shN.device)
                
                # Concatenate the existing shN with the zero padding
                loaded_splats_dict["shN"] = torch.cat([shN, padding], dim=1)
        
        # 3. Create a new ParameterDict from the (now padded) data
        self.splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v) for k, v in loaded_splats_dict.items()}
        ).to("cuda")

        # 4. Re-create the optimizers
        print("Splat parameters have been replaced. Re-initializing optimizers...")
        self._recreate_optimizers()

def save_rendered_image(
    render_tensor: Tensor,
    filepath: str
):
    """
    Saves a rendered image tensor to a file.

    Args:
        render_tensor (Tensor): The output tensor from the rasterization function,
                                expected shape [..., H, W, Channels].
        filepath (str): The path to save the image file (e.g., "output/render.png").
    """
    # Remove any batch dimensions (assuming we are saving one image at a time)
    image_tensor = render_tensor.squeeze()  # Shape: [H, W, Channels]

    # Handle cases where depth is included in the output (e.g., RGB+D)
    # We only save the first 3 channels (RGB)
    if image_tensor.shape[-1] > 3:
        print(f"Warning: Tensor has {image_tensor.shape[-1]} channels. Saving only the first 3 (RGB).")
        image_tensor = image_tensor[..., :3]

    # torchvision.utils.save_image expects the channel dimension to be first: [C, H, W]
    image_tensor_chw = image_tensor.permute(2, 0, 1)

    # Ensure the directory exists before saving
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Save the image
    torchvision.utils.save_image(image_tensor_chw, filepath)
    print(f"✅ Rendered image saved to {filepath}")



# --- Example of how to use the class ---
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--ply_path', type=str, default="/home/te/projects/InnovationDepot/up_mid_down_video/milo_output_superpoint/point_cloud/iteration_18000/point_cloud.ply", required=False, help='Path to the .ply file to load splats from')
    # parser.add_argument("--data_dir", type=str, default="/home/te/projects/InnovationDepot/up_mid_down_video", help="Path to the dataset directory")
    parser.add_argument('--ply_path', type=str, default="/home/te/projects/InnovationDepot/hloc_output_disk/gsplat_output_dapda_geometric1/ply/point_cloud_6999.ply", required=False, help='Path to the .ply file to load splats from')
    parser.add_argument("--data_dir", type=str, default="/home/te/projects/InnovationDepot/hloc_output_disk/", help="Path to the dataset directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize the model and create some initial splats
    cfg = Config()
    cfg.data_dir  = args.data_dir
    cfg.data_factor = 1
    cfg.use_bilateral_grid = False 
    parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=cfg.test_every,
        load_normals=cfg.load_normals
        )
    
    dataset = Dataset(
        parser, 
        split="train",
        patch_size=cfg.patch_size,
        load_depths=cfg.depth_loss,
        use_precomputed_depths=cfg.use_precomputed_depths
    )
    
    model = GaussianModel(cfg, device)
    # model.setup_model_and_optimizers(parser) # You would normally do this

    # 2. Assume a 'splats.ply' file exists. Load it to replace the model's data.
    # Let's create a dummy file first using your save function for a complete example.
    
    # Create dummy splats to save
    # dummy_splats_to_save, _ = create_splats_with_optimizers(parser)
    # save_ply(dummy_splats_to_save, "dummy_splats.ply")

    # Now, load the saved splats into our model instance
    model.load_splats_from_ply(args.ply_path)
    print("\nModel state after loading:")
    print(f"Number of loaded splats: {model.splats['means'].shape[0]}")
    print(f"Optimizers are ready: {'Yes' if model.optimizers else 'No'}")
    

    world_rank, world_size = 0, 1 
    runner = Runner(local_rank=0, world_rank=world_rank, world_size=world_size, cfg=cfg)

    runner.splats = model.splats
    runner.optimizers = model.optimizers

    print("Runner generated")

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    trainloader_iter = iter(trainloader)

    data = next(trainloader_iter)

    print("Data loaded")

    camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)
    Ks = data["K"].to(device) 
    pixels = data["image"].to(device) / 255.0 
    height, width = pixels.shape[1:3]
    sh_degree_to_use = 3
    near_plane=cfg.near_plane,
    far_plane=cfg.far_plane,
    image_ids = data["image_id"].to(device)
    render_mode="RGB",
    masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]

    print("Starting rasterization")

    renders, alphas, info = runner.rasterize_splats(
        camtoworlds=camtoworlds,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree_to_use,
        near_plane=cfg.near_plane,
        far_plane=cfg.far_plane,
        image_ids=image_ids,
        render_mode="RGB+ED" if cfg.depth_loss else "RGB",
        masks=masks,
    )

    print("Rasterization complete")

    save_rendered_image(
        renders,
        "output/rendered_image1.png"
    )

    for i in range(200):
        paused = True
        if i == 100:
            paused = False
        while paused:
            time.sleep(0.1)
        print(f"Step {i+1}/200")
        time.sleep(0.1)
