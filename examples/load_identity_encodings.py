import torch
import numpy as np
from pathlib import Path
import yaml
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial.distance import cdist
from collections import Counter
import argparse

from simple_trainer import Config, Parser, Dataset
from gsplat.rendering import rasterization


def load_and_inspect_identity_encodings(
    checkpoint_path: str,
    num_gaussians_to_show: int = 10,
    save_to_file: str = None
):
    """
    Load identity encodings from a checkpoint and display information.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        num_gaussians_to_show: Number of Gaussian encodings to print (default: 10)
        save_to_file: Optional path to save all encodings as .npy file
    
    Returns:
        identity_encodings: The full tensor of identity encodings [N, D]
    """
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint keys: ", list(checkpoint.keys()))

    # Extract identity encodings
    if 'splats' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'splats' dictionary")
    
    if 'identity_encodings' not in checkpoint['splats']:
        raise KeyError("Checkpoint does not contain 'identity_encodings'. "
                      "Was the model trained with 'with_segmentation=True'?")
    
    identity_encodings = checkpoint['splats']['identity_encodings']
    
    # Print summary statistics
    print("\n" + "="*60)
    print("IDENTITY ENCODINGS SUMMARY")
    print("="*60)
    print(f"Total number of Gaussians: {identity_encodings.shape[0]:,}")
    print(f"Encoding dimension: {identity_encodings.shape[1]}")
    print(f"Tensor dtype: {identity_encodings.dtype}")
    print(f"Tensor device: {identity_encodings.device}")
    
    # Statistics
    print("\nStatistics across all encodings:")
    print(f"  Mean: {identity_encodings.mean().item():.6f}")
    print(f"  Std:  {identity_encodings.std().item():.6f}")
    print(f"  Min:  {identity_encodings.min().item():.6f}")
    print(f"  Max:  {identity_encodings.max().item():.6f}")
    
    # Show sample encodings
    num_to_show = min(num_gaussians_to_show, identity_encodings.shape[0])
    print(f"\n{'='*60}")
    print(f"SAMPLE ENCODINGS (first {num_to_show} Gaussians)")
    print("="*60)
    
    for i in range(num_to_show):
        encoding = identity_encodings[i]
        print(f"\nGaussian {i}:")
        print(f"  Encoding vector: {encoding.numpy()}")
        print(f"  Norm: {encoding.norm().item():.4f}")
    
    # Optional: compute pairwise similarities for first few
    if num_to_show > 1:
        print(f"\n{'='*60}")
        print(f"PAIRWISE COSINE SIMILARITIES (first {min(5, num_to_show)} Gaussians)")
        print("="*60)
        
        sample_encodings = identity_encodings[:min(5, num_to_show)]
        # Normalize for cosine similarity
        normalized = torch.nn.functional.normalize(sample_encodings, dim=1)
        similarity_matrix = torch.mm(normalized, normalized.t())
        
        print("\nSimilarity matrix:")
        print(similarity_matrix.numpy())
        print("\n(Values close to 1.0 indicate similar encodings)")
    
    # Save to file if requested
    if save_to_file:
        save_path = Path(save_to_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, identity_encodings.numpy())
        print(f"\n{'='*60}")
        print(f"Saved all {identity_encodings.shape[0]:,} encodings to: {save_to_file}")
        print("="*60)
    
    return identity_encodings

def get_identity_map_from_checkpoint(
    cfg, 
    ckpt_path: Path, 
    image_index: int = 0, 
    device: str = "cuda"
):
    """
    Loads a trained model and renders the identity map for a specific camera view.
    """
    # print(f"Loading config from: {config_path}")
    # cfg_dict = yaml.safe_load(config_path.read_text())
    # # We only need a few parameters from the original config
    # cfg = Config(
    #     data_dir=cfg_dict['data_dir'],
    #     data_factor=cfg_dict['data_factor'],
    #     identity_dim=cfg_dict['identity_dim']
    # )

    # 1. --- Load the trained models and splat data ---
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Re-create the segmentation head MLP
    segmentation_head = torch.nn.Sequential(
        torch.nn.Linear(cfg.identity_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, cfg.identity_dim),
    ).to(device)
    
    # Load the saved weights
    if "segmentation_head" in checkpoint:
        segmentation_head.load_state_dict(checkpoint["segmentation_head"])
        segmentation_head.eval() # Set to evaluation mode
        print("✅ Segmentation head loaded successfully.")
    else:
        raise KeyError("Checkpoint does not contain 'segmentation_head' state_dict. "
                       "Please re-save your checkpoint using the updated script.")

    # Load splat parameters
    splats = checkpoint['splats']
    for key, value in splats.items():
        splats[key] = value.to(device)
    print("✅ Gaussian splat data loaded successfully.")


    # 2. --- Prepare a camera view from the dataset ---
    print(f"Loading camera view {image_index} from dataset...")
    parser = Parser(data_dir=cfg.data_dir, factor=cfg.data_factor)
    val_set = Dataset(parser, split="val")
    
    # Get a single camera view
    val_data = val_set[image_index]
    camtoworld = val_data["camtoworld"].unsqueeze(0).to(device) # Add batch dim
    K = val_data["K"].unsqueeze(0).to(device)                   # Add batch dim
    height, width = val_data["image"].shape[:2]


    # 3. --- Render the Identity Map ---
    with torch.no_grad():
        print("Processing raw identity encodings through the MLP...")
        raw_identities = splats["identity_encodings"]
        processed_identities = segmentation_head(raw_identities)

        print(f"Rendering identity map of size {width}x{height}...")
        # Note: The rasterization function takes the inverse of camtoworlds (viewmats)
        viewmats = torch.linalg.inv(camtoworld)

        # Call the rasterizer with the processed identities as override features
        identity_map, _, _ = rasterization(
            # means=splats["means"],
            # quats=splats["quats"],
            # scales=torch.exp(splats["scales"]),
            # opacities=torch.sigmoid(splats["opacities"]),
            colors=processed_identities, # <-- KEY STEP
            viewmats=viewmats,
            Ks=K,
            width=width,
            height=height,
            sh_degree=-1, # Important: Treat features as raw vectors, not SH
        )
    
    print("✅ Identity map generated successfully!")
    return identity_map.squeeze(0).cpu().numpy() # Remove batch dim and move to CPU

def DBSCAN_identity_encodings(encodings, eps, min_samples):
    print(f"Running DBSCAN with eps={eps} and min_samples={min_samples}...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(encodings)
    labels = clustering.labels_
    label_counts = Counter(labels)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"✅ Clustering complete!")
    print(f"Found {num_clusters} potential dominoes (clusters).")
    print(f"Number of Gaussians in each cluster:\n{label_counts}")


def kmeans_identity_encodings(identity_map, instance_mask, identity_encodings):
    anchor_encodings = {}
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]  # Exclude background (0)
    print("Finding anchor encodings for each object ID...")
    for obj_id in unique_ids:
        # Create a boolean mask for the current object
        mask = (instance_mask == obj_id)
        
        # Select the feature vectors from the identity map that correspond to this object
        object_features = identity_map[mask]
        
        # Calculate the mean vector for this object
        mean_feature = np.mean(object_features, axis=0)
        
        anchor_encodings[obj_id.item()] = mean_feature
        print(f"  - Found anchor for Object ID {obj_id.item()}")

    all_features = encodings
    NUM_CLUSTERS = len(anchor_encodings)
    print(f"\nRunning KMeans with {NUM_CLUSTERS} clusters")

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto').fit(all_features)

    # These are the results from K-Means:
    # 1. The cluster label (0, 1, or 2) assigned to each Gaussian
    kmeans_labels = kmeans.labels_
    # 2. The center (centroid) of each of the 3 clusters found
    kmeans_centroids = kmeans.cluster_centers_

    print("✅ K-Means clustering complete.")

    # Get your anchor vectors and their corresponding object IDs
    anchor_ids = list(anchor_encodings.keys())
    anchor_vectors = np.array(list(anchor_encodings.values()))

    # Calculate the distance between each K-Means centroid and each anchor vector
    # This creates a distance matrix: rows are K-Means clusters, columns are your objects
    distance_matrix = cdist(kmeans_centroids, anchor_vectors, 'euclidean')

    # For each K-Means cluster, find the closest anchor object ID
    # `argmin(axis=1)` finds the index of the minimum value in each row
    closest_anchor_indices = np.argmin(distance_matrix, axis=1)

    # Create the final mapping from the arbitrary K-Means label to your actual object ID
    # e.g., {0: 2, 1: 3, 2: 1} means "K-Means cluster 0 is actually Object ID 2"
    kmeans_to_object_id_map = {kmeans_label: anchor_ids[anchor_idx] for kmeans_label, anchor_idx in enumerate(closest_anchor_indices)}

    print("\nMapping K-Means clusters to Object IDs:")
    print(kmeans_to_object_id_map)

    # Initialize a dictionary to hold the lists of Gaussian indices
    # Use the real object IDs as keys
    object_groups = {obj_id: [] for obj_id in anchor_ids}
    object_groups["background"] = []

    # Keep track of all Gaussians that get assigned to a domino
    assigned_indices = set()

    # Iterate through every Gaussian and its assigned K-Means label
    for gaussian_idx, kmeans_label in enumerate(kmeans_labels):
        # Look up the real object ID using our map
        object_id = kmeans_to_object_id_map[kmeans_label]
        
        # Add the Gaussian's index to the correct object group
        object_groups[object_id].append(gaussian_idx)
        assigned_indices.add(gaussian_idx)

    # Now, find all the unassigned Gaussians for the background group
    num_total_gaussians = all_features.shape[0]
    all_indices = set(range(num_total_gaussians))
    background_indices = all_indices - assigned_indices
    object_groups["background"] = list(background_indices)


    print("\n✅ Final Object Groups Assembled!")
    for name, group in object_groups.items():
        print(f"  - Group '{name}': {len(group)} Gaussians")


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Uniform Data Cropping Utility (v2.1)")
    parser.add_argument("--data_dir", required=False, help="Input folder")
    parser.add_argument("--data_factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--identity_dim", type=int, default=16, help="Identity encoding dimension")
    args = parser.parse_args()
    # Update this path to your actual checkpoint
    CHECKPOINT_PATH = "/home/te/projects/splat_rigid_body/output/gsplat_spmax/ckpts/ckpt_6999_rank0.pt"
    
    # Load and inspect encodings
    encodings = load_and_inspect_identity_encodings(
        checkpoint_path=CHECKPOINT_PATH,
        num_gaussians_to_show=10,
        # save_to_file="identity_encodings.npy"  # Optional: save to file
    )

    identity_map = np.load("/home/te/projects/splat_rigid_body/output/gsplat_spmax/identity_maps/identity_map_step7000.npy")
    # print("Identity map shape ", identity_map.shape)
    # print("Unique identity encodings in map ", np.unique(identity_map))
    instance_mask = np.load("/home/te/projects/splat_rigid_body/output/gsplat_spmax/identity_maps/instance_mask_step7000.npy")
    print("Instance mask shape ", instance_mask.shape)
    print("Unique instance ids in mask ", np.unique(instance_mask))

    # identity_map = get_identity_map_from_checkpoint(ckpt_path=Path(CHECKPOINT_PATH),
    #                                  cfg=args,
    #                                  image_index=0)

    # DBSCAN_identity_encodings(encodings.numpy(), eps=0.5, min_samples=32)
    kmeans_identity_encodings(identity_map, instance_mask, encodings.numpy())

    
    print(f"\nReturned tensor shape: {encodings.shape}")