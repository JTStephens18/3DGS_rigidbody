import torch
from torch.nn import functional as F
import numpy as np
import re
from pathlib import Path
import yaml
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

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

def dbscan_identity_encodings(identity_encodings: torch.Tensor):
    """
    Performs DBSCAN clustering on normalized identity encodings to find object
    groups and a background group from noise points.
    
    Args:
        identity_encodings (torch.Tensor): The [N, D] tensor of features for each Gaussian.
    """
    if not isinstance(identity_encodings, torch.Tensor):
        raise TypeError("Input 'identity_encodings' must be a PyTorch tensor.")

    print("✅ Normalizing identity encodings...")
    # 1. Normalize the encodings, as this is what the network was trained on.
    # This makes the Euclidean distance used by DBSCAN more meaningful.
    all_features = F.normalize(identity_encodings, dim=1).cpu().numpy()

    # --- Parameter Tuning Section ---
    print("\n🔎 Searching for the best 'eps' value...")
    
    # Set a fixed min_samples. This usually requires less tuning than eps.
    # It should be large enough to not consider tiny random groups as objects.
    MIN_SAMPLES = 200 
    
    best_eps = None
    
    # Iterate through a range of potential eps values
    for eps_candidate in np.arange(0.1, 0.5, 0.02):
        # 2. Run DBSCAN with the candidate eps
        db = DBSCAN(eps=eps_candidate, min_samples=MIN_SAMPLES, n_jobs=-1).fit(all_features)
        
        # Get the unique cluster labels. Noise is labeled -1.
        unique_labels = set(db.labels_)
        
        # Calculate the number of actual clusters found (excluding the noise group)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        print(f"  - Trying eps={eps_candidate:.2f}... Found {num_clusters} clusters.")

        # 3. Check if we found the desired number of clusters
        if num_clusters == 3:
            print(f"🎉 Found optimal eps: {eps_candidate:.2f}\n")
            best_eps = eps_candidate
            break
            
    if best_eps is None:
        print("\n⚠️ Could not automatically find an 'eps' that resulted in 3 clusters.")
        print("Please try adjusting the search range in the script or the MIN_SAMPLES value.")
        return

    # --- Final Clustering and Grouping ---
    print("Running final DBSCAN with optimal parameters...")
    final_db = DBSCAN(eps=best_eps, min_samples=MIN_SAMPLES, n_jobs=-1).fit(all_features)
    labels = final_db.labels_
    
    # The number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points (background): {n_noise_}')

    # 4. Assemble the final object groups
    # Important: The cluster labels from DBSCAN (0, 1, 2...) are arbitrary.
    # You will still need your anchor-based mapping logic from the K-Means script
    # to assign these to your true object IDs (1, 2, 3 from the mask).
    
    # For now, we can group by the arbitrary DBSCAN labels.
    object_groups = {}
    for label in set(labels):
        if label == -1:
            # All noise points are considered background
            group_name = "background"
        else:
            # These are your object clusters
            group_name = f"object_cluster_{label}"
        
        # Find all indices where the label matches
        indices = np.where(labels == label)[0]
        object_groups[group_name] = indices.tolist()

    print("\n✅ Final Object Groups Assembled!")
    for name, group in object_groups.items():
        print(f"  - Group '{name}': {len(group)} Gaussians")
        
    return object_groups

def kmeans_identity_encodings(
    identity_map: np.ndarray,
    instance_mask: np.ndarray,
    identity_encodings: torch.Tensor,
    tsne: bool = True
):
    """
    Performs K-Means clustering on Gaussian identity encodings using an intelligent
    initialization strategy based on anchor vectors from a rendered identity map.

    Args:
        identity_map: A rendered feature map [H, W, D] from a single view.
        instance_mask: A ground-truth instance mask [H, W] for the same view.
        identity_encodings: The full tensor of identity encodings for all Gaussians [N, D].
        output_dir: Directory to save the t-SNE plot.
        tsne: Whether to generate and save a t-SNE visualization.

    Returns:
        A dictionary mapping object IDs (including "background") to lists of
        Gaussian indices belonging to that object.
    """
    # 1. Calculate Normalized Anchor Vectors for Initialization
    print("Finding normalized anchor vectors for K-Means initialization...")
    anchor_vectors_list = []
    anchor_ids_list = []
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]  # Exclude background

    for obj_id in unique_ids:
        mask = (instance_mask == obj_id)
        object_features = identity_map[mask]
        mean_feature = np.mean(object_features, axis=0)
        
        # CRITICAL: Normalize the anchor vector to match the data being clustered
        norm_feat = mean_feature / (np.linalg.norm(mean_feature) + 1e-6)
        
        anchor_vectors_list.append(norm_feat)
        anchor_ids_list.append(obj_id.item())
        print(f"  - Calculated anchor for Object ID {obj_id.item()}")

    initial_centroids = np.array(anchor_vectors_list)
    NUM_CLUSTERS = len(initial_centroids)

    # 2. Create a Reliable Mapping *Before* Clustering
    # The mapping is certain because we control the initial centroid order.
    # K-Means cluster 'i' will correspond to the i-th anchor vector we provide.
    kmeans_to_object_id_map = {i: obj_id for i, obj_id in enumerate(anchor_ids_list)}
    print("\nUsing pre-defined mapping based on initialization:")
    print(kmeans_to_object_id_map)

    # 3. Normalize the Full Feature Set for Clustering
    # This aligns the data with the training loss metric (cosine similarity).
    all_features = F.normalize(identity_encodings.cpu(), dim=1).numpy()

    # 4. Run K-Means with Intelligent Initialization
    print(f"\nRunning KMeans with {NUM_CLUSTERS} clusters (using anchor initialization)")
    kmeans = KMeans(
        n_clusters=NUM_CLUSTERS,
        init=initial_centroids,  # Use anchors as the starting point
        n_init=1,                # Only need one run with a good initialization
        random_state=42
    ).fit(all_features)
    print("✅ K-Means clustering complete.")

    kmeans_labels = kmeans.labels_

    # 5. Optional: Generate t-SNE visualization with the new, correct labels
    if tsne:
        print("Running t-SNE for 2D visualization (this may take a moment)...")
        tsne_model = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300, random_state=42)
        tsne_results = tsne_model.fit_transform(all_features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.5, s=5)
        
        # Create legend labels from our known mapping
        legend_labels = [f'Cluster {i} (Obj ID {kmeans_to_object_id_map[i]})' for i in range(NUM_CLUSTERS)]
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
        
        plt.title('t-SNE Visualization of Identity Encodings')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)
        plt.savefig(f"{args.data_dir}/identity_maps/tsne_visualization_new.png")
        print(f"✅ Saved t-SNE plot")

    # 6. Assemble Final Object Groups Using the Reliable Map
    object_groups = {obj_id: [] for obj_id in anchor_ids_list}
    assigned_indices = set()

    for gaussian_idx, kmeans_label in enumerate(kmeans_labels):
        # Check if the label is valid (sometimes K-Means can produce an unexpected label)
        if kmeans_label in kmeans_to_object_id_map:
            object_id = kmeans_to_object_id_map[kmeans_label]
            object_groups[object_id].append(gaussian_idx)
            assigned_indices.add(gaussian_idx)

    # Assign all remaining Gaussians to a "background" group
    num_total_gaussians = all_features.shape[0]
    all_indices = set(range(num_total_gaussians))
    background_indices = all_indices - assigned_indices
    object_groups["background"] = list(background_indices)
    
    print("\n✅ Final Object Groups Assembled!")
    for name, group in object_groups.items():
        print(f"  - Group '{name}': {len(group)} Gaussians")
        
    return object_groups


def kmeans_identity_encodings_background(
    identity_map: np.ndarray,
    instance_mask: np.ndarray,
    identity_encodings: torch.Tensor,
    background_percentile: float = 95.0,
    tsne: bool = True
):
    """
    Performs K-Means clustering to segment Gaussians, with a post-processing
    step to identify and assign outliers to a background group.

    Args:
        identity_map: A rendered feature map [H, W, D] from a single view.
        instance_mask: A ground-truth instance mask [H, W] for the same view.
        identity_encodings: The full tensor of identity encodings [N, D].
        output_dir: Directory to save the t-SNE plot.
        background_percentile: The distance percentile within each cluster
            above which a Gaussian is considered background (e.g., 95.0 means
            the 5% furthest Gaussians are re-assigned).
        tsne: Whether to generate and save a t-SNE visualization.

    Returns:
        A dictionary mapping object IDs (including "background") to lists of
        Gaussian indices belonging to that object.
    """
    # 1. Calculate Normalized Anchor Vectors for Initialization
    print("Finding normalized anchor vectors for K-Means initialization...")
    anchor_vectors_list = []
    anchor_ids_list = []
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]

    for obj_id in unique_ids:
        mask = (instance_mask == obj_id)
        if not np.any(mask): continue
        object_features = identity_map[mask]
        mean_feature = np.mean(object_features, axis=0)
        norm_feat = mean_feature / (np.linalg.norm(mean_feature) + 1e-6)
        anchor_vectors_list.append(norm_feat)
        anchor_ids_list.append(obj_id.item())
        print(f"  - Calculated anchor for Object ID {obj_id.item()}")

    initial_centroids = np.array(anchor_vectors_list)
    NUM_CLUSTERS = len(initial_centroids)

    # 2. Create a Reliable Mapping *Before* Clustering
    kmeans_to_object_id_map = {i: obj_id for i, obj_id in enumerate(anchor_ids_list)}
    print("\nUsing pre-defined mapping based on initialization:")
    print(kmeans_to_object_id_map)

    # 3. Normalize the Full Feature Set and Run K-Means
    all_features = F.normalize(identity_encodings.cpu(), dim=1).numpy()
    print(f"\nRunning KMeans with {NUM_CLUSTERS} clusters (using anchor initialization)")
    kmeans = KMeans(
        n_clusters=NUM_CLUSTERS,
        init=initial_centroids,
        n_init=1,
        random_state=42
    ).fit(all_features)
    print("✅ K-Means clustering complete.")

    initial_labels = kmeans.labels_
    final_labels = np.copy(initial_labels)

    # 4. NEW: Identify and Re-assign Outliers to Background
    print(f"\nIdentifying outliers (top {100 - background_percentile:.1f}%) for background group...")
    # Get the distance from each point to every cluster's center
    distances_to_all_centroids = kmeans.transform(all_features)
    
    for i in range(NUM_CLUSTERS):
        # Get all points assigned to this cluster
        cluster_mask = (initial_labels == i)
        
        # Get the distance of each of these points to its *own* centroid
        distances_to_own_centroid = distances_to_all_centroids[cluster_mask, i]
        
        # If the cluster is empty, skip
        if len(distances_to_own_centroid) == 0: continue
            
        # Calculate the distance threshold for this specific cluster
        threshold = np.percentile(distances_to_own_centroid, background_percentile)
        
        # Find which points in this cluster are outliers (distance > threshold)
        point_indices_in_cluster = np.where(cluster_mask)[0]
        outlier_mask_in_cluster = (distances_to_own_centroid > threshold)
        outlier_indices = point_indices_in_cluster[outlier_mask_in_cluster]
        
        # Re-assign these outliers to the background (label -1)
        final_labels[outlier_indices] = -1
        print(f"  - Re-assigned {len(outlier_indices)} Gaussians from Cluster {i} to background.")

    # 5. Assemble Final Object Groups Using the New Labels
    object_groups = {obj_id: [] for obj_id in anchor_ids_list}
    object_groups["background"] = []

    for gaussian_idx, label in enumerate(final_labels):
        if label == -1:
            object_groups["background"].append(gaussian_idx)
        else:
            object_id = kmeans_to_object_id_map[label]
            object_groups[object_id].append(gaussian_idx)
    
    print("\n✅ Final Object Groups Assembled!")
    for name, group in object_groups.items():
        print(f"  - Group '{name}': {len(group)} Gaussians")

    # 6. Optional: t-SNE visualization with background points
    if tsne:
        # (Same t-SNE logic, but using `final_labels` to show background points)
        print("\nRunning t-SNE for 2D visualization...")
        tsne_model = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300, random_state=42)
        tsne_results = tsne_model.fit_transform(all_features)
        
        # Use a different color for the background points
        plt.figure(figsize=(10, 8))
        # Use a colormap and normalize labels to map colors correctly
        cmap = plt.cm.viridis
        unique_labels = np.unique(final_labels)
        norm = plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max())
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=final_labels, cmap='viridis', alpha=0.5, s=5)
        
        legend_handles = []
        legend_labels = []
        
        # Handle for Background
        if -1 in unique_labels:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=cmap(norm(-1)), markersize=8, label='Background'))
            legend_labels.append("Background")

        # Handles for Object Clusters
        for label_val, obj_id in sorted(kmeans_to_object_id_map.items()):
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=cmap(norm(label_val)), markersize=8, label=f'Cluster {label_val} (Obj ID {obj_id})'))
            legend_labels.append(f'Cluster {label_val} (Obj ID {obj_id})')
        
        plt.legend(handles=legend_handles, labels=legend_labels)
        plt.title('t-SNE Visualization of Identity Encodings (with Background)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)
        plt.savefig(f"{args.data_dir}/identity_maps/tsne_visualization_background.png")
        print(f"✅ Saved t-SNE plot")
        
    return object_groups


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Uniform Data Cropping Utility (v2.1)")
    parser.add_argument("--data_dir", required=False, help="Input folder")
    parser.add_argument("--data_factor", type=int, default=1, help="Downsampling factor")
    parser.add_argument("--identity_dim", type=int, default=16, help="Identity encoding dimension")
    args = parser.parse_args()
    # Update this path to your actual checkpoint
    CHECKPOINT_PATH = f"{args.data_dir}/ckpts/ckpt_29999_rank0.pt"
    
    # Load and inspect encodings
    encodings = load_and_inspect_identity_encodings(
        checkpoint_path=CHECKPOINT_PATH,
        num_gaussians_to_show=10,
        # save_to_file="identity_encodings.npy"  # Optional: save to file
    )

    identity_map = np.load(f"{args.data_dir}/identity_maps/identity_map_step30000.npy")
    # print("Identity map shape ", identity_map.shape)
    # print("Unique identity encodings in map ", np.unique(identity_map))
    instance_mask = np.load(f"{args.data_dir}/identity_maps/instance_mask_step30000.npy")
    print("Instance mask shape ", instance_mask.shape)
    print("Unique instance ids in mask ", np.unique(instance_mask))

    # identity_map = get_identity_map_from_checkpoint(ckpt_path=Path(CHECKPOINT_PATH),
    #                                  cfg=args,
    #                                  image_index=0)

    # dbscan_identity_encodings(encodings)
    object_groups = kmeans_identity_encodings_background(identity_map, instance_mask, encodings, tsne=True)

    cluster_groups_save_path = f"{args.data_dir}/identity_maps/cluster_groups.npy"
    save_dict = {str(k): v for k, v in object_groups.items()}
    np.savez_compressed(cluster_groups_save_path, **save_dict)
    print(f"✅ Object groups saved to {cluster_groups_save_path}")
    
    print(f"\nReturned tensor shape: {encodings.shape}")