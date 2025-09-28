# 3DGS Rigid Body Dynamics

This project demonstrates how to add dynamic rigid body physics to a static 3D Gaussian Splatting (3DGS) scene. The core idea is to segment a trained 3DGS scene into distinct objects (e.g., dominoes), compute their physical properties, and then use a physics engine like **NVIDIA Warp** to simulate their interactions. The resulting rigid body transformations are then applied back to the Gaussians to render dynamic animations.



## 📋 Core Workflow

The end-to-end pipeline follows these steps:

1.  **Train & Segment:** Start with a high-quality static 3DGS scene. Add a learnable "identity vector" to each Gaussian and train a small network to cluster Gaussians belonging to the same object, supervised by 2D instance masks.
2.  **Extract Rigid Bodies:** For each segmented cluster, calculate its rigid body parameters: mass, center of mass (COM), inertia tensor, and a collision shape (e.g., an oriented bounding box).
3.  **Simulate with Warp:** Set up a physics simulation in NVIDIA Warp, creating one rigid body for each object cluster. Apply forces (like an initial push) to start the simulation.
4.  **Transform Gaussians:** At each physics step, retrieve the updated position and orientation for each rigid body from Warp. Use these poses to transform all the corresponding Gaussians in the cluster into their new world-space positions. This is done efficiently on the GPU.
5.  **Render:** Use a 3DGS renderer to render the transformed Gaussians for each frame, creating the final animation.

---

## 🚀 Getting Started

### Prerequisites

* **Software**: Python, PyTorch, NVIDIA Warp
* **Optional Libraries**: Open3D, NumPy, OpenCV
* **Input Data**:
    * A pre-trained static 3D Gaussian Splatting scene.
    * A clustering method that assigns an object ID to each Gaussian.

### Implementation Plan

1.  **Data Capture & Prep**: Photograph your scene (e.g., a domino rally) from multiple viewpoints.
2.  **COLMAP Processing**: Run COLMAP on the images to get camera poses.
3.  **Base 3DGS Training**: Train a standard, high-quality 3DGS of the static scene.
4.  **Segmentation**:
    * Generate 2D instance masks for a subset of your training images using a tool like SAM.
    * Modify your 3DGS code to include a learnable identity vector for each Gaussian.
    * Train the identity vectors using a loss function that compares rendered object IDs against the 2D masks.
    * Run a clustering algorithm (e.g., K-Means) on the final identity vectors to get the object groups.
5.  **Physics Preparation**:
    * For each object group, compute its center of mass and inertia tensor. See the detailed steps below.
6.  **Integration & Simulation**:
    * Write a simulation loop that uses Warp to update each object's state.
    * Apply the calculated transformations to the underlying Gaussians.
    * Apply an initial "push" to the first object and run the simulation, rendering each frame.