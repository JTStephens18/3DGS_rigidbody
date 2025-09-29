#!/usr/bin/env python3
"""
Uniform Data Cropping Utility (v2.0)

- Analyze a primary dataset (images + .npy arrays) to find the non-black content bounding boxes.
- Compute a uniform crop size (min width, min height across primary content bboxes).
- Apply that uniform crop (centered on each file's content center) to:
    * primary dataset -> save cropped files (same format) to primary_output_folder
    * secondary dataset -> 
        - if input is image: save cropped image to secondary_output_folder and also save a .npy
          of the cropped image to secondary_npy_output_folder
        - if input is .npy: save cropped .npy file to secondary_output_folder
- Skips unreadable files or files with no detectable content.
- If a centered crop would fall outside a file's bounds: skip with a clear warning.

Dependencies:
    pip install opencv-python numpy

Usage:
    Edit the folder variables in the main block OR run with optional args:
        python uniform_crop_utility.py --primary_input ./p_in --primary_output ./p_out ...
"""

import os
import sys
import glob
import argparse
from typing import Tuple, Optional, List
import numpy as np
import cv2

CONTENT_THRESHOLD = 1
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTS


def list_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    files.extend(glob.glob(os.path.join(folder, "*.npy")))
    files.sort()
    return files


def load_image(path: str) -> Optional[np.ndarray]:
    try:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def load_npy(path: str) -> Optional[np.ndarray]:
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None


def save_image(path: str, img: np.ndarray) -> bool:
    try:
        return cv2.imwrite(path, img)
    except Exception:
        return False


def save_npy(path: str, arr: np.ndarray) -> bool:
    try:
        np.save(path, arr)
        return True
    except Exception:
        return False


def detect_content_bbox_from_array(arr: np.ndarray, threshold: int = CONTENT_THRESHOLD) -> Optional[Tuple[int, int, int, int]]:
    if arr is None:
        return None
    if arr.ndim == 2:
        mask = arr
    elif arr.ndim == 3:
        mask = np.max(arr, axis=2)
    else:
        return None
    if np.issubdtype(mask.dtype, np.floating):
        m = np.abs(mask) > (threshold / 255.0)
    else:
        m = mask > threshold
    coords = np.argwhere(m)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return int(x_min), int(y_min), int(width), int(height)


# def crop_centered_on_bbox(arr: np.ndarray, bbox: Tuple[int, int, int, int], crop_w: int, crop_h: int) -> Optional[np.ndarray]:
#     if arr is None:
#         return None
#     H, W = arr.shape[:2]
#     x_min, y_min, w, h = bbox
#     cx = x_min + w / 2.0
#     cy = y_min + h / 2.0
#     x0 = int(round(cx - crop_w / 2.0))
#     x1 = x0 + crop_w
#     y0 = int(round(cy - crop_h / 2.0))
#     y1 = y0 + crop_h
#     if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
#         return None
#     return arr[y0:y1, x0:x1].copy() if arr.ndim == 2 else arr[y0:y1, x0:x1, ...].copy()

def crop_centered_on_bbox(arr, bbox, crop_w, crop_h):
    H, W = arr.shape[:2]
    x_min, y_min, w, h = bbox
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0

    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))

    # Clamp to image boundaries
    x0 = max(0, min(W - crop_w, x0))
    y0 = max(0, min(H - crop_h, y0))

    x1 = x0 + crop_w
    y1 = y0 + crop_h

    return arr[y0:y1, x0:x1].copy()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def analyze_primary_for_uniform_size(primary_input_folder: str) -> Tuple[int, int]:
    files = list_files(primary_input_folder)
    if not files:
        raise SystemExit(f"No files in primary input: {primary_input_folder}")
    bboxes = []
    for path in files:
        arr = load_image(path) if is_image_file(path) else load_npy(path)
        if arr is None:
            continue
        bbox = detect_content_bbox_from_array(arr)
        if bbox is not None:
            bboxes.append(bbox)
    if not bboxes:
        raise SystemExit("No detectable content in primary dataset.")
    widths = [w for (_, _, w, _) in bboxes]
    heights = [h for (_, _, _, h) in bboxes]
    return min(widths), min(heights)


def get_size_from_reference(reference_file: str) -> Tuple[int, int]:
    arr = load_image(reference_file) if is_image_file(reference_file) else load_npy(reference_file)
    if arr is None:
        raise SystemExit(f"Failed to load reference crop file: {reference_file}")
    h, w = arr.shape[:2]
    print(f"Using reference crop file size: width={w}, height={h}")
    return w, h


def process_dataset(input_folder: str, output_folder: str, npy_output_folder_for_images: Optional[str],
                    target_w: int, target_h: int, is_secondary: bool):
    files = list_files(input_folder)
    if not files:
        print(f"Warning: no files in {input_folder}")
        return
    ensure_dir(output_folder)
    if is_secondary and npy_output_folder_for_images:
        ensure_dir(npy_output_folder_for_images)
    for path in files:
        basename = os.path.basename(path)
        arr = load_image(path) if is_image_file(path) else load_npy(path)
        if arr is None:
            print(f"  Skipping unreadable: {basename}")
            continue
        bbox = detect_content_bbox_from_array(arr)
        if bbox is None:
            print(f"  Skipping no-content: {basename}")
            continue
        cropped = crop_centered_on_bbox(arr, bbox, target_w, target_h)
        if cropped is None:
            print(f"  Skipping out-of-bounds crop: {basename}")
            continue
        if is_image_file(path):
            out_img = os.path.join(output_folder, basename)
            save_image(out_img, cropped)
            if is_secondary and npy_output_folder_for_images:
                npy_path = os.path.join(npy_output_folder_for_images, os.path.splitext(basename)[0] + ".npy")
                save_npy(npy_path, cropped)
        else:
            out_npy = os.path.join(output_folder, os.path.splitext(basename)[0] + ".npy")
            save_npy(out_npy, cropped)


def main(primary_input_folder: str,
         primary_output_folder: str,
         secondary_input_folder: Optional[str],
         secondary_output_folder: Optional[str],
         secondary_npy_output_folder: Optional[str],
         reference_crop_file: Optional[str]):
    if reference_crop_file:
        target_w, target_h = get_size_from_reference(reference_crop_file)
    else:
        target_w, target_h = analyze_primary_for_uniform_size(primary_input_folder)
        print(f"Calculated uniform crop size from primary dataset: width={target_w}, height={target_h}")
    print("\nCropping primary dataset...")
    process_dataset(primary_input_folder, primary_output_folder, None, target_w, target_h, False)
    if secondary_input_folder:
        print("\nCropping secondary dataset...")
        process_dataset(secondary_input_folder, secondary_output_folder, secondary_npy_output_folder,
                        target_w, target_h, True)
    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uniform Data Cropping Utility (v2.1)")
    parser.add_argument("--primary_input_folder", required=False, help="Primary input folder")
    parser.add_argument("--primary_output_folder", required=False, help="Primary output folder")
    parser.add_argument("--secondary_input_folder", default=None, help="Secondary input folder")
    parser.add_argument("--secondary_output_folder", default=None, help="Secondary output folder")
    parser.add_argument("--secondary_npy_output_folder", default=None, help="Secondary npy output folder")
    parser.add_argument("--reference_crop_file", default=None, help="Use this file's size as crop dimensions")
    args = parser.parse_args()

    # --------- Configure paths here (edit these variables directly if you prefer) ----------
    # You can also pass them via command-line args (--primary_input_folder ...)
    primary_input_folder = args.primary_input_folder or "/home/te/projects/splat_rigid_body/data/images_og"
    primary_output_folder = args.primary_output_folder or "/home/te/projects/splat_rigid_body/data/images"
    secondary_input_folder = args.secondary_input_folder or "/home/te/projects/splat_rigid_body/data/masks/instance_ids_uncropped"  # set to None or "" to skip secondary processing
    secondary_output_folder = args.secondary_output_folder or "/home/te/projects/splat_rigid_body/data/masks/instance_ids"
    secondary_npy_output_folder = args.secondary_npy_output_folder or "/home/te/projects/splat_rigid_body/data/masks/instance_ids_npy"
    reference_crop_file = args.reference_crop_file or "/home/te/project/splat_rigid_body/data/cropped_image.jpg"  # set to None to auto-calculate from primary dataset
    # --------------------------------------------------------------------------------------

    # If user provided empty string for secondary, treat as None
    if secondary_input_folder == "":
        secondary_input_folder = None

    print("Uniform Data Cropping Utility (v2.0)")
    print("Primary input:", primary_input_folder)
    print("Primary output:", primary_output_folder)
    if secondary_input_folder:
        print("Secondary input:", secondary_input_folder)
        print("Secondary output:", secondary_output_folder)
        print("Secondary npy output:", secondary_npy_output_folder)
    else:
        print("Secondary dataset: NONE (skipping)")

    try:
        main(primary_input_folder, primary_output_folder,
             secondary_input_folder, secondary_output_folder, secondary_npy_output_folder, reference_crop_file)
    except SystemExit as e:
        print(f"Fatal: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unhandled exception: {e}")
        sys.exit(1)
