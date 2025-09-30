from PIL import Image
import numpy as np
import os
from pathlib import Path
import argparse

def create_instance_id_map(image_path: str) -> np.ndarray:
    """
    Loads an RGB image, maps specific colors to integer instance IDs,
    and returns a 2D NumPy array representing the instance map.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
    except Exception as e:
        print(f"Error opening or processing image {image_path}: {e}")
        return None

    # --- IMPORTANT ---
    # Define the exact RGB colors for your objects and their corresponding IDs.
    # Use an image editor's eyedropper tool to get these values.
    color_to_id = {
        (0, 0, 0): 0,          # Black background
        (254, 127, 14): 1,      # Orange object
        (31, 120, 180): 2,      # Blue object
        (174, 198, 232): 3,    # Light Blue object
        # Add a fourth color if needed, e.g., (255, 0, 0): 4 for a red object
    }

    h, w, _ = img_array.shape
    instance_id_map = np.zeros((h, w), dtype=np.uint8) # Use uint8 for efficiency

    # Iterate through the color map and create the instance ID map
    for color, instance_id in color_to_id.items():
        matches = np.all(img_array == np.array(color), axis=-1)
        instance_id_map[matches] = instance_id

    print("Instance id map shape ", instance_id_map.shape)
    print("Unique instance ids ", np.unique(instance_id_map))
    return instance_id_map

def process_folder(input_folder: str, output_folder: str):
    """
    Processes all images in an input folder and saves the resulting
    instance ID maps as .npy files in the output folder.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output will be saved to: {os.path.abspath(output_folder)}")

    supported_extensions = ['.jpg', '.jpeg', '.png']

    for filename in sorted(os.listdir(input_folder)):
        file_path = Path(input_folder) / filename
        
        # Check if it's a file with a supported image extension
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            print(f"Processing: {filename}...")
            
            # Generate the instance ID map from the image
            instance_map = create_instance_id_map(str(file_path))
            
            if instance_map is not None:
                # Construct the output path with the new .npy extension
                output_filename = file_path.stem + '.npy'
                output_path = Path(output_folder) / output_filename
                
                # Save the NumPy array
                np.save(output_path, instance_map)
                print(f"  -> Saved instance map to {output_path}")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert a folder of instance ID images to NumPy arrays (.npy)."
    )
    parser.add_argument(
        "--input_folder", 
        type=str, 
        help="Path to the folder containing the instance ID images."
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        help="Path to the folder where the .npy files will be saved."
    )
    
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.output_folder)
    print("\n✅ All images processed successfully!")

if __name__ == "__main__":
    main()