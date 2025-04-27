import os
from PIL import Image
import re
import numpy as np

def combine_images(input_dir, output_dir, crop_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image files sorted by their file name
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.tif'))])
    numpy_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    nomask_files= sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.tif')) and "_nodata_mask_" in f])
    if not image_files:
        print("No image files found in the input directory.")
        return

    # Parse coordinates from file names and organize images into a grid
    grid = {}
    nomask_grid={}
    numpy_grid = {}
    max_x, max_y = 0, 0
    
    for file in image_files:
        # Extract x and y coordinates from the file name
        match = re.search(r'image_(\d+)_(\d+)', file)
        x = int(match.group(1))
        y = int(match.group(2))
        grid[(y,x)] = file
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    # Also process nomask files if available
    for file in nomask_files:
        match = re.search(r'mask_(\d+)_(\d+)', file)
        x = int(match.group(1))
        y = int(match.group(2))
        nomask_grid[(y,x)] = file
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    # Also process numpy files if available
    for file in numpy_files:
        match = re.search(r'image_(\d+)_(\d+)', file)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            numpy_grid[(y,x)] = file

    # Calculate the size of the combined image
    combined_width = max_x + crop_size
    combined_height = max_y + crop_size

    # Create a blank image to hold the combined result
    combined_image = Image.new('L', (combined_width, combined_height))
    combined_image_nodata = Image.new('L', (combined_width, combined_height))
    # Paste each image into the correct position
    for (y, x), file in grid.items():
        img = Image.open(os.path.join(input_dir, file))    
        combined_image.paste(img, (y, x))

    # Save the combined image
    combined_image.save(os.path.join(output_dir, "combined_image_dlv3+.tif"))
    print(f"Combined image saved to {output_dir}")
    
    for(y,x),file in nomask_grid.items():
        img = Image.open(os.path.join(input_dir, file))
        combined_image_nodata.paste(img, (y, x))
    
    combined_image_nodata.save(os.path.join(output_dir, "combined_image_nodata_dlv3+.tif"))
    print(f"Combined image with no data mask saved to {output_dir}")
    # If numpy files are available, combine them as well
    if numpy_grid:
        # Determine the dimensions of numpy arrays
        sample_file = list(numpy_grid.values())[0]
        sample_array = np.load(os.path.join(input_dir, sample_file))
        array_shape = sample_array.shape
        
        # Create a larger numpy array to hold the combined result
        if len(array_shape) == 2:
            combined_array = np.zeros((combined_height, combined_width))
        else:
            combined_array = np.zeros((array_shape[0], combined_height, combined_width))
        
        # Place each numpy array in the correct position
        for (y, x), file in numpy_grid.items():
            array = np.load(os.path.join(input_dir, file))
            if len(array_shape) == 2:
                combined_array[x:x+crop_size,y:y+crop_size] = array
            else:
                combined_array[:, y:y+crop_size, x:x+crop_size] = array
        
        # Save the combined numpy array
        np.save(os.path.join(output_dir, "combined_array_dlv3+.npy"), combined_array)
        print(f"Combined numpy array saved to {output_dir}")

# Example usage
input_dir = "E:\\imagedata\\2305-06\\clipped_images\\prediction_dlv3+"
output_dir = "E:\\imagedata\\2305-06"
combine_images(input_dir, output_dir)