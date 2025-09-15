import cv2
import numpy as np
import sys

# get the image path from the command line argument
if len(sys.argv) < 2:
    print("Usage: python check_mask.py <path_to_mask_image>")
    sys.exit(1)
    
mask_path = sys.argv[1]

# load the mask in grayscale mode. this is crucial.
mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask_image is None:
    print(f"Error: Could not load image at {mask_path}")
else:
    # find all unique pixel values in the image
    unique_values = np.unique(mask_image)
    print(f"Unique pixel values in '{mask_path}': {unique_values}")

    # check for expected values
    if np.array_equal(unique_values, [0, 1]):
        print("Result: OK! Mask contains correct values for nnU-Net (0 and 1).")
    elif np.array_equal(unique_values, [0]):
         print("Result: WARNING! Mask only contains background (0). No waveform was detected.")
    else:
        print("Result: ERROR! Mask contains unexpected values.")