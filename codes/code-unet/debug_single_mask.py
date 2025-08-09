# --- File: debug_single_mask.py (CORRECTED & UPGRADED) ---

import cv2
import numpy as np
import argparse

# --- Using the improved function from your create_masks.py script ---
def create_segmentation_mask(image, v_max, s_min, close_kernel_size, open_kernel_size):
    """
    Creates a segmentation mask using a robust CLOSE then OPEN morphological strategy.
    Returns all intermediate steps for debugging.
    """
    if image is None: return None, None, None, None
    
    # 1. Thresholding to get the raw, disconnected signal
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, s_min, 0])
    upper_bound = np.array([179, 255, v_max])
    raw_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # 2. MORPH_CLOSE: Connect the dots into a solid line.
    # A rectangular kernel is excellent for connecting horizontal lines.
    close_kernel = np.ones((1, close_kernel_size), np.uint8)
    connected_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)

    # 3. MORPH_OPEN: Clean noise from the now-solid line.
    # A square kernel is good for removing small, non-line-shaped noise.
    open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
    cleaned_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_OPEN, open_kernel)
    
    return raw_mask, connected_mask, cleaned_mask

def main():
    parser = argparse.ArgumentParser(description="Visually debug the mask creation for a single image.")
    parser.add_argument('--image', type=str, required=True, help='Path to a single cropped lead image.')
    parser.add_argument('--v-max', type=int, default=120, help='Brightness threshold (Value in HSV).')
    parser.add_argument('--s-min', type=int, default=50, help='Saturation threshold (in HSV).')
    # --- New arguments for the better strategy ---
    parser.add_argument('--close-kernel', type=int, default=20, help='Kernel size for MORPH_CLOSE to connect dots.')
    parser.add_argument('--open-kernel', type=int, default=3, help='Kernel size for MORPH_OPEN to clean noise.')
    args = parser.parse_args()

    # --- Load Image ---
    original_image = cv2.imread(args.image)
    if original_image is None:
        print(f"FATAL ERROR: Could not load image at {args.image}")
        return

    # --- Process and Get Intermediate Steps ---
    raw_mask, connected_mask, final_mask = create_segmentation_mask(
        image=original_image,
        v_max=args.v_max,
        s_min=args.s_min,
        close_kernel_size=args.close_kernel,
        open_kernel_size=args.open_kernel
    )
    
    # --- Visualize Everything ---
    print(f"Raw Mask unique values:      {np.unique(raw_mask)}")
    print(f"Connected Mask unique values: {np.unique(connected_mask)}")
    print(f"Final Mask unique values:      {np.unique(final_mask)}")
    
    cv2.imwrite('1_original_image.png', original_image)
    cv2.imwrite('2_raw_mask.png', raw_mask)
    # This is the new, important image to check!
    cv2.imwrite('3_connected_mask.png', connected_mask) 
    cv2.imwrite('4_final_mask.png', final_mask)
    
    print("\nCheck the output images: '1_original_image.png', '2_raw_mask.png', '3_connected_mask.png', '4_final_mask.png'")

if __name__ == '__main__':
    main()