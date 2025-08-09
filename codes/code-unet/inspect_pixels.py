# --- File: inspect_pixels.py ---

import cv2
import numpy as np
import argparse

# This global variable will store the image so the callback can access it
image = None

def mouse_callback(event, x, y, flags, param):
    """
    This function is called every time a mouse event happens in the window.
    We only care about the left-button click.
    """
    global image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the BGR value of the pixel at the clicked coordinates (y, x)
        bgr_pixel = image[y, x]
        
        # To convert a single pixel to HSV, we need to treat it as a 1x1 image.
        # This looks a bit strange, but it's how OpenCV works.
        hsv_pixel = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
        
        # Print the values to the console.
        print(f"--- Click at ({x}, {y}) ---")
        print(f"  BGR Values: Blue={bgr_pixel[0]}, Green={bgr_pixel[1]}, Red={bgr_pixel[2]}")
        print(f"  HSV Values: Hue={hsv_pixel[0]}, Saturation={hsv_pixel[1]}, Value={hsv_pixel[2]}")
        print("-" * 20)

def main():
    global image
    parser = argparse.ArgumentParser(description="Interactively inspect BGR and HSV values of an image's pixels.")
    parser.add_argument('image_path', type=str, help='Path to the image you want to inspect.')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"FATAL: Could not load image at {args.image_path}")
        return

    # Create a window to display the image
    window_name = 'Pixel Inspector (Press Q to quit)'
    cv2.namedWindow(window_name)
    
    # Tell OpenCV to call our 'mouse_callback' function for any mouse events in this window
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print(f"Image '{args.image_path}' loaded.")
    print("Click on different parts of the image (waveform, grid, background) to see their HSV values.")
    print("Press the 'q' key on your keyboard while the image window is active to quit.")

    # Main loop to keep the window open
    while True:
        cv2.imshow(window_name, image)
        # Wait for a key press. If it's 'q', break the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()