from PIL import Image
from plot_line_on_image import plot_line_on_image
from my_hough_transform import my_hough_transform
from image_preprocessing import image_preprocessing
import numpy as np
import matplotlib.pyplot as plt
import cv2
    
if __name__ == "__main__":
    """
    Performs line detection applying the Hough Trnasform  to an imported
    image and saves both the accumulator figure and the image with
    spotted lines in the same directory
    """
    
    # Path to the image inside the subfolder - CHANGE IF NEEDED
    image_path = 'D:/Σχολή/8ο εξάμηνο/Ψηφιακή Επεξεργασία Εικόνας/hw2/images/'
    image_name = 'im2'
    
    # Load the image
    img = Image.open(image_path + image_name + '.jpg')   
    if img == None:
            print("Error while loading the image")
            exit(1)
    
    # Execute pre-processing
    ratio = img.width // 600
    img_array = image_preprocessing(img, 3, ratio, 5)
    
    # Apply Canny edge detector using OpenCV with a threshold
    binary_image = cv2.Canny(img_array, 100, 300)

    # Set parameters for Hough Transform
    d_rho = 1               # rho step
    d_theta = 0.005         # theta step in radians
    n = 20                  # number of lines to detect

    # Apply Hough Transformation function
    H, L, res = my_hough_transform(binary_image, d_rho, d_theta, n)
    
    # Save the accumulator transform with spotted maximums
    plt.imshow(H, cmap='gray')
    r = (L[:,0] / d_rho).astype(int)
    t = (L[:,1] / d_theta).astype(int)
    plt.scatter(t, r, marker='+', c='red', s=1)
    plt.title(f'Hough Accumulator for {image_name}.jpg')
    plt.savefig(image_path + image_name + '_transform.png')
    
    
    # Display the results
    L[:,0] = L[:,0] * ratio # undo resizing
    np.set_printoptions(suppress=True)
    print("Parameters of the n most powerful lines (rho, theta):\n", L)
    print("Number of points not belonging to the n detected lines:", res)
    
    # Save image with plotted lines in local directory
    img_array_with_lines = plot_line_on_image(np.array(img), L, (200, 0, 100))
    img_with_lines = Image.fromarray(img_array_with_lines)
    img_with_lines.save(image_path + image_name + '_lines.png')
    
    

