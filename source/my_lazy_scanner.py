import numpy as np
from PIL import Image
import cv2
from image_preprocessing import image_preprocessing
from my_hough_transform import my_hough_transform
from my_harris import my_corner_harris, my_corner_peaks
from resembles_paralleogram import perpendicular_intersection, resembles_parallelogram
from my_img_rotation import my_img_rotation

if __name__ == "__main__":
    """
    Combines the developped methods (Hough, Harris, etc) to extract and
    rotate documents from an image
    """

    # Path to the image inside the subfolder - CHANGE IF NEEDED
    image_path = 'D:/Σχολή/8ο εξάμηνο/Ψηφιακή Επεξεργασία Εικόνας/hw2/images/'
    image_name = 'im1'
    
    # Load the image
    img = Image.open(image_path + image_name + '.jpg')   
    if img == None:
            print("Error while loading the image")
            exit(1)
    N1 = img.width
    N2 = img.height

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
    L[:,0] = L[:,0] * ratio # undo resizing
    
    # Execute pre-processing
    ratio = img.width // 600
    img_array = image_preprocessing(img, 5, ratio, 15)
    
	# Apply Harris corner detection and filter with a threshold
    harris_response = my_corner_harris(img_array, k=0.2, sigma=0.4)
    corner_locations = my_corner_peaks(harris_response, 0.4)
    corner_locations *= ratio

    # Find intersections of perpendicular lines
    intersections = perpendicular_intersection(L)
    
    # Keep harris peaks that are close to perpendicular intersections
    intersections_harris = []
    for x,y in intersections:
        for yc, xc in corner_locations:
            if abs(xc - x) <= 50 and abs(yc - y) <= 50:
                   intersections_harris.append([xc,yc])
    
    # Find for pairs of 4 points the documents, rotate and save them
    img_array = np.array(img)
    documents = resembles_parallelogram(np.array(intersections_harris))
    d_ind = 0
    for d in documents:
         # Calculate angle of rotation
         d = d[np.lexsort((d[:,1], d[:,0]))]
         dist = np.sqrt((d[0,0] - d[1,0])**2 + (d[0,1] - d[1,1])**2)
         dx = d[0,0] - d[1,0]
         angle = np.arcsin(dx/dist)
         
         # Take slice of initial image
         min_y = np.min(d[:,1])
         max_y = np.max(d[:,1])
         min_x = np.min(d[:,0])
         max_x = np.max(d[:,0])
         img_array_doc = img_array[min_x:max_x, min_y:max_y]

         # Rotate and save
         img_array_doc = my_img_rotation(img_array_doc, angle)
         img_doc = Image.fromarray(img_array_doc)
         img_doc.save(image_path + image_name + '_' + str(d_ind) + '.jpg')
         d_ind += 1

                            
    

    