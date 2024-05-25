import numpy as np
from my_harris import my_corner_harris, my_corner_peaks
from image_preprocessing import image_preprocessing
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Performs corner detection applying the Harris algorithm to an
    imported image and saves the image with the marked corners
    """
    # Path to the image inside the subfolder - CHANGE IF NEEDED
    image_path = 'D:/Σχολή/8ο εξάμηνο/Ψηφιακή Επεξεργασία Εικόνας/hw2/images/'
    image_name = 'im2'
    
    # Load the image
    img = Image.open(image_path + image_name + '.jpg')

    if img == None:
            print("Error while loading the image")
            exit(1)
    
    img_array = np.array(img.convert('L')) / 255
    
    # Execute pre-processing
    ratio = img.width // 600
    img_array = image_preprocessing(img, 5, ratio, 15)
    
    # Apply Harris corner detection and filter with a threshold
    harris_response = my_corner_harris(img_array, k=0.2, sigma=0.4)
    corner_locations = my_corner_peaks(harris_response, 0.4)

    # Initialize draw object
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 50)

    # Save response locally
    plt.imshow(harris_response)
    plt.title(f'Harris Response for {image_name}.jpg')
    plt.savefig(image_path + image_name + '_response.jpg')

    # Mark the corners in the original image and save locally
    for marker in corner_locations:
         y = ratio * marker[0]
         x = ratio * marker[1]
         draw.text((x, y), "s", fill="red", font=font)
    img.save(image_path + image_name + '_harris.jpg')