import numpy as np
from PIL import Image
from my_img_rotation import my_img_rotation

if __name__ == "__main__":
    """
    Rotates an image first by 54 and then by 213 degrees
    """
    # Path to the image inside the subfolder - CHANGE IF NEEDED
    image_path = 'D:/Σχολή/8ο εξάμηνο/Ψηφιακή Επεξεργασία Εικόνας/hw2/images/'
    image_name = 'im2'
    
    # Load the image
    img = Image.open(image_path + image_name + '.jpg')

    if img == None:
            print("Error while loading the image")
            exit(1)

    img_array_rot1 = my_img_rotation(np.array(img), 54 * np.pi / 180)
    img_rot1 = Image.fromarray(img_array_rot1)
    img_rot1.save(image_path + image_name + '_rot1.jpg')
    
    img_array_rot2 = my_img_rotation(np.array(img), 213 * np.pi / 180)
    img_rot2 = Image.fromarray(img_array_rot2)
    img_rot2.save(image_path + image_name + '_rot2.jpg')
