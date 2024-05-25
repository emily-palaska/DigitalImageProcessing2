from PIL import Image, ImageEnhance
import numpy as np
import cv2

def image_preprocessing(img:Image, contrast:float, ratio:int, blur:int) -> np.ndarray:
    """
    Pre-processes image by converting to grayscale, down-sizing,
    increasing contrast and applying Gaussian blur.
     Input:
        - img: an image in the form of PIL library
        - contrast: the multiplier for contrast enhancing
        - ratio: the ration for the down-sizing
        - blur: metric for Gaussian kernel
     Output:
        - img_array: the processed image in the form of numpy array
    """    
    
    # Resize image according to ratio
    w_new = img.width // ratio
    h_new = img.height // ratio
    img_resize = img.resize((w_new, h_new), Image.LANCZOS)
    
    # Make the image grayscale 
    img_grayscale = img_resize.convert('L')        
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(img_grayscale)
    img_grayscale = enhancer.enhance(contrast)
    
    # Convert the image to a NumPy array
    img_array = np.array(img_grayscale)
    
    # Apply Gaussian Blur
    if not blur == None:
        img_array = cv2.GaussianBlur(img_array, (blur, blur), 0)
    return img_array