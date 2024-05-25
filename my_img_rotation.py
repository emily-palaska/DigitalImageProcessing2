import numpy as np

def my_img_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    """"
    Rotates an image counter clockwise by an angle

     Input:
        - img: a numpy array representing an image
        - angle: the rotation angle in radians
     Output:
        - rotated_img: a numpy array representing the rotated image
    """ 
    # Get the dimensions of the image
    if img.ndim == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    
    # Calculate the center of the image
    center_x, center_y = w / 2, h / 2
    
    # Calculate the new image dimensions
    new_w = int(abs(h * np.sin(angle)) + abs(w * np.cos(angle)))
    new_h = int(abs(h * np.cos(angle)) + abs(w * np.sin(angle)))
    
    # Create an empty image with the new dimensions
    rotated_img = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
    mask = np.zeros((new_h, new_w), dtype=bool)

    # Calculate the translation needed to keep the image centered
    tx = (new_w - w) / 2
    ty = (new_h - h) / 2
    
    # Create the rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    for y in range(h):
        for x in range(w):
            # Apply the rotation transformation
            new_x = int((x - center_x) * cos_a - (y - center_y) * sin_a + center_x + tx)
            new_y = int((x - center_x) * sin_a + (y - center_y) * cos_a + center_y + ty)
            
            if 0 <= new_x < new_w and 0 <= new_y < new_h:
                rotated_img[new_y, new_x] = img[y, x]
                mask[new_y, new_x] = True
    
    # Interpolate empty pixels
    for y in range(1, new_h - 1):
        for x in range(1, new_w - 1):
            if not mask[y, x]:
                # Average the neighboring pixels
                neighbors = []
                if mask[y - 1, x]: neighbors.append(rotated_img[y - 1, x])
                if mask[y + 1, x]: neighbors.append(rotated_img[y + 1, x])
                if mask[y, x - 1]: neighbors.append(rotated_img[y, x - 1])
                if mask[y, x + 1]: neighbors.append(rotated_img[y, x + 1])
                
                if neighbors:
                    rotated_img[y, x] = np.mean(neighbors, axis=0)

    return rotated_img