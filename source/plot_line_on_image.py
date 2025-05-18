import numpy as np
import cv2

def find_intersection(line1: tuple, line2:tuple):
    """
    Finds the intersection of two lines (if any) with format ax+by=c
     Input:
        - line1: tuple of the coefficients a,b,c of first line
        - line2: tuple of the coefficients a,b,c of second line
     Output:
        - None if intersection is not found OR
        - Tuple containing the integer part of the intersection coordinates
    """
    
    # Extract line coefficients
    a1, b1, c1, = line1
    a2, b2, c2 = line2
    
    # Calculate determinant 
    determinant = a1 * b2 - a2 * b1
    
    if determinant == 0:
        # If determinant is zero, lines are parallel, no intersection
        return None
    
    # Find the interseciton point
    x = (c1 * b2 - c2 * b1) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    if abs(x) > 10**17 or abs(y) > 10 ** 17:
        # If one of the two approximates infinity, no intersection
        return None
    return (int(x), int(y))


def plot_line_on_image(image:np.ndarray, L:np.ndarray, color:tuple) -> np.ndarray:
    """
    Plots some lines on the image based on the given rho and theta.

     Input:
        - image: the input image on which the line is to be plotted
        - L: an nx2 array of n lines with the distance from the upper
          left corner to the line (rho) and the angle in radians between
          the x-axis and the line (theta)
        - color: the color of the plotted line as an RGB tuple

     Output:
        - img_with_lines: the image with the plotted lines as a numpy array
    """
    
    # Extract image dimensions
    if image.ndim == 3:
        N1, N2, _ = image.shape
    else:
        N1, N2 = image.shape
    
    # Create array of the limiting lines, meaning edges of image
    limits = [(0, 1, 0), (0, 1, N1 - 1), (1, 0, 0), (1, 0, N2 - 1)]
    
    # Initialize resulting image
    image_with_lines = image.copy()
    
    # Iterate all lines
    for rho, theta in L:    
        # Calculate line coefficients
        line = (np.cos(theta), np.sin(theta), rho)

        # Find intersections with every edge
        intersections = []
        for i in range(4):
            temp_point = find_intersection(line, limits[i])
            if not temp_point == None:
                intersections.append(temp_point)

        # Draw all combinations of intersections as lines
        for i in range(len(intersections)):
            for j in range(i):
                image_with_lines = cv2.line(image_with_lines, intersections[i], intersections[j], color, 2)
    return image_with_lines