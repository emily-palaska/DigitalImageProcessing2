import numpy as np
from scipy.ndimage import convolve

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """ Generate a (2k+1)x(2k+1) Gaussian kernel with mean=0 and standard deviation=sigma """
    k = size // 2
    ax = np.linspace(-k, k, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def my_corner_harris(image:np.ndarray, k:float, sigma:float) -> np.ndarray:
    """
    Calculates the Harris response of an image, meaning the probability
    of a corner in each pixel
     Input:
        - image: a numpy array representing an image
        - k: the k multiplier of the traces
        - sigma: the deviation of the gaussian kernel
     Output:
        - R: the harris response, metric of corner importance 
    """
    
    N1, N2 = image.shape   

    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Convolve the image with Sobel operators
    I1 = convolve(image, sobel_x)
    I2 = convolve(image, sobel_y)

    # Compute products of derivatives
    I1_2 = I1 ** 2
    I2_2 = I2 ** 2
    I1_I2 = I1 * I2

    # Create Gaussian kernel
    size = int(6 * sigma) + 1  # size of the kernel (typically 6*sigma is a good choice)
    gaussian = gaussian_kernel(size, sigma)

    # Convolve gradient products with Gaussian kernel
    S1 = convolve(I1_2, gaussian)
    S2 = convolve(I2_2, gaussian)
    S3 = convolve(I1_I2, gaussian)
    
    # Compute corner response at all pixels
    R = np.zeros((N1, N2))
    for p1 in range(N1):
        for p2 in range(N2):
            M = np.array([[S1[p1, p2], S3[p1, p2]],
                          [S3[p1, p2], S2[p1, p2]]])
            R[p1, p2] = np.linalg.det(M) - k * (np.trace(M) ** 2)
    return R

def my_corner_peaks(harris_response: np.ndarray, rel_threshold: float) -> np.ndarray:
    """
    Applies a threshold to the harris response and determines the corner coordinates
     Input:
        - harris_response: the metric of corner probability in each pixel of an image
        - rel_threshold: threshold multiplier for the maximum value of the reponse 
     Output:
        - corner_locations: the cartesian coordinates of the determined corners 
    """ 
    # Calculate the threshold as a percentage of the maximum value
    thres = np.max(harris_response) * rel_threshold
    # Filter the harris response by the calculated threshold
    corner_locations = np.array(np.where(harris_response >= thres))
    return np.transpose(corner_locations)