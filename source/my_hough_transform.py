import numpy as np
from plot_line_on_image import plot_line_on_image

def my_hough_transform(binary_image: np.ndarray, d_rho: int, d_theta: float, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Calculates the hough transform, n most important lines and residual
    points given a binary image.
     Input:
        - binary_image: a numpy array consisting of 1s and 0s representing the edges of an image
        - d_rho: the rho step
        - d_theta: the theta step
        - n: the number of lines to be detected
     
     Output:
      - H: the accumulator of the transform
      - L: rho and theta parameters of the n most prominent lines
      - res: number of edges not belonging to any lines
    """
    
    # Calculate diagonal and dimension of accumulator
    N1, N2 = binary_image.shape    
    diagonal = np.hypot(N1, N2)
    R = int(diagonal / d_rho) + 1
    T = int(2 * np.pi / d_theta) + 1
    
    # Initialize Hough accumulator aand extract edges as vectors
    H = np.zeros((R, T))
    y_idxs, x_idxs = np.nonzero(binary_image)
    
    # Populate the Hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        for theta_index in range(T):
            theta = d_theta * theta_index
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_index = int(rho / d_rho)
            if 0 <= rho_index < R:
                H[rho_index, theta_index] += 1 
    
    # Find the n most powerful lines
    H_temp = H.copy()
    L = np.zeros((n,2))
    index = 0
    while index < n:
        # Find the indices of the max value in Htemp
        rho_index, theta_index = np.unravel_index(np.argmax(H_temp), H_temp.shape)
        
        # Calculate rho and theta for that indices
        max_r = d_rho * rho_index
        max_t = d_theta * theta_index      

        # Add line coefficients to L
        if not (np.any(L[:, 0] == max_r) and  np.any(L[:,1] == max_t)):
            L[index, 0] = max_r
            L[index, 1] = max_t
            index += 1

        # Clear a region close to the maximum to avoid duplicate lines
        r_min = np.maximum(int(rho_index - R / n), 0)
        r_max = np.minimum(int(rho_index + R / n), R)
        t_min = np.maximum(int(theta_index - T / n), 0)
        t_max = np.minimum(int(theta_index + T / n), T)
        H_temp[r_min:r_max, t_min:t_max] =0


    # Method of finding the maximums without filtering - less accurate results     
    #flat_indices = np.argpartition(H.flatten(), -n)[-n:]
    #r_indices, t_indices = np.unravel_index(flat_indices, H.shape)
    #L[:, 0] = r_indices * d_rho
    #L[:, 1] = t_indices * d_theta 
                 
    # Determine residual points by removing found lines from binary image
    res_img = plot_line_on_image(binary_image, L, (0, 0, 0))
    res = np.count_nonzero(res_img)
    return H, L, res