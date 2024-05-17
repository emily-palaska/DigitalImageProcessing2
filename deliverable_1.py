from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def my_hough_transform(binary_image: np.ndarray, d_rho: int, d_theta: float, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    # Extract dimensions
    N1, N2 = binary_image.shape    
    diagonal = np.hypot(N1, N2)

    R = int(diagonal / d_rho) + 1
    T = int(np.pi / (2 * d_theta)) + 1
    
    print('T is ', T, ' and R is ', R)
    # Initialize Hough accumulator aand extract edges as vectors
    H = np.zeros((T, R))
    x_idxs, y_idxs = np.nonzero(binary_image)
    
    # Populate the Hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        for theta_index in range(T):
            theta = d_theta * theta_index
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            if 0 <= rho < diagonal:
                rho_index = int(rho / d_rho)
                H[theta_index, rho_index] += 1
    
    # Find the n most powerful lines
    L = np.partition(H, -n)[-n:]        
    
    # Step 4: Determine residual points
    res = 0

    
    return H, L, res

if __name__ == "__main__":
    # Path to the image inside the subfolder - CHANGE IF NEEDED
    image_path = './images/im2.jpg'

    # Load the image and turn to grayscale numpy array
    img = Image.open(image_path).convert('L')    

    # Handle loading error
    if img == None:
            print("Error while loading the image")
            exit(1)
    
    # Convert the image to a NumPy uint8 array
    img_array = np.array(img)
    #print('Original image size: ', img_array.shape)
    
    # Apply Canny edge detector using OpenCV with a threshold
    #binary_image = cv2.Canny(img_array, 300, 400)
    #print('Number of edges: ', np.count_nonzero(binary_image))
    # Calculate amount and percentage of edges 
    #nnz = np.count_nonzero(edges)
    #p = np.round(nnz / (img_array.shape[0] * img_array.shape[1]), decimals=4)
    #print(f'Amount of non zeros in edges: {nnz}')
    #print(f'In percentage: {100 * p}%')
    
    # Save edges in local directory
    #edges_img = Image.fromarray(binary_image)
    #edges_img.save('./images/edges2.png')
    
    N1 = 100
    N2 = 100
    binary_image = np.zeros((N1, N2))
    for _ in range(100):
        binary_image[random.randint(0, N1 - 1), random.randint(0, N2 - 1)] = 1

    # Vizualize data
    x_ids, y_ids = np.nonzero(binary_image)
    plt.scatter(x_ids, y_ids)
    plt.title("Edge points")
    plt.show()    
    
    # Parameters for Hough Transform
    d_rho = 2
    d_theta = 0.05  # 1 degree in radians
    n = 5  # Number of lines to detect

    # Apply Hough Transformation function
    H, L, res = my_hough_transform(binary_image, d_rho, d_theta, n)
    
    # Display the results
    #print("Hough Transform Matrix H:\n", H)
    print("Parameters of the n most powerful lines (rho, theta):\n", L)
    #print("Number of points not belonging to the n detected lines:\n", res)
    
    # Visualize the accumulator
    plt.imshow(H, cmap='gray')
    plt.show()

