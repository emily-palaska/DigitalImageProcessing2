import numpy as np
from scipy.spatial.distance import pdist
from plot_line_on_image import find_intersection

def perpendicular_intersection(L: np.ndarray) -> np.ndarray:
    """
    Retuns the intersections of (approximately) perpendicualr line pairs
     Input:
        - L: an nx2 array of n lines with the distance from the upper
          left corner to the line (rho) and the angle in radians between
          the x-axis and the line (theta)
     Output:
        - numpy array of perpendicular line intersections
    """
    n, _ = L.shape
    perp_intersec = []
    tolerance = 100 # in pixels

    # Extract all interesction with theta difference around pi/2 or 3pi/2
    for i in range(n):
         for j in range(n):
               theta_diff = abs(L[i,1] - L[j,1])
               if abs(theta_diff - np.pi/2) <= 0.3 or abs(theta_diff - 3 * np.pi / 2) <= 0.3:
                   line1 = (np.cos(L[i,1]), np.sin(L[i,1]), L[i,0])
                   line2 = (np.cos(L[j,1]), np.sin(L[j,1]), L[j,0])
                    
                   temp = find_intersection(line1, line2)
                   if temp != None:      
                       x, y = temp
                       if not [x,y] in perp_intersec:
                           perp_intersec.append([x,y])
     
    # Filter duplicates out
    for i in range(len(perp_intersec) - 1):
        if perp_intersec == None:
             continue
        for j in range(i + 1, len(perp_intersec)):
           if perp_intersec[j] == None:
                continue
           x1, y1 = perp_intersec[i]
           x2, y2 = perp_intersec[j]
           dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
           if dist <= tolerance:
               perp_intersec[i] = None
               break
    
    return np.array([p for p in perp_intersec if p != None])

def resembles_parallelogram(points:list) -> np.ndarray:
    """
    Given a set of points, checks every possible combination of 4 points
    to find a parallelogram-resembling shape
     Input:
        - points: a list of points
     Output:
        - parallelograms: a numpy array of sets of 4 points that ressemble
          a parallelogram 
    """
    # Find all the combinations of points
    comb_list = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    comb_list.append((points[i], points[j], points[k], points[l]))
    combinations = np.array(comb_list)
    
    # Iterate every combination
    parallelograms = []
    for c in combinations:
        # Find the distance of each pair of points
        pairwise_distances = pdist(c, 'euclidean')
        
        # Sort the distances and define a tolerance
        pairwise_distances = np.sort(pairwise_distances)
        tolerance = pairwise_distances[3] * 0.05
        
        # Calculate the difference of the distances and check against tolerance
        diff1 = pairwise_distances[0] - pairwise_distances[1]
        diff2 = pairwise_distances[2] - pairwise_distances[3]
        
        if abs(diff1) <= tolerance and abs(diff2) <= tolerance:
            parallelograms.append(c)
    return np.array(parallelograms)     