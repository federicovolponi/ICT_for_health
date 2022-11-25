import numpy as np

# Function to evaluate euclidean distance between two vectors
def euclidean_distance(p, q):
  dist = np.sqrt(np.sum(np.square(p - q)))
  return dist