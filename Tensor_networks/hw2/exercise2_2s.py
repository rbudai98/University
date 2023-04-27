import numpy as np

def spectral(matrix):
	# Your code goes here
	eig_val, eig_vec = np.linalg.eig(matrix)
	return eig_val, eig_vec
	
def singular(matrix):
	# Your code goes here
    s,u,vdagger = np.linalg.svd(matrix)
    return u, s, vdagger


matrix = [[0, 3/5, 4/5], [-3/5, 0, 0], [-4/5, 0, 0]]
print (spectral(matrix))
print(singular(matrix))


