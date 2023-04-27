import numpy as np

def compress(matrix, max_sv):
	
	# TODO: Compute the SVD-compressed approximation of `matrix`
	# Hint: `u[:, :χ]` selects the leading `χ` columns of the matrix `u`.
    s,u,vdagger = np.linalg.svd(matrix)
    print(u)
    u = u[ :max_sv]
    s = s[:,:max_sv]
    vdagger=vdagger[:max_sv,:]
    matrix_trunc = np.matmul(s, np.diag(u))
    matrix_trunc = np.matmul(matrix_trunc, vdagger)
    return matrix_trunc

matrix = [[ 1.02650,   0.92840,   0.54947,   0.98317,   0.71226,   0.55847],
[ 0.92889,   0.89021,   0.49605,   0.93776,   0.62066,   0.52473],
[ 0.56184,   0.49148,   0.80378,   0.68346,   1.02731,   0.64579],
[ 0.98074,   0.93973,   0.69170,   1.03432,   0.87043,   0.66371],
[ 0.69890,   0.62694,   1.02294,   0.87822,   1.29713,   0.82905],
[ 0.56636,   0.51884,   0.65096,   0.66109,   0.82531,   0.55098]]

compress(matrix, 3)