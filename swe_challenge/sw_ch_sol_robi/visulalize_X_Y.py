import plotly.graph_objs as go
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

x_dim = 50
y_dim = 50

def load_np(name):
	B_load = np.load(name)
	B = np.zeros((x_dim, y_dim, 3))

	for i in range(x_dim):
		for j in range(y_dim):
			for k in range(3):
				B[i,j,k] = B_load[(i*x_dim + j)*3 + k]
	return B


# Load data
B = load_np('processed_data/B_Lorentzian.npy')
B_rough = np.load('processed_data/B_rough.npy')

B_mag = load_np('processed_data/B_mag_Lorentzian.npy')
B_mag_rough = np.load('processed_data/B_mag_rough.npy')


B_x = B[:,:,0]
B_y = B[:,:,1]
B_z = B[:,:,2]

plt.quiver(np.arange(0, x_dim), np.arange(0, y_dim), B_x, B_z, alpha=.5, scale=10, width=0.001)

plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.show()

