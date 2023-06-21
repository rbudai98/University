import numpy as np
import matplotlib.pyplot as plt

def W(a):
    return np.array([[a, 1j*np.sqrt(1-a**2)],[1j*np.sqrt(1-a**2), a]])

def exponential(phi):
    return np.array([[np.exp(1j*phi), 0],[0, np.exp(-1j*phi)]])

def U(a, phi):
    prod = exponential(phi[0])
    for i in phi[1:]:
        prod = prod*W(a)*exponential(i)
    return prod

eta = 1/2*np.arccos(-1/4)
BB1 = [np.pi/2, -eta, 2*eta, 0, -2*eta, eta]
a = np.linspace(-1,1,100)

# print(a)
P_bb1 = []
for i in a:
    P_bb1.append(U(i, BB1)[0,0])



plt.plot(a, P_bb1)
plt.ylabel('Test')
plt.show()


# U = np.exp(1j * )