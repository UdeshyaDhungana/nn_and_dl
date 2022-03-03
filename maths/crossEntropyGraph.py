# Graph Cross Entropy Function

import numpy as np
import matplotlib.pyplot as plt


def cross_entropy(a, y):
    return y * np.log(a) + (1 - y) * np.log(1 - a)


x = np.arange(0.001, 1, 0.001)
X, Y = np.meshgrid(x, x)
Z = cross_entropy(X, Y)

plt.contour(X, Y, Z, cmap='RdGy', origin='lower',);
plt.xlabel('a')
plt.ylabel('y')
plt.colorbar()
plt.grid()
plt.show()

# Conclusion
# Similar the values of y and a, lower the value of z, and vice versa.