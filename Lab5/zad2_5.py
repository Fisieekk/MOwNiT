import numpy as np
import matplotlib.pyplot as plt 
from numpy.polynomial.chebyshev import Chebyshev

# dane
x = np.linspace(0, 2, 100)
y = np.sqrt(x)

# funkcja aproksymująca
def approx(x, y, m):
    p = Chebyshev.fit(x, y, m)
    return p

# funkcja ekstrapolująca
def extrapolate(p, x):
    return p(x)

# aproksymacja
m = 2
p = approx(x, y, m)
y_hat = extrapolate(p, x)

# wykres
plt.plot(x, y, label='f(x) = √x')
plt.plot(x, y_hat, label='p(x)')
plt.legend()
plt.show()