import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss as gauss
def f(x):
    return 4/(1+x**2)

def gauss_legendre_rule(a, b, n):
    x, w = gauss(n)
    return np.sum(w * f((b - a) / 2 * x + (a + b) / 2))

def calculate_integral(a, b, n, method):
    return method(a, b, n)

def calculate_error(a, b, n, method):
    return np.abs(np.pi - calculate_integral(a, b, n, method))

def calculate_errors(a, b, n, method):
    errors = []
    for i in range(1, n + 1):
        errors.append(calculate_error(a, b, i, method))
    return errors

a = 0
b = 1
n = 25

gauss_legendre_errors = calculate_errors(a, b, n, gauss_legendre_rule)

n = np.arange(1, n + 1)
plt.plot(n, gauss_legendre_errors, label='Gauss-Legendre rule')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
