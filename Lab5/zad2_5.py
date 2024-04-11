# Wykonaj aproksymację średniokwadratową ciągłą funkcji f(x) =
# √
# x w przedziale [0,2] wielomianem drugiego stopnia, używając wielomianów
# Czebyszewa. Aproksymacja ta jest tańszym obliczeniowo zamiennikiem aproksymacji jednostajnej.


import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import quad
from numpy.polynomial.chebyshev import Chebyshev
from scipy.special import chebyt
# dane

def f(x):
    return np.sqrt(x)
def w(x):
    t=x-1
    return (1-t**2)**(-1/2)
def T(k,x):
    x=x-1
    return np.cos(k*np.arccos(x))
def phi(k):
    return np.pi if k==0 else np.pi/2
def c(k):
    c, _ = quad(lambda x: T(k,x)*f(x)*w(x), 0, 2)
    c=c/phi(k)
    return c
def p_gwiazdka(m):
    ck=[]
    for k in range(m+1):
        ck.append(c(k))
    p  = Chebyshev(ck, domain=(0,2))
    return p

def approx(m):
    return p_gwiazdka(m)

# aproksymacja
x = np.linspace(0, 2, 100)
y = f(x)
m = 2
p = approx(m)
y_hat = [p(x_) for x_ in x]
print(p)
print(y_hat)

# wykres
plt.plot(x, y, label='f(x) = √x')
plt.plot(x, y_hat, label='p(x)')
plt.legend()
plt.show()
