#Rozwiązania równań nieliniowych
# Dla poniższych funkcji i punktów początkowych metoda Newtona
# zawodzi. Wyjaśnij dlaczego. Następnie znajdź pierwiastki, modyfikując wywołanie funkcji scipy.optimize.newton lub używając innej metody.
# (a) f(x) = x^3 − 5x, x0 = 1
# (b) f(x) = x^3 − 3x + 1, x0 = 1
# (c) f(x) = 2 − x^5 , x0 = 0.01
# (d) f(x) = x^4 − 4.29x^2 − 5.29, x0 = 0.8

import numpy as np
from scipy.optimize import newton


def f1(x):
    return x**3 - 5*x
x0_a = 1
def f2(x):
    return x**3 - 3*x + 1
x0_b = 1
def f3(x):
    return 2 - x**5
x0_c = 0.01
def f4(x):
    return x**4 - 4.29*x**2 - 5.29
x0_d = 0.8
print("a) f(x) = x^3 − 5x, x0 = 1")
print(newton(f1, 1))
print("b) f(x) = x^3 − 3x + 1, x0 = 1")
print(newton(f2, 1))
print("c) f(x) = 2 − x^5 , x0 = 0.01")
print(newton(f3, 0.01))
print("d) f(x) = x^4 − 4.29x^2 − 5.29, x0 = 0.8")
print(newton(f4, 0.8))