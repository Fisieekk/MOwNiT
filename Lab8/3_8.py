# Napisz schemat iteracji wg metody Newtona dla każdego z następujących równań nieliniowych:
# (a) x^3 − 2x − 5 = 0
# (b) e^−x = x
# (c) x sin(x) = 1.
# Jeśli x0 jest przybliżeniem pierwiastka z dokładnością 4 bitów, ile iteracji należy
# wykonać aby osiągnąć:
# • 24-bitową dokładność
# • 53-bitową dokładność?


import numpy as np
import sympy as sp

def f_a(x):
    return x**3 - 2*x - 5
x0_a = 1
def f_b(x):
    return np.exp(-x) - x
x0_b = 1
def f_c(x):
    return x*np.sin(x) - 1
x0_c = 1

def f_a_prime(x):
    return 3*x**2 - 2
def f_b_prime(x):
    return -np.exp(-x) - 1
def f_c_prime(x):
    return np.sin(x) + x*np.cos(x)

def newton_iterations(f, f_prime, x0, n,focus):
    x = x0
    iterations = 0
    for i in range(n):
        x = x - f(x)/f_prime(x)
        iterations += 1
        if(abs(f(x)) < 2**(-focus)):
            break
    return x, iterations


def main():
    tolerance24 = 2**(-24)
    tolerance53 = 2**(-53)
    x0=1.0
    results24 = {}
    results53 = {}
    iterations24 = {}
    iterations53 = {}
    functions = {"a":(f_a, f_a_prime, x0_a), "b":(f_b, f_b_prime, x0_b), "c":(f_c, f_c_prime, x0_c)}
    for key in functions:
        results24[key], iterations24[key] = newton_iterations(functions[key][0], functions[key][1], functions[key][2], 100, 24)
        results53[key], iterations53[key] = newton_iterations(functions[key][0], functions[key][1], functions[key][2], 100, 53)
    print("24-bit accuracy:")
    for key in results24:
        print(f"Function {key}: {results24[key]} after {iterations24[key]} iterations")
    print("53-bit accuracy:")
    for key in results53:
        print(f"Function {key}: {results53[key]} after {iterations53[key]} iterations")


main()