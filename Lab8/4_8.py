# Napisz schemat iteracji wg metody Newtona dla następującego
# układu równań nieliniowych:
# x
# 2
# 1 + x
# 2
# 2 = 1
# x
# 2
# 1 − x2 = 0.
# Korzystając z faktu, że dokładne rozwiązanie powyższego układu równań to:
# x1 = ±
# s√
# 5
# 2
# −
# 1
# 2
# (7a)
# x2 =
# √
# 5
# 2
# −
# 1
# 2
# (7b)
# oblicz błąd względny rozwiązania znalezionego metodą Newtona


import numpy as np
import sympy as sp

def f1(x1,x2):
    return x1**2 + x2**2 - 1

def f2(x1,x2):
    return x1**2 - x2

def newton_method(f1,f2,x1,x2,n,focus):
    x = np.array([x1,x2],dtype=float)
    for i in range(n):
        f = np.array([f1(x[0],x[1]),f2(x[0],x[1])])
        jacobian = np.array([[2*x[0],2*x[1]],[2*x[1],-1]])
        delta_x = np.linalg.solve(jacobian,-f)
        x+=delta_x  
        if np.linalg.norm(delta_x) < focus:
            return x,i
    raise("No convergence")

def relative_error(x,true_x):
    return abs(x - true_x)/abs(true_x)


def main():
    x1 = np.sqrt(np.sqrt(5)/2 - 1/2)
    x2 = np.sqrt(5)/2 - 1/2
    focus = 10**(-6)
    max_iterations = 100
    x,iterations = newton_method(f1,f2,1.0,1.0,max_iterations,focus)    
    errors = [relative_error(x1,x[0]),relative_error(x2,x[1])]
    print("Newton method result: ", x)
    print("True Values: ",x1, x2)
    print("Iterations:", iterations)
    print("x1 Relative Error:", errors[0])
    print("x2 Relative Error:", errors[1])

main()