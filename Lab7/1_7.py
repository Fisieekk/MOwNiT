# calculate integral from 0 to 1 of 4/(1+x^2) by
# a) adaptive Gauss-kronrod quadrature
# b) adaptive trapezoidal quadrature
# use tolerance between 10 and 10^-14
# create chart of error vs number of function evaluations
# exact value is pi
# you have to use quad_vec from scipy.integrate

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec

def f1(x):
    return 4/(1+x**2)

def f2(x):
    return np.sqrt(x)*np.log(x)



exact1 = np.pi
exact2 = -4/9
def integrator(f,exact):
    tolerance_range = np.logspace(1,-14,num=100)
    errors_gk = []
    evaluations_gk = []
    errors_trap = []
    evaluations_trap = []
    for tol in tolerance_range:
        result, error_estimate, num_func_evals = quad_vec(f, 0, 1,quadrature='gk15', epsabs=tol,full_output=True)
        errors_gk.append(np.abs(result - exact))
        evaluations_gk.append(num_func_evals.neval)
        result, error_estimate, num_func_evals = quad_vec(f, 0, 1, quadrature='trapezoid', epsabs=tol,full_output=True)
        errors_trap.append(np.abs(result - exact))
        evaluations_trap.append(num_func_evals.neval)
    return errors_gk, evaluations_gk, errors_trap, evaluations_trap
errors_gk, evaluations_gk, errors_trap, evaluations_trap = integrator(f1,exact1)
# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(evaluations_gk, errors_gk, label='Gauss-Kronrod')
plt.loglog(evaluations_trap, errors_trap, label='Trapezoidal')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Error')
plt.title('Error vs Number of Function Evaluations')
plt.legend()
plt.grid(True)
plt.show()


errors_gk1, evaluations_gk1, errors_trap1, evaluations_trap1 = integrator(f2,exact2)
# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(evaluations_gk1, errors_gk1, label='Gauss-Kronrod')
plt.loglog(evaluations_trap1, errors_trap1, label='Trapezoidal')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Error')
plt.title('Error vs Number of Function Evaluations')
plt.legend()
plt.grid(True)
plt.show()