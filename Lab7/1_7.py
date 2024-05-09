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

def f3(x):
    return (1/(((abs(x)-0.3)**2)+0.001)) + (1/(((abs(x)-0.9)**2)+0.004)) - 6

def f3_error(y):
    a = 0.001
    b = 0.004
    return np.abs(y - ((1/np.sqrt(a))*(np.arctan(0.7/np.sqrt(a))+np.arctan(0.3/np.sqrt(a))) + (1/np.sqrt(b))*(np.arctan(0.1/np.sqrt(b))+np.arctan(0.9/np.sqrt(b))) - 6))/np.abs(((1/np.sqrt(a))*(np.arctan(0.7/np.sqrt(a))+np.arctan(0.3/np.sqrt(a))) + (1/np.sqrt(b))*(np.arctan(0.1/np.sqrt(b))+np.arctan(0.9/np.sqrt(b))) - 6))    

def integrator_f1_f2(f):
    errors_gk = []
    evaluations_gk = []
    errors_trap = []
    evaluations_trap = []
    for i in range(15):
        result_gk, error_estimate_gk, num_func_evals_gk = quad_vec(f, 0, 1,quadrature='gk15', epsrel=(1/10)**i,full_output=True)
        errors_gk.append(error_estimate_gk)
        evaluations_gk.append(num_func_evals_gk.neval)
        result_trapz, error_estimate_trapz, num_func_evals_trapz = quad_vec(f, 0, 1, quadrature='trapezoid', epsabs=(1/10)**i,full_output=True)
        errors_trap.append(error_estimate_trapz)
        evaluations_trap.append(num_func_evals_trapz.neval)
    return errors_gk, evaluations_gk, errors_trap, evaluations_trap
errors_gk1, evaluations_gk1, errors_trap1, evaluations_trap1 = integrator_f1_f2(f1)
# Plotting
x=np.arange(15)
plt.figure(figsize=(10, 6))
plt.loglog(evaluations_gk1, errors_gk1, label='Gauss-Kronrod')
plt.loglog(evaluations_trap1, errors_trap1, label='Trapezoidal')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Error')
plt.title('Error vs Number of Function Evaluations')
plt.legend()
plt.grid(True)
plt.show()


errors_gk2, evaluations_gk2, errors_trap2, evaluations_trap2 = integrator_f1_f2(f2)
# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(evaluations_gk2, errors_gk2, label='Gauss-Kronrod')
plt.loglog(evaluations_trap2, errors_trap2, label='Trapezoidal')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Error')
plt.title('Error vs Number of Function Evaluations')
plt.legend()
plt.grid(True)
plt.show()


def integrator_f3(f,exact):
    errors_gk = []
    evaluations_gk = []
    for i in range(15):
        result_gk, error_estimate_gk, num_func_evals_gk = quad_vec(f, 0, 1,quadrature='gk15', epsrel=(1/10)**i,full_output=True)
        errors_gk.append(f3_error(result_gk))
        evaluations_gk.append(num_func_evals_gk.neval)
    return errors_gk, evaluations_gk

errors_gk2, evaluations_gk2 = integrator_f3(f3,6)

# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(evaluations_gk2, errors_gk2)
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Error')
plt.title('Error vs Number of Function Evaluations')
plt.grid(True)
plt.show()



