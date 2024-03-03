import numpy as np
import matplotlib.pyplot as plt

def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def true_derivative_tan(x):
    return 1 + np.tan(x)**2

def absolute_error(approximate, true):
    return np.abs(approximate - true)

def compute_errors(f, x, h_values, derivative_function,diff):
    numerical_errors = []
    approximation_errors = []
    computational_errors = []
    true_value = derivative_function(x)
    for h in h_values:
        numerical_derivative = diff(f, x, h)
        
        numerical_error = absolute_error(numerical_derivative, true_value)
        
        approximation_errors.append(numerical_error)
        numerical_errors.append(numerical_error)
        
        computational_error = np.abs(h)
        
        computational_errors.append(computational_error)
    print("numerical_errors",numerical_errors, "approximation_errors",approximation_errors, "computational_errors",computational_errors)
    return numerical_errors, approximation_errors, computational_errors

# Funkcja tan(x)
f = np.tan
x = 1
h_values = np.power(10., -np.arange(17))

numerical_errors_fd, approximation_errors_fd, computational_errors_fd = compute_errors(f, x, h_values, true_derivative_tan,forward_difference)

h_min_fd = 2 * np.sqrt(np.finfo(float).eps / np.abs(np.tan(x)**3))
h_min_fd_values = np.full_like(h_values, h_min_fd)

numerical_errors_cd, approximation_errors_cd, computational_errors_cd = compute_errors(f, x, h_values, true_derivative_tan,central_difference)

h_min_cd = np.cbrt(3 * np.finfo(float).eps / np.abs(np.tan(x)**3))
h_min_cd_values = np.full_like(h_values, h_min_cd)

# Tworzenie wykresów
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.loglog(h_values, numerical_errors_fd, label='Numerical Error (Forward Difference)')
# plt.loglog(h_values, approximation_errors_fd, label='Approximation Error (Forward Difference)')
# plt.loglog(h_values, computational_errors_fd, label='Computational Error (Forward Difference)')
plt.loglog(h_values, h_min_fd_values, '--', label='h_min (Forward Difference)', color='black')
plt.title('Errors Analysis (Forward Difference)')
plt.xlabel('h')
plt.ylabel('Absolute Error')
plt.legend()

plt.subplot(2, 1, 2)
plt.loglog(h_values, numerical_errors_cd, label='Numerical Error (Central Difference)')
# plt.loglog(h_values, approximation_errors_cd, label='Approximation Error (Central Difference)')
# plt.loglog(h_values, computational_errors_cd, label='Computational Error (Central Difference)')
plt.loglog(h_values, h_min_cd_values, '--', label='h_min (Central Difference)', color='black')
plt.title('Errors Analysis (Central Difference)')
plt.xlabel('h')
plt.ylabel('Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()
