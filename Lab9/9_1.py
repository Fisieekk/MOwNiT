# Przedstaw każde z poniższych równań różniczkowych zwyczajnych
# jako równoważny układ równań pierwszego rzędu (ang. first-order system of
# ODEs):
# (a) równanie Van der Pol’a:
# y'' = y' (1 − y^2) − y.
# (b) równanie Blasiusa:
# y''' = −yy''
# (c) II zasada dynamiki Newtona dla problemu dwóch ciał:
# y''1 = −GMy1/ (y1^2 + y2^2)^(3/2)
# y''
# 2 = −GMy2/(y1^2 + y2^2)^(3/2)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# (a) równanie Van der Pol’a:
# y'' = y' (1 − y^2) − y.

def van_der_pol(t, y):
    return [y[1], y[1] * (1 - y[0]**2) - y[0]]

t = np.linspace(0, 20, 1000)
sol = solve_ivp(van_der_pol, [0, 20], [1, 0], t_eval=t)
plt.plot(sol.t, sol.y[0], label='y(t)')
plt.plot(sol.t, sol.y[1], label="y'(t)")
plt.legend()
plt.show()

# (b) równanie Blasiusa:
# y''' = −yy''
def blasius(t, y):
    return [y[1], y[2], -y[0]*y[2]]

t = np.linspace(0, 20, 1000)
sol = solve_ivp(blasius, [0, 20], [1, 0, 0], t_eval=t)
plt.plot(sol.t, sol.y[0], label='y(t)')
plt.plot(sol.t, sol.y[1], label="y'(t)")
plt.plot(sol.t, sol.y[2], label="y''(t)")
plt.legend()
plt.show()

# (c) II zasada dynamiki Newtona dla problemu dwóch ciał:
# y''1 = −GMy1/ (y1^2 + y2^2)^(3/2)
# y''
# 2 = −GMy2/(y1^2 + y2^2)^(3/2)
def newton(t, y):
    G = 1
    M = 1
    return [y[2], y[3], -G*M*y[0]/(y[0]**2 + y[1]**2)**(3/2), -G*M*y[1]/(y[0]**2 + y[1]**2)**(3/2)]

t = np.linspace(0, 20, 1000)
sol = solve_ivp(newton, [0, 20], [1, 0, 0, 1], t_eval=t)
plt.plot(sol.t, sol.y[0], label='y1(t)')
plt.plot(sol.t, sol.y[1], label="y2(t)")
plt.plot(sol.t, sol.y[2], label="y1'(t)")
plt.plot(sol.t, sol.y[3], label="y2'(t)")
plt.legend()
plt.show()
