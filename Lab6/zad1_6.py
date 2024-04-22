# Wiadomo, że
# Z 1
# 0
# 4
# 1 + x
# 2
# dx = π . (1)
# Powyższą równość można wykorzystać do obliczenia przybliżonej wartości π poprzez całkowanie numeryczne.
# (a) Oblicz wartość powyższej całki, korzystając ze złożonych kwadratur otwartej prostokątów (ang. mid-point rule), trapezów i Simpsona. Można wykorzystać funkcje integrate.trapz i integrate.simps z biblioteki scipy. Na
# przedziale całkowania rozmieść 2
# m +1 równoodległych węzłów. W kolejnych
# próbach m wzrasta o 1, tzn. między każde dwa sąsiednie węzły dodawany
# jest nowy węzeł, a ich zagęszczenie zwiększa się dwukrotnie. Przyjmij zakres
# wartości m od 1 do 25.
# Dla każdej metody narysuj wykres wartości bezwzględnej błędu względnego
# w zależności od liczby ewaluacji funkcji podcałkowej, n + 1 (gdzie n =
# 1/h, z krokiem h). Wyniki przedstaw na wspólnym wykresie, używając skali
# logarytmicznej na obu osiach.
# (b) Czy istnieje pewna wartość, poniżej której zmniejszanie kroku h nie zmniejsza już błędu kwadratury? Porównaj wartość hmin, odpowiadającą minimum wartości bezwzględnej błędu względnego, z wartością wyznaczoną w
# laboratorium 1.
# (c) Dla każdej z użytych metod porównaj empiryczny rząd zbieżności z rząd
# zbieżności przewidywanym przez teorię. Aby wyniki miały sens, do obliczenia rzędu empirycznego użyj wartości h z zakresu, w którym błąd metody
# przeważa nad błędem numerycznym.

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# funkcja do całkowania

def f(x):
    return 4 / (1 + x ** 2)


m_values = np.arange(1, 26)

def calculate_integral(x, y, method):
    return method(y, x)
    
# metody całkowania
def mid_point_rule(y, x):
    sum = 0
    h = x[1] - x[0]
    for i in range(1, len(x)):
        mid = (x[i - 1] + x[i]) / 2
        sum += y(mid) * h   
    return sum


methods = [mid_point_rule, integrate.trapezoid, integrate.simpson]


def calculate_error(excat, approx):
    return np.abs((excat - approx) / excat)

# obliczenia

def calculate_integral_error(f,methods,m_values):
    errors_trapz = []
    errors_simps = []
    errors_midpoint = []
    exact=np.pi
    a,b=0,1
    for m in m_values:
        x = np.linspace(a, b, 2 ** m + 1)
        y = f(x)
    
        # wartość całki
        value = calculate_integral(x,f,methods[0])
        errors_midpoint.append(calculate_error(value, exact))

        value = calculate_integral(x,y,methods[1])
        errors_trapz.append(calculate_error(value, exact))

        value = calculate_integral(x,y,methods[2])
        errors_simps.append(calculate_error(value, exact))


    return errors_midpoint, errors_trapz, errors_simps

errors_midpoint, errors_trapz, errors_simps = calculate_integral_error(f,methods,m_values)

# rysowanie wykresu

plt.figure(figsize=(10, 6))
plt.plot(m_values, errors_midpoint, label="Midpoint rule")
plt.plot(m_values, errors_trapz, label="Trapezoidal rule")
plt.plot(m_values, errors_simps, label="Simpson's rule")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of evaluations")
plt.ylabel("Relative error")
plt.legend()
plt.show()

