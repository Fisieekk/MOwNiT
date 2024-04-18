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
from scipy.integrate import quad
from scipy.integrate import trapezoid as trapz
from scipy.integrate import simpson as simps

def f(x):
    return 4/(1+x**2)


def mid_point_rule(x1,y1):
    return np.sum(y1*(x1[1]-x1[0]))
def trapezoidal_rule(x1,y1):
    return trapz(x=x1,y=y1)

def simpson_rule(x1,y1):
    return simps(x=x1,y=y1)

def calculate_errors(x, y, m):
    mid_point_errors = []
    trapezoidal_errors = []
    simpson_errors = []
    for i in range(1, m + 1):
        ab = np.linspace(start=0, stop=1, num=2**i+1)
        yab = f(ab)
        x1=np.array([(ab[i]+ab[i-1])/2 for i in range(1, len(ab))])
        y1=f(x1)
        mid_point_errors.append(np.abs(np.pi - mid_point_rule(x1, y1)))
        trapezoidal_errors.append(np.abs(np.pi - trapezoidal_rule(ab, yab)))
        simpson_errors.append(np.abs(np.pi - simpson_rule(ab, yab)))
    return mid_point_errors, trapezoidal_errors, simpson_errors

a = 0   
b = 1
m=25
n = np.array([1+2**(i) for i in range(1,m+1)])

mid_point_errors, trapezoidal_errors, simpson_errors = calculate_errors(a, b, m)
print(mid_point_errors)
plt.plot(n, mid_point_errors, label='Mid-point rule')
plt.plot(n, trapezoidal_errors, label='Trapezoidal rule')
plt.plot(n, simpson_errors, label='Simpson rule')
plt.yscale('log')
plt.xscale('log',base=2)
plt.legend()
plt.show()
