# Dane jest równanie:
# f(x) = x^2 − 3x + 2 = 0 
# Każda z następujących funkcji definiuje równoważny schemat iteracyjny:
# g1(x) = (x^2 + 2)/3, 
# g2(x) = sqrt(3x − 2),
# g3(x) = 3 − 2/x,
# g4(x) = (x^2 − 2)/(2x − 3).
# (a) Przeanalizuj zbieżność oraz rząd zbieżności schematów iteracyjnych odpowiadających funkcjom gi(x) 
#dla pierwiastka x = 2 badając wartość |g'i(2)|.
# (b) Potwierdź analizę teoretyczną implementując powyższe schematy iteracyjne
# i weryfikując ich zbieżność (lub brak). Każdy schemat iteracyjny wykonaj
# przez 10 iteracji.
# Wyznacz eksprymentalnie rząd zbieżności każdej metody iteracyjnej ze wzoru
# r =ln(εk/(εk+1))/(ln((εk−1)/εk))
# gdzie błąd bezwzględny εk definiujemy jako εk = |xk − x∗|, 
#xk jest przybliżeniem pierwiastka w k-tej iteracji, a x∗ dokładnym położeniem pierwiastka równania.
# (c) Na wspólnym rysynku przedstaw wykresy błędu względnego każdej metody
# w zależności od numeru iteracji. Użyj skali logarytmicznej na osi y (pomocna
# będzie funkcja semilogy).
# Stwórz drugi rysunek, przedstawiający wykresy błędu względnego tylko dla
# metod zbieżnych.


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def f(x):
    return x**2 - 3*x + 2

def g1(x):
    return (x**2 + 2)/3

def g2(x):
    return np.sqrt(3*x - 2)

def g3(x):
    return 3 - 2/x

def g4(x):
    if(x == 3/2):
        return 1000000
    return (x**2 - 2)/(2*x - 3)

def g1_prime(x):
    return 2*x/3

def g2_prime(x):
    return 3/(2*np.sqrt(3*x - 2))

def g3_prime(x):
    return 2/(x**2)

def g4_prime(x):
    return (2*x*(2*x-3) - (x**2 - 2)*2)/(2*x - 3)**2

derivetives_in_2 = {"g1'":g1_prime(2),"g2'":g2_prime(2),"g3'":g3_prime(2),"g4'":g4_prime(2)}

def abs_error(x, x_star):
    return abs(x-x_star)

def iterator(g,x0,n):
    x = x0
    errors = []
    for i in range(n):
        x = g(x)
        errors.append(abs_error(x,2))
    return errors

def iteration_scheme(gs,n):
    x0=1.5
    errors = {}
    for key in gs:
        errors[key] = iterator(gs[key],x0,n)
    return errors
        
def convergence_order(errors):
    orders = []
    for key in errors:
        error = errors[key]
        order = []
        for i in range(1,len(error)-1):
            order.append(np.log(error[i-1]/error[i])/np.log(error[i]/error[i+1]))
        orders.append(order)
    return orders

def main():
    iterations=10
    gs={'g1':g1,'g2':g2,'g3':g3,'g4':g4}
    errors = iteration_scheme(gs,iterations)
    orders = convergence_order(errors)
    plt.figure(figsize=(10, 5))
    for key in errors:
        plt.semilogy(errors[key], label=key, marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.title('Wykresy błędu względnego dla każdej funkcji g')
    plt.xlabel('Numer iteracji')
    plt.ylabel('Błąd względny')
    plt.legend()
    plt.grid(True)
    plt.show()

print(main())