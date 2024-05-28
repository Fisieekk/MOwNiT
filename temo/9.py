# Zadanie 3. Model Kermack’a-McKendrick’a przebiegu epidemii w populacji
# opisany jest układem równań różniczkowych:
# S
# 0 = −
# β
# N
# IS, (3)
# I
# 0 =
# β
# N
# IS − γI, (4)
# R
# 0 = γI, (5)
# gdzie
# S reprezentuje liczbę osób zdrowych, podatnych na zainfekowanie,
# I reprezentuje liczbę osób zainfekowanych i roznoszących infekcję,
# R reprezentuje liczbę osób ozdrowiałych.
# Liczba N to liczba osób w populacji. Parametr β reprezentuje współczynnik
# zakaźności (ang. transmission rate). Parametr γ reprezentuje współczynnik wyzdrowień (ang. recovery rate). Wartość 1/γ reprezentuje średni czas choroby.
# Założenia modelu:
# • Przyrost liczby osób zakażonych jest proporcjonalny do liczby osób zakażonych oraz do liczby osób podatnych.
# • Przyrost liczby osób odppornych lub zmarłych jest wprost proporcjonalny
# do liczby aktualnie chorych.
# • Okres inkubacji choroby jest zaniedbywalnie krótki.
# • Populacja jest wymieszana.
# Susceptible Infectious Recovered
# βIS γI
# Jako wartości początkowe ustal:
# S(0) = 762, I(0) = 1, R(0) = 0 .
# Przyjmij też N = S(0)+I(0)+R(0) = 763 oraz β = 1. Zakładając, że średni
# czas trwania grypy wynosi 1/γ = 7 dni, przyjmij γ = 1/7.
# Całkując od t = 0 do t = 14 z krokiem 0.2, rozwiąż powyższy układ równań:
# • jawną metodą Eulera
# yk+1 = yk + hkf(tk, yk)
# • niejawną metodą Eulera
# yk+1 = yk + hkf(tk+1, yk+1)
# 2
# • metodą Rungego-Kutty czwartego rzędu (RK4)
# yk+1 = yk +
# hk
# 6
# (k1 + 2k2 + 2k3 + k4), gdzie
# k1 = f(tk, yk)
# k2 = f(tk + hk/2, yk + hkk1/2)
# k3 = f(tk + hk/2, yk + hkk2/2)
# k4 = f(tk + hk, yk + hkk3)
# Wykonaj nastepujące wykresy:
# • Dla każdej metody przedstaw na wspólnym rysunku wykresy komponentów rozwiązania (S, I, R) jako funkcje t (3 wykresy).
# • Na wspólnym rysunku przedstaw wykresy funkcji S(t) +I(t) +R(t) znalezione przez każdą metodę (1 wykres). Czy niezmiennik S(t)+I(t)+R(t) ≡
# N jest zachowany?
# Wiemy, że liczba osób zakażonych w pewnej szkole kształtowała się następująco:
# Dzień, t 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
# Zakażeni, I 1 3 6 25 73 222 294 258 237 191 125 69 27 11 4
# Wybierz jedną z powyższych metod numerycznych i oszacuj prawdziwe wartości współczynników θ = [β, γ]. W tym celu wykonaj minimalizację funkcji
# kosztu. Jako funkcję kosztu wykorzystaj sumę kwadratów reszt (ang. residual
# sum of squares):
# L(θ) = X
# T
# i=0
# (Ii − ˆIi)
# 2
# ,
# gdzie Ii oznacza prawdziwą liczbę zakażonych, a ˆIi oznacza liczbę zakażonych
# wyznaczonych metodą numeryczną. Ponieważ nie znamy gradientu ∇θL(θ), do
# minimalizacji wykorzystaj metodę Neldera-Meada, która nie wymaga informacji
# o gradiencie.
# Powtórz obliczenia, tym razem jako funkcję kosztu wykorzystując:
# L(θ) = −
# X
# T
# i=0
# Ii
# ln ˆIi +
# X
# T
# i=0
# ˆIi
# .
# Ile wynosił współczynnik reprodukcji R0 = β/γ w każdym przypadku?

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parametry
N = 763
beta = 1
gamma = 1/7
S0 = 762

# Funkcje
def f(t, y):
    S, I, R = y
    dS = -beta/N * S * I
    dI = beta/N * S * I - gamma * I
    dR = gamma * I
    return [dS, dI, dR]

def euler_explicit(f, y0, t0, t_max, h):
    t = np.arange(t0, t_max+h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * np.array(f(t[i-1], y[i-1]))
    return t, y

def euler_implicit(f, y0, t0, t_max, h):
    t = np.arange(t0, t_max+h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * np.array(f(t[i], y[i]))
    return t, y

def runge_kutta(f, y0, t0, t_max, h):
    t = np.arange(t0, t_max+h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + h/2, y[i-1] + h/2 * np.array(k1))
        k3 = f(t[i-1] + h/2, y[i-1] + h/2 * np.array(k2))
        k4 = f(t[i-1] + h, y[i-1] + h * np.array(k3))
        print(k1, k2, k3, k4)
        y[i] = y[i-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

# Rozwiązania
t, y_euler_explicit = euler_explicit(f, [S0, 1, 0], 0, 14, 0.2)
t, y_euler_implicit = euler_implicit(f, [S0, 1, 0], 0, 14, 0.2)
t, y_runge_kutta = runge_kutta(f, [S0, 1, 0], 0, 14, 0.2)

# Wykresy
plt.figure()
plt.plot(t, y_euler_explicit[:, 0], label='S')
plt.plot(t, y_euler_explicit[:, 1], label='I')
plt.plot(t, y_euler_explicit[:, 2], label='R')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Euler explicit')
plt.legend()
plt.grid()

plt.figure()
plt.plot(t, y_euler_implicit[:, 0], label='S')
plt.plot(t, y_euler_implicit[:, 1], label='I')
plt.plot(t, y_euler_implicit[:, 2], label='R')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Euler implicit')
plt.legend()
plt.grid()

plt.figure()
plt.plot(t, y_runge_kutta[:, 0], label='S')
plt.plot(t, y_runge_kutta[:, 1], label='I')
plt.plot(t, y_runge_kutta[:, 2], label='R')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Runge-Kutta')
plt.legend()
plt.grid()

plt.figure()
plt.plot(t, y_euler_explicit[:, 0] + y_euler_explicit[:, 1] + y_euler_explicit[:, 2], label='Euler explicit')
plt.plot(t, y_euler_implicit[:, 0] + y_euler_implicit[:, 1] + y_euler_implicit[:, 2], label='Euler implicit')
plt.plot(t, y_runge_kutta[:, 0] + y_runge_kutta[:, 1] + y_runge_kutta[:, 2], label='Runge-Kutta')
plt.xlabel('t')
plt.ylabel('y')
plt.title('S + I + R')
plt.legend()
plt.grid()

# Współczynniki
I = np.array([1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4])
def cost_function(theta, I):
    beta, gamma = theta
    t, y = euler_explicit(f, [S0, 1, 0], 0, 14, 0.2)
    return sum((I - y[:, 1])**2)

theta = minimize(cost_function, [1, 1/7], args=(I))
print('Współczynniki:', theta.x)
R0 = theta.x[0] / theta.x[1]
print('R0:', R0)

def cost_function2(theta, I):
    beta, gamma = theta
    t, y = euler_explicit(f, [S0, 1, 0], 0, 14, 0.2)
    return -sum(I * np.log(y[:, 1]) + y[:, 1])

theta = minimize(cost_function2, [1, 1/7], args=(I))
print('Współczynniki:', theta.x)
R0 = theta.x[0] / theta.x[1]
print('R0:', R0)

plt.show()

