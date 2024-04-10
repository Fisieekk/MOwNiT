import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.polynomial.polynomial import Polynomial
# dane
years= np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population = np.array([76212168, 92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203302031, 226542199])    

def build_matrix(m,n,x):
    A = np.zeros((m,n+1))
    for i in range(m):
        for j in range(n+1):
            A[i][j] = x[i]**j 
    return A


def approx(x,y,n):
    m = len(x)
    A = build_matrix(m,n,x)
    c=(inv(A.T@A)@A.T)@y
    return c

def extrapolate(c,x):
    y = 0
    for i in range(len(c)):
        y += c[i]*x**i
    return y


def relative_error(y, y_hat):
    return np.abs(y - y_hat) / y
def optimize(x, y, m):
    p = approx(x, y, m)
    y_hat = extrapolate(p, 1990)
    return relative_error(248709873, y_hat)

n = np.arange(7)
errors = np.array([optimize(years, population, i) for i in n])
plt.plot(n, errors,'o', label='Błąd względny')
plt.xlabel('Stopień')
plt.ylabel('Błąd względny')
plt.legend()
plt.show()
best_m = n[np.argmin(errors)]
print(f'Najlepszy stopień wielomianu: {best_m}')
print(f'Błąd względny dla najlepszego stopnia wielomianu: {round(errors[best_m]*100,2)}%')

# aproksymacja dla najlepszego stopnia wielomianu
p = approx(years, population, best_m)
y_hat = extrapolate(p, 1990)
print(f'Wartość ekstrapolowana dla roku 1990: {y_hat}')

# kryterium informacyjne Akaikego
def AIC(y, y_hat, k):
    n = len(y)
    return 2 * k + n * np.log(np.sum((y - y_hat) ** 2) / n)

# funkcja kryterium informacyjnego Akaikego ze składnikiem koregującym

def AICc(y, y_hat, k):
    n = len(y)
    return AIC(y, y_hat, k) + 2 * k * (k + 1) / (n - k - 1)



# aic = np.array([AIC(population, extrapolate(approx(years, population, i), years), i) for i in n])
# best_aic = n[np.argmin(aic)]
# print(f'Najlepszy stopień wielomianu wg AIC: {best_aic}')
# print(f'Wartość kryterium informacyjnego Akaikego dla najlepszego stopnia wielomianu: {aic[best_aic]}')

# kryterium informacyjne Akaikego z korektą

aic_c = np.array([AICc(population, extrapolate(approx(years, population, i), years), i) for i in n])
best_aic_c = n[np.argmin(aic_c)]
print(aic_c)
print(f'Najlepszy stopień wielomianu wg AICc: {best_aic_c}')
print(f'Wartość kryterium informacyjnego Akaikego z korektą dla najlepszego stopnia wielomianu: {aic_c[best_aic_c]}')

# wykres

plt.plot(years, population, 'o', label='Dane')
plt.plot(years, extrapolate(p, years), label=f'Ekstrapolacja dla m={best_m}')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.legend()
plt.show()
