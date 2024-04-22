import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# dane
years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population=np.array([76212168,92228496,106021537,123202624,132164569,151325798,179323175,203302031,226542199])

# funkcja aproksymująca

def approx(x, y, m):
    p = Polynomial.fit(x, y, m)
    return p

# funkcja ekstrapolująca

def extrapolate(p, x):
    return p(x)

# funkcja błędu względnego

def relative_error(y, y_hat):
    return np.abs(y - y_hat) / y

# funkcja kryterium informacyjnego Akaikego

def AIC(y, y_hat, k):
    n = len(y)
    return 2 * k + n * np.log(np.sum((y - y_hat) ** 2) / n)

# funkcja kryterium informacyjnego Akaikego ze składnikiem koregującym

def AICc(y, y_hat, k):
    n = len(y)
    return AIC(y, y_hat, k) + 2 * k * (k + 1) / (n - k - 1)

# funkcja optymalizacyjna

def optimize(x, y, m):
    p = approx(x, y, m)
    y_hat = extrapolate(p, 1990)
    return relative_error(248709873, y_hat)

# optymalizacja stopnia wielomianu

m = np.arange(7)
errors = np.array([optimize(years, population, i) for i in m])
print(errors)
plt.plot(m, errors,'o', label='Błąd względny')
plt.xlabel('Stopień')
plt.ylabel('Błąd względny')
plt.legend()
plt.show()
best_m = m[np.argmin(errors)]
print(f'Najlepszy stopień wielomianu: {best_m}')
print(f'Błąd względny dla najlepszego stopnia wielomianu: {round(errors[best_m]*100,2)}%')

# aproksymacja dla najlepszego stopnia wielomianu

p = approx(years, population, best_m)
y_hat = extrapolate(p, 1990)
print(f'Wartość ekstrapolowana dla roku 1990: {y_hat}')


# kryterium informacyjne Akaikego z korektą

aic_c = np.array([AICc(population, extrapolate(approx(years, population, i), years), i) for i in m])
best_aic_c = m[np.argmin(aic_c)]
print(aic_c)
print(f'Najlepszy stopień wielomianu wg AICc: {best_aic_c}')
print(f'Wartość kryterium informacyjnego Akaikego z korektą dla najlepszego stopnia wielomianu: {aic_c[best_aic_c]}')

# wykres

plt.plot(years, population, 'o', label='Dane')
for i in m:
    plt.plot(years, extrapolate(approx(years, population, i), years), label=f'm={i}')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.legend()
plt.show()

# Wnioski:
