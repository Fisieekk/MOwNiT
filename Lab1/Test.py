import numpy as np
import matplotlib.pyplot as plt

# Definicja równania różnicowego
def x_next(x_k, x_km1):
    return 2.25 * x_k - 0.5 * x_km1

# Funkcja generująca ciąg dla pojedynczej precyzji
def generate_sequence_single_precision(n):
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1/3
    x[1] = 1/12
    for k in range(2, n):
        x[k] = x_next(x[k-1], x[k-2])
    return x

# Funkcja generująca ciąg dla podwójnej precyzji
def generate_sequence_double_precision(n):
    x = np.zeros(n, dtype=np.float64)
    x[0] = 1/3
    x[1] = 1/12
    for k in range(2, n):
        x[k] = x_next(x[k-1], x[k-2])
    return x

# Funkcja generująca ciąg dla biblioteki fractions
def generate_sequence_fractions(n):
    from fractions import Fraction
    x = [Fraction(1, 3), Fraction(1, 12)]
    for k in range(2, n):
        x.append(2.25 * x[k-1] - 0.5 * x[k-2])
    return x

# Obliczenia i wykresy
n_single = 225
n_double = 60
n_fractions = 225

x_single = generate_sequence_single_precision(n_single)
x_double = generate_sequence_double_precision(n_double)
x_fractions = generate_sequence_fractions(n_fractions)

# Wykres wartości ciągu w zależności od k (skala logarytmiczna na osi y)
plt.figure(figsize=(10, 6))
plt.semilogy(range(n_single), x_single, label='Single Precision')
plt.semilogy(range(n_double), x_double, label='Double Precision')
plt.semilogy(range(n_fractions), [float(x) for x in x_fractions], label='Fractions')
plt.xlabel('k')
plt.ylabel('x[k]')
plt.title('Wartość ciągu w zależności od k (skala logarytmiczna)')
plt.legend()
plt.show()

# Wykres wartości bezwzględnej błędu względnego w zależności od k
exact_solution = [4**(-k)/3 for k in range(n_single)]

error_single = np.abs((x_single - exact_solution) / exact_solution)
error_double = np.abs((x_double - exact_solution[:n_double]) / exact_solution[:n_double])
error_fractions = [abs(float(x - exact_solution[i]) / float(exact_solution[i])) for i, x in enumerate(x_fractions)]

plt.figure(figsize=(10, 6))
plt.plot(range(n_single), error_single, label='Single Precision')
plt.plot(range(n_double), error_double, label='Double Precision')
plt.plot(range(n_fractions), error_fractions, label='Fractions')
plt.xlabel('k')
plt.ylabel('Bezwzględny błąd względny')
plt.title('Bezwzględny błąd względny w zależności od k')
plt.legend()
plt.show()
