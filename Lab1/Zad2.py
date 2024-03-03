import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction as Frac
def x_k(i,x_k_values):
    if(i==0):
        return x_k_values[0]
    elif(i==1):
        return x_k_values[1]
    else:
        return 2.25*x_k_values[i-1]-0.5*x_k_values[i-2]


def generator(n,precision):
    x_k_values = np.zeros(n, dtype=precision)
    x_k_values[0]=Frac(1,3)
    x_k_values[1]=Frac(1,12)
    for i in range(2,n):
        x_k_values[i]=(x_k(i,x_k_values))
    return x_k_values

def good_value(n):
    x_k_values = np.zeros(n)
    for i in range(n):
        x_k_values[i]=(4**(-i))/3
    return x_k_values

def main():
    n_single = 225
    n_double = 60
    n_fraction = 225
    x_single=generator(n_single,np.float32)
    x_double=generator(n_double,np.float64)
    x_fractions=generator(n_fraction,Frac)
    x_proper=good_value(n_single)
    error_single = np.abs((x_single - x_proper) / x_proper)
    print(error_single)
    error_double = np.abs((x_double - x_proper[:n_double]) / x_proper[:n_double])
    print(error_double)
    error_fractions = [abs(float(x - x_proper[i]) / float(x_proper[i])) for i, x in enumerate(x_fractions)] 
    print(error_fractions)
    plt.semilogy(np.arange(1, n_fraction+1),x_fractions, label='Fractions')
    plt.semilogy(np.arange(1, n_single+1),x_single, label='Single Precision')
    plt.semilogy(np.arange(1, n_double+1),x_double , label='Double Precision')
    plt.semilogy(np.arange(1, n_fraction+1), x_proper, label='Good Values')
    plt.title('Sequence Values')
    plt.xlabel('k')
    plt.ylabel('Sequence Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(n_single), error_single, label='Single Precision')
    plt.plot(range(n_double), error_double, label='Double Precision')
    plt.plot(range(n_fraction), error_fractions, label='Fractions')
    plt.xlabel('k')
    plt.ylabel('Bezwzględny błąd względny')
    plt.title('Bezwzględny błąd względny w zależności od k')
    plt.legend()
    plt.show()

main()
