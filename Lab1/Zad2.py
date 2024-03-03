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
def exact_solution(k):
    return Frac(4, 3) ** -k    

print(np.zeros(3,np.float64))
def generator(n,precision):
    x_k_values = np.zeros(n, dtype=precision)
    x_k_values[0]=Frac(1,3)
    x_k_values[1]=Frac(1/12)
    for i in range(2,n):
        x_k_values[i]=(x_k(i,x_k_values))
    print(x_k_values)
    return x_k_values

def main():
    n_single = 225
    n_double = 60
    n_fraction = 225
    plt.semilogy(np.arange(1, n_single+1), generator(n_single,np.float32), label='Single Precision')
    plt.semilogy(np.arange(1, n_double+1), generator(n_double,np.float64), label='Double Precision')
    plt.semilogy(np.arange(1, n_fraction+1), generator(n_fraction,Frac), label='Fractions')
    plt.title('Sequence Values')
    plt.xlabel('k')
    plt.ylabel('Sequence Value')
    plt.legend()
    plt.grid(True)
    plt.show()
print(main())