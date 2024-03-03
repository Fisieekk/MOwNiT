import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys
epsilon=np.finfo(float).eps



def f(x):
    return np.tan(x)
def f_prime(x):
    return 1/np.cos(x)**2

def f_prime_prime(x):
    return 2*np.tan(x)/(np.cos(x)**2)

def f_prime_prime_prime(x):
    return 2*(1/np.cos(x)**2)**2 + 2*np.tan(x)*(-2*np.tan(x)/np.cos(x)**2)

def forward_diff(x,h,f):
    return (f(x+h)-f(x))/h

def central_diff(x,h,f):
    return (f(x+h)-f(x-h))/(2*h)

def h_value():
    h_values=[]
    for i in range(0,17):
        h_values.append(10**(-i))
    return h_values

def h_min_forward():
    return 2*np.sqrt(epsilon)/abs(f_prime_prime(1))

def h_min_central():
    return np.cbrt(3*epsilon/abs(f_prime_prime_prime(1)))

def errors(f,M,method):
    h_min=1
    h_values = h_value()
    num_diff_error = []
    rou_diff_error  =[]
    tru_diff_error = []
    for h in h_values:
        num_diff_error.append(abs(method(1,h,f)-f_prime(1)))
        rou_diff_error.append(2*epsilon/h)
        tru_diff_error.append(M/6*h**2)
        if(h_min>abs(method(1,h,f)-f_prime(1))):
            h_min=abs(method(1,h,f)-f_prime(1))
    return num_diff_error,rou_diff_error,tru_diff_error,h_min

def main(f):
    h_values = h_value()
    num_diff_error, rou_diff_error, tru_diff_error,h_min = errors(f, abs(f_prime_prime(1)), forward_diff)
    num_diff_error2, rou_diff_error2, tru_diff_error2 ,h_min2= errors(f, abs(f_prime_prime_prime(1)), central_diff)
    
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(h_values, num_diff_error, label='Forward Difference', marker='o')
    plt.plot(h_values, rou_diff_error, label='Central Difference', marker='o')
    plt.plot(h_values, tru_diff_error, label='True Error', marker='o')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('Absolute Error')
    plt.title('Forward Difference')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(h_values, num_diff_error2, label='Forward Difference', marker='o')
    plt.plot(h_values, rou_diff_error2, label='Central Difference', marker='o')
    plt.plot(h_values, tru_diff_error2, label='True Error', marker='o')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('Absolute Error')
    plt.title('Central Difference')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


    print(h_min_forward(),h_min_central())
    return h_min,h_min-h_min_forward(),h_min2,h_min2-h_min_central()


print(main(f))



