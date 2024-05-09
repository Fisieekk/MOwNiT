import numpy as np
import scipy as sp
from scipy import integrate
from scipy.misc import derivative
import matplotlib.pyplot as plt


def piIntegral(x):
    return 4 / (1 + np.power(x, 2))
def xlogIntegral(x):
    return np.sqrt(abs(x)) * np.log(abs(x))

def sumIntegral(x):
    return (1 / (((abs(x) - 0.3) ** 2) + 0.001)) + (1 / (((abs(x) - 0.9) ** 2) + 0.004)) - 6

def xlogError(y):
    return abs(y + 4 / 9) / (4 / 9)

def sumError(y):
    a = 0.001
    b = 0.004
    return abs(y - ((1 / np.sqrt(a)) * (np.arctan(0.7 / np.sqrt(a)) + np.arctan(0.3 / np.sqrt(a))) + (
            1 / np.sqrt(b)) * (np.arctan(0.1 / np.sqrt(b)) + np.arctan(0.9 / np.sqrt(b))) - 6)) / abs(
        ((1 / np.sqrt(a)) * (np.arctan(0.7 / np.sqrt(a)) + np.arctan(0.3 / np.sqrt(a))) + (
                1 / np.sqrt(b)) * (np.arctan(0.1 / np.sqrt(b)) + np.arctan(0.9 / np.sqrt(b))) - 6))

def evalErrorLab1(quadFunc, errTab, valTab, function, errFunc, xtab, nodescount, i):
    xtab[i] = nodescount
    Yval = quadFunc(function, nodescount)
    valTab[i] = Yval
    err = errFunc(Yval)
    errTab[i] = err

def evalErrorLab2(quad, errTab, function, xtab, tolerancy):
    res, err, info = sp.integrate.quad_vec(function, 0.0001, 1, epsabs=(1 / 10) ** tolerancy,quadrature=quad,full_output=True)
    errTab.append(err)
    xtab.append(info.neval)

def quadMidPoint(function, nodesCount):
    nodes = np.linspace(0, 1, nodesCount)
    sum = 0
    interval = nodes[1] - nodes[0]
    for i in range(1, len(nodes)):
        xMiddle = (nodes[i - 1] + nodes[i]) / 2
        sum += function(xMiddle) * interval
    return sum

def integalTrapez(function, nodesCount):
    nodes = np.linspace(0.0000000000000000000001, 1, nodesCount)
    fnodes = function(nodes)
    return sp.integrate.trapezoid(fnodes, nodes)

def integalSimps(function, nodesCount):
    nodes = np.linspace(0.0000000000000000000001, 1, nodesCount)
    fnodes = function(nodes)
    return sp.integrate.simps(fnodes, nodes)

def gaussLegrande(function, nodesCount):
    nodes, weightTab = np.polynomial.legendre.leggauss(nodesCount * 2)
    sum = 0
    for i in range(0, nodesCount * 2):
        sum += weightTab[i] * function(nodes[i])
    return sum / 2

# zad1
# _, _, test = sp.integrate.quad_vec(piIntegral, 0, 1, epsabs=(1 / 10) ** 15, quadrature="trapezoid", full_output=True)

gaussKronrodErr = []
trapezoidErr = []
x = np.arange(0, 15)
nevalTrap = []
resTrap = []
nevalGauss = []
resGauss = []

for i in range(15):
    res, err, info = sp.integrate.quad_vec(piIntegral, 0, 1, epsabs=(1 / 10) ** i, quadrature="gk15", full_output=True)
    gaussKronrodErr.append(err)
    nevalTrap.append(info.neval)
    resGauss.append(res)

    res, err, info = sp.integrate.quad_vec(piIntegral, 0, 1, epsabs=(1 / 10) ** i, quadrature="trapezoid",full_output=True)
    resTrap.append(res)
    trapezoidErr.append(err)
    nevalGauss.append(info.neval)

print(nevalGauss)
print(nevalTrap)

# WYKRESY DO ZAD 1

plt.yscale("log")
# plt.xscale("log")
plt.plot(x, gaussKronrodErr, label="Gauss-Kronrod")
plt.plot(x, trapezoidErr, label="AdaptiveTrapez")
plt.xlabel("n")
plt.ylabel("Wartosc błędy względnego")
plt.legend()
plt.show()
# print(resGauss)
# print(trapezoidErr.index(min(trapezoidErr)))
# print(gaussKronrodErr.index(min(gaussKronrodErr)))
# print(gaussKronrodErr[gaussKronrodErr.index(min(gaussKronrodErr))])
# print(nevalGauss[gaussKronrodErr.index(min(gaussKronrodErr))])
# print(trapezoidErr.index(min(trapezoidErr)))
# print(resTrap[trapezoidErr.index(min(trapezoidErr))])
# print(trapezoidErr[trapezoidErr.index(min(trapezoidErr))])
#
# # zad2
# n = 15
#
# tabQuad = np.zeros(n)
# tabTrapez = np.zeros(n)
# tabSimps = np.zeros(n)
# tabGauss = np.zeros(n)
# tabQuadErr = np.zeros(n)
# tabTrapezErr = np.zeros(n)
# tabSimpsErr = np.zeros(n)
# tabGaussErr = np.zeros(n)
# nevalGauss = []
# nevalTrap = []
# gaussKronrodErr = []
# trapezoidErr = []
# xScale = np.arange(0, n)
# for i in range(n):
#     print(i)
#     # logarytmiczna (zakomentuj jedną)
#     func = xlogIntegral
#     funcErr = xlogError
#
#     # ta inna (zakomentuj jedną)
#     # func = sumIntegral
#     # funcErr = sumError
#
#     evalErrorLab1(quadMidPoint, tabQuadErr, tabQuad, func, funcErr, xScale, np.power(2, i) + 1, i)
#     evalErrorLab1(integalTrapez, tabTrapezErr, tabTrapez, func, funcErr, xScale, np.power(2, i) + 1, i)
#     evalErrorLab1(integalSimps, tabSimpsErr, tabSimps, func, funcErr, xScale, np.power(2, i) + 1, i)
#     evalErrorLab1(gaussLegrande, tabGaussErr, tabGauss, func, funcErr, xScale, np.power(2, i) + 1, i)
#     if i <= 13:
#         evalErrorLab1(gaussLegrande, tabGaussErr, tabGauss, func, funcErr, xScale, np.power(2, i) + 1, i)
#     if i <= 7:
#         evalErrorLab2("trapezoid", trapezoidErr, func, nevalTrap, i)
#         evalErrorLab2("gk15", gaussKronrodErr, func, nevalGauss, i)
#
# plt.plot(xScale, tabQuadErr)
# plt.plot(xScale, tabTrapezErr)
# plt.plot(xScale, tabSimpsErr)
# plt.plot(xScale, tabGaussErr)
# plt.plot(nevalTrap, gaussKronrodErr)
# plt.plot(nevalGauss, trapezoidErr)
# plt.show()
#
# # print(tabTrapez)
# print(tabQuadErr)
# print(tabTrapezErr)
# print(tabSimpsErr)
# print(tabGaussErr)
# # print(tabSimps)
#
# print(nevalTrap)
# print(xScale)