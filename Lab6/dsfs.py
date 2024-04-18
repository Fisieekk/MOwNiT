import numpy as np
import scipy as sp
from scipy import integrate
from scipy.misc import derivative
import matplotlib.pyplot as plt


def piIntegral(x):
    return 4 / (1 + np.power(x, 2))


def quadMidPoint(section, function, nodesCount):
    nodes = np.linspace(section[0], section[1], nodesCount)
    sum = 0
    interval = nodes[1] - nodes[0]
    for i in range(1, len(nodes)):
        xMiddle = (nodes[i - 1] + nodes[i]) / 2
        sum += function(xMiddle) * interval
    return sum


def integalTrapez(section, function, nodesCount):
    nodes = np.linspace(section[0], section[1], nodesCount)
    fnodes = function(nodes)
    return sp.integrate.trapz(fnodes, nodes)


def integalSimps(section, function, nodesCount):
    nodes = np.linspace(section[0], section[1], nodesCount)
    fnodes = function(nodes)
    return sp.integrate.simps(fnodes, nodes)


def gaussLegrande(function, nodesCount):
    nodes, weightTab = np.polynomial.legendre.leggauss(nodesCount * 2)
    sum = 0
    for i in range(0, nodesCount * 2):
        sum += weightTab[i] * function(nodes[i])
    return sum / 2


def orderOfConvergence(tab, h1, h2):
    H1 = np.linspace(0, 1, np.power(2, h1) + 1)[1]
    H2 = np.linspace(0, 1, np.power(2, h2) + 1)[1]

    return np.log(tab[h2] / tab[h1]) / np.log(H2 / H1)


# def errors(h):
#     tabErrors = []
#     tabErrors.append((sp.misc.derivative(piIntegral, 0.5, dx=1e-6, n=2) * np.power(h, 3)) / 24)
#     tabErrors.append((sp.misc.derivative(piIntegral, 0.5, dx=1e-6, n=2) * np.power(h, 3)) / 12)
#     tabErrors.append((sp.misc.derivative(piIntegral, 0.5, dx=1e-6, n=4) * np.power(h, 3)) / 90)
#     return tabErrors


# no dla 25 się strasznie długo robi i to nawet nie moja wina jak złożoność jest wykładnicza xd
n = 10

tabQuad = np.zeros(n)
tabTrapez = np.zeros(n)
tabSimps = np.zeros(n)
tabGauss = np.zeros(n)
tabQuadErr = np.zeros(n)
tabTrapezErr = np.zeros(n)
tabSimpsErr = np.zeros(n)
tabGaussErr = np.zeros(n)
for i in range(n):
    print(i)
    tabQuad[i] = quadMidPoint((0, 1), piIntegral, np.power(2, i) + 1)
    tabTrapez[i] = integalTrapez((0, 1), piIntegral, np.power(2, i) + 1)
    tabSimps[i] = integalSimps((0, 1), piIntegral, np.power(2, i) + 1)
    tabQuadErr[i] = abs(np.pi - tabQuad[i]) / np.pi
    tabTrapezErr[i] = abs(np.pi - tabTrapez[i]) / np.pi
    tabSimpsErr[i] = abs(np.pi - tabSimps[i]) / np.pi
    # zad2
    tabGauss[i] = gaussLegrande(piIntegral, np.power(2, i) + 1)
    tabGaussErr[i] = abs(np.pi - tabGauss[i]) / np.pi
    # koniec zad2 xd
print(tabQuad)
print(tabTrapez)
print(tabSimps)
print(tabQuadErr)
print(tabTrapezErr)
print(tabSimpsErr)

xScale = np.arange(0, n)
plt.yscale("log")
plt.xscale("log")
plt.plot(xScale, tabQuadErr, 'ro', markersize=5)
plt.plot(xScale, tabTrapezErr, 'bo', markersize=5)
plt.plot(xScale, tabSimpsErr, 'go', markersize=5)
plt.plot(xScale, tabGaussErr, 'yo', markersize=5)
plt.show()

# zad1b

# dla tangensa to było w labie 1 ale spoko XD
print("lab 1 ", 6.2230220976289274e-12)
print("Quad hmin", min(tabQuadErr))
print("Trapez hmin", min(tabTrapezErr))
print("Simps hmin", min(tabSimpsErr))

# to jest krok dla któregho wartość kwadratury nie poprawia się
print(np.where(tabQuadErr == min(tabQuadErr))[0][0])
print(np.where(tabTrapezErr == min(tabTrapezErr))[0][0])
print(np.where(tabSimpsErr == min(tabSimpsErr))[0][0])

# zad1c
# my se weźmiemy h3 i h4 bo czm nie

# nie zgadza się ( patz na intro do wykładu i rate of convergence) 1 iteracja zamiast zgodnie z teorią 2
print(orderOfConvergence(tabQuadErr, 3, 4))
# zgadza się 2 iteracja zgodnie z teorią
print(orderOfConvergence(tabTrapezErr, 3, 4))
# nie zgadza się 3 iteracja z terią zgodna jest 4
print(orderOfConvergence(tabSimpsErr, 3, 4))

# ten wynik wyznacza nam rząd do którego powinno zbiegać a w której iteracji to mówi nam rate of covergence
# możliwe też że jak zbiega szybciej to dobrze


# zad2

print(tabGauss)
print(tabGaussErr)