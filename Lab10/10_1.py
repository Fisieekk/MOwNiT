# Równania różniczkowe - spectral bias
# Zadanie 1. Dane jest równanie różniczkowe zwyczajne
# du(x)/dx= cos(ωx) dla x ∈ Ω , (1)
# gdzie:
# x, ω, u ∈ R,
# x to położenie,
# Ω to dziedzina, na której rozwiązujemy równanie, Ω = { x | −2π ≤ x ≤ 2π }.
# u(·) to funkcja, której postaci szukamy.
# Warunek początkowy zdefiniowany jest następująco:
# u(0) = 0 
# Analityczna postać rozwiązania równania z warunkiem początkowym
# jest następująca:
# u(x) = (1/ω)*sin(ωx). (3)
# Rozwiąż powyższe zagadnienie początkowe Do rozwiązania użyj sieci neuronowych PINN (ang. Physics-informed Neural Network ) [1]. Można wykorzystać szablon w pytorch-u lub bibliotekę DeepXDE [2].
# Koszt rezydualny zdefiniowany jest następująco:
# Lr(θ) = (1/N)*suma(||duˆ(x)/dx − cos(ωx)||^2), do N, i=1
# gdzie N jest liczbą punktów kolokacyjnych.
# Koszt związany z warunkiem początkowym przyjmuje postać:
# LIC (θ) = ||uˆ(0) − 0||2
# Funkcja kosztu zdefiniowana jest następująco:
# L(θ) = Lr(θ) + LIC (θ).
# Warstwa wejściowa sieci posiada 1 neuron, reprezentujący zmienną x, Warstwa
# wyjściowa także posiada 1 neuron, reprezentujący zmienną uˆ(x). Uczenie trwa
# przez 50 000 kroków algorytmem Adam ze stałą uczenia równą 0.001. Jako funkcję aktywacji przyjmij tangens hiperboliczny, tanh.
# (a) Przypadek ω = 1.
# Ustal następujące wartości:
# – 2 warstwy ukryte, 16 neuronów w każdej warstwie
# – liczba punktów treningowych: 200
# – liczba punktów testowych: 1000
# (b) Przypadek ω = 15.
# Ustal następujące wartości:
# – liczba punktów treningowych: 200 · 15 = 3000
# – liczba punktów testowych: 5000
# Eksperymenty przeprowadź z trzema architekturami sieci:
# – 2 warstwy ukryte, 16 neuronów w każdej warstwie
# – 4 warstwy ukryte, 64 neurony w każdej warstwie
# – 5 warstw ukrytych, 128 neuronów w każdej warstwie
# (c) Dla wybranej przez siebie sieci porównaj wynik z rozwiązaniem, w którym
# przyjęto, że szukane rozwiązanie (ansatz ) ma postać:
# uˆ(x; θ) = tanh(ωx) · NN (x; θ)
# Taka postać rozwiązania gwarantuje spełnienie warunku uˆ(0) = 0 bez wprowadzania składnika LIC do funkcji kosztu.
# (d) Porównaj pierwotny wynik z rozwiązaniem, w którym pierwszą warstwę
# ukrytą zainicjalizowano cechami Fouriera:
# γ(x) = [ sin(20πx), cos(20πx), . . . ,sin(2^(L−1)πx), cos(2^(L−1)πx) ] . (8)
# Dobierz L tak, aby nie zmieniać szerokości warstwy ukrytej.
# Dla każdego z powyższych przypadków stwórz następujące wykresy:
# – Wykres funkcji u(x), tj. dokładnego rozwiązania oraz wykres funkcji uˆ(x),
# tj. rozwiązania znalezionego przez sieć neuronową
# – Wykres funkcji błędu.
# Stwórz także wykres funkcji kosztu w zależności od liczby epok.
# Uwaga. W przypadku wykorzystania biblioteki DeepXDE i backendu tensorflow
# należy użyć wersji tensorflow v1.


import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

# Definicja funkcji kosztu

def loss_fn(u_hat, du_hat, x, omega):