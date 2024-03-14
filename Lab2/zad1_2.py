import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (a) Otwórz zbiory danych
column_names = []

with open("data\\breast-cancer.labels") as f:
    for line in f.readlines():
        column_names.append(line.strip())

breast_cancer_train = pd.io.parsers.read_csv(
    "data\\breast-cancer-train.dat", header=None, names=column_names)
breast_cancer_validate = pd.io.parsers.read_csv(
    "data\\breast-cancer-validate.dat", header=None, names=column_names)

# (b) Stwórz histogram i wykres wybranej kolumny danych
plt.hist(breast_cancer_train["radius (mean)"],
         bins=20, color='blue', alpha=0.7)
plt.ylabel("Frequency")
plt.xlabel("Radius (mean)")
plt.show()

plt.plot(breast_cancer_train["radius (mean)"])
plt.title("Patients' radius (mean)")
plt.xlabel("Patient")
plt.ylabel("Radius (mean)")
plt.show()

# (c) Stwórz reprezentacje danych
# Train
# Liniowa
A_lin_train = breast_cancer_train.drop("Malignant/Benign", axis=1).values
A_lin_train = np.column_stack((np.ones(A_lin_train.shape[0]), A_lin_train))
# Kwadratowa
A_quad_train = breast_cancer_train[[
    "radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]].values
A_quad_train = np.column_stack(
    (np.ones(A_quad_train.shape[0]), A_quad_train, A_quad_train**2))

# Validate
# Liniowa
A_lin_validate = breast_cancer_validate.drop("Malignant/Benign", axis=1).values
A_lin_validate = np.column_stack(
    (np.ones(A_lin_validate.shape[0]), A_lin_validate))
# Kwadratowa
A_quad_validate = breast_cancer_validate[[
    "radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]].values
A_quad_validate = np.column_stack(
    (np.ones(A_quad_validate.shape[0]), A_quad_validate, A_quad_validate**2))

# (d) Stwórz wektor b
b_train = np.where(breast_cancer_train["Malignant/Benign"] == "M", 1, -1)
b_validate = np.where(breast_cancer_validate["Malignant/Benign"] == "M", 1, -1)

# (e) Znajdź wagi
# Liniowa
w_lin = np.linalg.solve(A_lin_train.T @ A_lin_train, A_lin_train.T @ b_train)
# Kwadratowa
w_quad = np.linalg.solve(A_quad_train.T @ A_quad_train,
                         A_quad_train.T @ b_train)
# # (f) Oblicz współczynniki uwarunkowania
cond_lin = (np.linalg.cond(A_lin_train))**2
cond_quad = (np.linalg.cond(A_quad_train))**2
print(cond_lin)
print(cond_quad)

# (g) Sprawdź jak dobrze otrzymane wagi przewidują typ nowotworu
# Liniowa
p_lin = A_lin_validate @ w_lin
p_lin = np.where(p_lin > 0, 1, -1)

# Kwadratowa
p_quad = A_quad_validate @ w_quad
p_quad = np.where(p_quad > 0, 1, -1)

# # Porównaj wektory p
# print("Liniowa")
# print(p_lin)
# print(b_validate)
# print("Kwadratowa")
# print(p_quad)
# print(b_validate)

# # Oblicz liczbę fałszywie dodatnich oraz fałszywie ujemnych przypadków
# # Liniowa
false_positives_lin = np.sum(np.logical_and(p_lin > 0, b_validate < 0))
false_negatives_lin = np.sum(np.logical_and(p_lin < 0, b_validate > 0))

# Kwadratowa
false_positives_quad = np.sum(np.logical_and(p_quad > 0, b_validate < 0))
false_negatives_quad = np.sum(np.logical_and(p_quad < 0, b_validate > 0))

print("Liniowa")
print("False positives:", false_positives_lin)
print("False negatives:", false_negatives_lin)
print("Kwadratowa")
print("False positives:", false_positives_quad)
print("False negatives:", false_negatives_quad)
