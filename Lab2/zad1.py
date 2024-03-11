<<<<<<< Updated upstream
import numpy as np
import pandas as pd

# (a) Wczytanie danych
train_data = pd.read_csv('breast-cancer-train.dat', header=None)
validate_data = pd.read_csv('breast-cancer-validate.dat', header=None)

# Wczytanie nazw kolumn
column_names = pd.read_csv('breast-cancer.labels', header=None)
column_names = column_names.squeeze().tolist()

# Dodanie nazw kolumn do danych
train_data.columns = column_names
validate_data.columns = column_names

# (b) Histogram i wykres
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.hist(train_data['radius (mean)'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Radius (mean)')
plt.ylabel('Frequency')
plt.title('Histogram of Radius (mean)')
plt.grid(True)
plt.show()

# (c) Przygotowanie danych i reprezentacje
X_train_linear = train_data.drop(['patient ID', 'Malignant/Benign'], axis=1).values
X_train_quadratic = train_data[['radius (mean)', 'perimeter (mean)', 'area (mean)', 'symmetry (mean)']].values
y_train = np.where(train_data['Malignant/Benign'] == 'M', 1, -1)

# (d) Wektor b
y_validate = np.where(validate_data['Malignant/Benign'] == 'M', 1, -1)

# (e) Wagi metody najmniejszych kwadratów
A_linear = np.column_stack((np.ones(len(X_train_linear)), X_train_linear))
w_linear = np.linalg.solve(A_linear.T @ A_linear, A_linear.T @ y_train)

A_quadratic = np.column_stack((np.ones(len(X_train_quadratic)), X_train_quadratic))
w_quadratic = np.linalg.solve(A_quadratic.T @ A_quadratic, A_quadratic.T @ y_train)

# (f) Współczynniki uwarunkowania
cond_linear = np.linalg.cond(A_linear.T @ A_linear)
cond_quadratic = np.linalg.cond(A_quadratic.T @ A_quadratic)

# (g) Predykcja i ocena
def predict(X, w):
    return np.sign(X @ w)

predictions_linear = predict(A_linear, w_linear)
predictions_quadratic = predict(A_quadratic, w_quadratic)

# Liczba fałszywie dodatnich i fałszywie ujemnych przypadków
false_positives_linear = np.sum((predictions_linear > 0) & (y_validate == -1))
false_negatives_linear = np.sum((predictions_linear <= 0) & (y_validate == 1))

false_positives_quadratic = np.sum((predictions_quadratic > 0) & (y_validate == -1))
false_negatives_quadratic = np.sum((predictions_quadratic <= 0) & (y_validate == 1))

print("Linear Representation:")
print("False Positives:", false_positives_linear)
print("False Negatives:", false_negatives_linear)

print("\nQuadratic Representation:")
print("False Positives:", false_positives_quadratic)
print("False Negatives:", false_negatives_quadratic)
    
=======
import pandas as pd

import os

# Wczytanie danych
train_data = pd.read_csv('data\\breast-cancer-train.dat', header=None)
validate_data = pd.read_csv("data\\breast-cancer-validate.dat", header=None)

# Wczytanie nazw kolumn
column_labels = pd.read_csv("data\\breast-cancer.labels", header=None)
print(column_labels)
# Ustawienie nazw kolumn
train_data.columns = column_labels[0]
validate_data.columns = column_labels[0]

import matplotlib.pyplot as plt

# Histogram wybranej kolumny
# plt.hist(train_data['radius (mean)'], bins=20, color='blue', alpha=0.7)
# plt.xlabel('Promień (mean)')
# plt.ylabel('Liczba przypadków')
# plt.title('Histogram promienia (mean)')
# plt.show()

# # Wykres wybranej kolumny
# plt.plot(train_data['perimeter (mean)'], train_data['area (mean)'], 'o', color='green')
# plt.xlabel('Obwód (mean)')
# plt.ylabel('Powierzchnia (mean)')
# plt.title('Wykres obwodu vs powierzchni (mean)')
# plt.show()


import numpy as np

# Liniowa reprezentacja
A_linear_train = train_data.drop(['patient ID', 'Malignant/Benign'], axis=1).values
A_linear_validate = validate_data.drop(['patient ID', 'Malignant/Benign'], axis=1).values
print(A_linear_train)
print(A_linear_validate)
# Kwadratowa reprezentacja (dla wybranych cech)
selected_features = ['radius (mean)', 'perimeter (mean)', 'area (mean)', 'symmetry (mean)']
A_quad_train = train_data[selected_features].values
A_quad_train = np.column_stack((A_quad_train, A_quad_train**2))
A_quad_validate = validate_data[selected_features].values
A_quad_validate = np.column_stack((A_quad_validate, A_quad_validate**2))


# Wektor b
b_train = np.where(train_data['Malignant/Benign'] == 'M', 1, -1)
b_validate = np.where(validate_data['Malignant/Benign'] == 'M', 1, -1)


# Wagi dla reprezentacji liniowej
w_linear = np.linalg.solve(np.dot(A_linear_train.T, A_linear_train), np.dot(A_linear_train.T, b_train))

# Wagi dla reprezentacji kwadratowej
w_quad = np.linalg.solve(np.dot(A_quad_train.T, A_quad_train), np.dot(A_quad_train.T, b_train))


# Współczynnik uwarunkowania macierzy dla reprezentacji liniowej
cond_linear = np.linalg.cond(np.dot(A_linear_train.T, A_linear_train))

# Współczynnik uwarunkowania macierzy dla reprezentacji kwadratowej
cond_quad = np.linalg.cond(np.dot(A_quad_train.T, A_quad_train))


# Predykcja dla reprezentacji liniowej
p_linear = np.dot(A_linear_validate, w_linear)

# Predykcja dla reprezentacji kwadratowej
p_quad = np.dot(A_quad_validate, w_quad)

# Porównanie predykcji z wektorem b
fp_linear = np.sum(np.where(p_linear > 0, 1, 0) != np.where(b_validate > 0, 1, 0))
fn_linear = np.sum(np.where(p_linear <= 0, 1, 0) != np.where(b_validate <= 0, 1, 0))

fp_quad = np.sum(np.where(p_quad > 0, 1, 0) != np.where(b_validate > 0, 1, 0))
fn_quad = np.sum(np.where(p_quad <= 0, 1, 0) != np.where(b_validate <= 0, 1, 0))

print("Liniowa reprezentacja:")
print("Liczba fałszywie dodatnich:", fp_linear)
print("Liczba fałszywie ujemnych:", fn_linear)

print("\nKwadratowa reprezentacja:")
print("Liczba fałszywie dodatnich:", fp_quad)
print("Liczba fałszywie ujemnych:", fn_quad)
>>>>>>> Stashed changes
