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
    