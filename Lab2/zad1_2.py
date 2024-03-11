

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (a) Otwórz zbiory danych
column_names = []

with open("data\\breast-cancer.labels") as f:
    for line in f.readlines():
        column_names.append(line.strip())

breast_cancer_train = pd.io.parsers.read_csv("data\\breast-cancer-train.dat", header=None, names=column_names)
breast_cancer_validate = pd.io.parsers.read_csv("data\\breast-cancer-validate.dat", header=None, names=column_names)

# (b) Stwórz histogram i wykres wybranej kolumny danych
# plt.hist(breast_cancer_train["radius (mean)"])
# plt.ylabel("Frequency")
# plt.xlabel("Radius (mean)")
# plt.show()

# plt.plot(breast_cancer_train["radius (mean)"])
# plt.title("Patients' radius (mean)")
# plt.xlabel("Patient")
# plt.ylabel("Radius (mean)")
# plt.show()
selected_columns = ['radius (mean)', 'perimeter (mean)', 'area (mean)', 'symmetry (mean)']

breast_cancer_train_linear_matrix = breast_cancer_train.map(lambda x: 1 if x == 'M' else (-1 if x == 'B' else x)).to_numpy()
breast_cancer_train_linear_matrix = breast_cancer_validate.map(lambda x: 1 if x == 'M' else (-1 if x == 'B' else x)).to_numpy()
breast_cancer_train_square_matrix = breast_cancer_train[selected_columns].map(lambda x: 1 if x == 'M' else (-1 if x == 'B' else x)).apply(np.square).to_numpy()
breast_cancer_train_square_matrix = breast_cancer_validate[selected_columns].map(lambda x: 1 if x == 'M' else (-1 if x == 'B' else x)).apply(np.square).to_numpy()
b_train = np.where(breast_cancer_train['Malignant/Benign'] == 'M', 1, -1)
b_validate = np.where(breast_cancer_validate['Malignant/Benign'] == 'M', 1, -1)

A_linear = np.hstack((np.ones((len(b_train), 1)), breast_cancer_train_linear_matrix))

w_linear = np.linalg.solve(A_linear.T @ A_linear, A_linear.T @ b_train)

A_square = np.hstack((np.ones((len(b_train), 1)), breast_cancer_train_square_matrix, breast_cancer_train_square_matrix))

w_square = np.linalg.solve(A_square.T @ A_square, A_square.T @ b_train)

print("Wagi dla reprezentacji liniowej:", w_linear)
print("\nWagi dla reprezentacji kwadratowej:", w_square)
