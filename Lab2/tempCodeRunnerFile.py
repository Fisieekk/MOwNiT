import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
column_names = []

with open("data\\breast-cancer.labels") as f:
    for line in f.readlines():
        column_names.append(line[:(len(line))-1])
breast_cancer_train = pd.io.parsers.read_csv("data\\breast-cancer-train.dat")
breast_cancer_train.columns = column_names

breast_cancer_validate =  pd.io.parsers.read_csv("data\\breast-cancer-validate.dat")
breast_cancer_validate.columns = column_names
plt.hist(breast_cancer_train["radius (mean)"])
plt.ylabel("radius (mean)")
plt.show()
plt.plot(breast_cancer_train["radius (mean)"])
plt.title("patients radius (mean)")
plt.xlabel("patient")
plt.ylabel("radius (mean)")
plt.show()
breast_cancer_train_linear_matrix = np.matrix(breast_cancer_train)
breast_cancer_validate_linear_matrix = np.matrix(breast_cancer_validate)

breast_cancer_train_square_matrix = np.zeros([299,16])
breast_cancer_validate_square_matrix = np.zeros([299,16])