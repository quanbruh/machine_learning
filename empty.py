import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class ridge_regression:
    def __init__(self, landa):
        self.coefi_ = None
        self.landa = landa
        self.intercept_ = 0.0

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        m, n = X.shape

        I = np.eye(n + 1)
        I[0, 0] = 0


        Xb = np.column_stack((np.ones(m), X))

        A = np.linalg.inv(Xb.T @ Xb + self.landa * I) @ Xb.T @ y

        self.coefi_ = A[1:]
        self.intercept_ = A[0] 

    def predict(self, x):
        return x @ self.coefi_ + self.intercept_

df = pd.read_csv("housing.csv")
x1 = df["housing_median_age"].values[:9]
x2 = df["median_income"].values[:9]
y = df["median_house_value"].values[:9]
X = np.column_stack((x1, x2))

model = ridge_regression(0.01)
quan = model.fit(X, y)
