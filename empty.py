import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class linear_regression:
    def __init__(self):
        

        self.coefi_ = None
        self.intercept_ = 0.0


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        taoso1 = np.ones((X.shape[0], 1))
        Xb = np.c_[taoso1, X]

        A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

        self.intercept_ = A[0]
        self.coefi_ = A[1:]
        


    def predict(self, x):
        return x @ self.coefi_ + self.intercept_
    

dulieu = pd.read_csv("housing.csv")

x1 = dulieu["median_income"].values[:5]

x2 = dulieu["housing_median_age"].values[:5]

x3 = dulieu["total_bedrooms"].values[:5]

x4 = dulieu["population"].values[:5]

X = np.c_[x1, x2, x3, x4]

y = dulieu["median_house_value"].values[:5]

mohinh = linear_regression()

mohinh.fit(X, y)

print(f"cac he so theta khac {mohinh.coefi_}")

print(f" theta 0 {mohinh.intercept_}")

x1_line = np.linspace(x1.min(), x1.max(), 1000)


means = X.mean(axis=0)                  # shape (4,)
X_line = np.ones((len(x1_line), 4)) * means
X_line[:, 0] = x1_line 

Y_line = mohinh.predict(X_line)

plt.scatter(x1, y)

plt.plot(x1_line, Y_line, color = 'pink')

plt.show()





        
