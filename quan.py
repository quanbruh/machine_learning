import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LINEAR REGRESSION CLASS
# =========================
class linear_regression:
    def __init__(self):
        self.coefi_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Thêm bias (cột 1)
        Xb = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal Equation
        A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

        self.intercept_ = A[0]
        self.coefi_ = A[1:]

    def predict(self, X):
        X = np.array(X)
        return X @ self.coefi_ + self.intercept_


# =========================
# LOAD DATA
# =========================
data = pd.read_csv("housing.csv")

# Lấy 2 feature
x1 = data["median_income"].values[:20]
x2 = data["housing_median_age"].values[:20]

# Target
y = data["median_house_value"].values[:20]

# Combine feature (CHUẨN ML)
X = np.column_stack((x1, x2))

# =========================
# TRAIN MODEL
# =========================
model = linear_regression()
model.fit(X, y)

print("Intercept (theta0):", model.intercept_)
print("Coefficients (theta1, theta2):", model.coefi_)

# =========================
# VISUALIZATION (2 feature → vẽ 1 feature)
# Giữ x2 cố định
# =========================
x1_line = np.linspace(x1.min(), x1.max(), 100)
x2_fixed = np.mean(x2)

X_line = np.column_stack((
    x1_line,
    np.full_like(x1_line, x2_fixed)
))

y_line = model.predict(X_line)




quan = np.array([[8, 52]])
print(model.predict(quan))


# =========================
# PLOT
# =========================
plt.figure(figsize=(8, 5))
plt.scatter(x1, y, label="Data points")
plt.plot(x1_line, y_line, color="red", label="Regression line")

plt.xlabel("median_income")
plt.ylabel("median_house_value")
plt.title("Linear Regression (2 features, visualized on 1 feature)")
plt.legend()
plt.grid(True)

plt.show()


