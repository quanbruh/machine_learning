import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LINEAR REGRESSION CLASS
# =========================
class linear_regression:
    def __init__(self):

        # y = mx + b
        # b la coefi
        # m la intercept

        self.coefi_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):                                
        X = np.array(X)
        y = np.array(y)

        # Thêm bias (cột 1)

        #np.ones(()) () ben ngoai la goi ham ben trong la tuple

        Xb = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal Equation
        A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

        self.intercept_ = A[0]
        self.coefi_ = A[1:]

    def predict(self, X):
        X = np.array(X)
        return X @ self.coefi_ + self.intercept_

if __name__=="__main__":
    # =========================
    # LOAD DATA
    # =========================
    data = pd.read_csv("housing.csv")

    # Lấy 2 feature
    x1 = np.array([1, 2, 3, 4])
    x2 = np.array([3, 7, 11, 9])            

    # Target
    y = np.array([15, 18, 23, 30])

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

    #linspace(start, end, step) de tao ca diem du lieu 

    x1_line = np.linspace(x1.min(), x1.max(), 100)
    x2_fixed = np.mean(x2)

    
    #column_stack de gop 2 ma tran thanh 2 cot

    #full_like de tranform matrix 2 giong voi matrix 1 (1, 2)

    X_line = np.column_stack((
        x1_line,
        np.full_like(x1_line, x2_fixed)
    ))

    y_line = model.predict(X_line)

    #neu co nhieu feature thi dung 

    # means = X.mean(axis=0)             

    #X.mean tinh trung binh matranj axis = 0 theo cot axis = 1 theo hang

    #len check phan tu de tao ma tran phu hop 1 

    # X_line = np.ones((len(x1_line), 4)) * means

    # X_line[:, 0] = x1_line   thay the hang 1 bang feature visualize chinh 


    
    
    # quan = np.array([[4, 8]])
    # print(model.predict(quan))


    # # =========================
    # # PLOT
    # # =========================
    plt.figure(figsize=(8, 5))
    plt.scatter(x1, y, label="cac diem data ")

    plt.plot(x1_line, y_line, color="red", label="duong mo hinh")

    plt.xlabel("median_income")
    plt.ylabel("median_house_value")
    plt.title("Linear Regression (2 features, visualized on 1 feature)")
    plt.legend()
    plt.grid(True)

    plt.show()


