import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# from gradien_descent import linear_regression
from sklearn import linear_model

reg = linear_model.linear_regression()

reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

print(reg.coef_)

print(reg.intercept_)