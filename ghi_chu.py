import numpy as np 
import pandas as pd

data = np.array([[1], [4], [6]])

#tao matrix relevant voi shape khai bao
one = np.ones((data.shape[0], 1))

lee = np.ones((data.shape[1], 2))

#shape de xem co bao nhieu hang cot shape[0] hang shape[1] cot
cot = data.shape[1]

hang = data.shape[0]

#ghep vs nhau 
combine = np.c_[one, data]

# # print(data)
# # print(one)  
# # print(quan)

# print(combine )
# print("_________")

# print(combine[0])
# print("_________")

# print(combine[1:])


data = pd.read_csv("housing.csv")
print(data.head())

x1 = data["median_income"].values[:5]

x3 = data["median_income"]

x2 = data["housing_median_age"].values[:5]






X = np.column_stack((x1, x2))

y = data["median_house_value"]

Y = np.array(y[:5])

print(x1)

print(x2)

print(X)

print(f"output {Y}")




debug = 1