import numpy as np
import pandas as pd 
from datetime import datetime, timedelta

power_data={
    'timestamp':pd.date_range('2021-5-4', periods = 10, freq ='min'),




    'power_kwh':[100 + 50 * (i % 24)/24 + np.random.normal(0, 10) for i in range(10)]
}

temp_data ={
    'timestamp':pd.date_range('2024-01-01', periods = 10, freq='h'),
    'temperature':[25 + 10 * np.sin(2*np.pi*i/24) + np.random.normal(0,2) for i in range(5)]
}

print(power_data)
# print(temp_data)

