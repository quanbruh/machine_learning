import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# import seaborn as sns


# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set_palette("husl")

# B1 doc housing.csv lay du lieu 


# housing_median_age  total_rooms  total_bedrooms  population  households  median_income   targe :median_house_value

df = pd.read_csv("housing.csv")

df = df[["total_rooms", "median_income", "median_house_value"]]

df = df.iloc[:1000]

# phan tich truc quan hoa du lieu 

#B2 phan tich truc quan


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#1 thu nhap trung binh voi gia nha 

axes[0].scatter(df['median_income'], df["median_house_value"])
axes[0].set_title("thu nhap trung binh gia nha")
axes[0].set_xlabel("thu nhap trung binh")
axes[0].set_ylabel("gia nha")


#2total room voi gia nha

axes[1].scatter(df["total_rooms"], df["median_house_value"])
axes[1].set_title("so phong vs gia nha")
axes[1].set_xlabel("so phong")
axes[1].set_ylabel("gia nha")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# --- BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU ---
print("\n=== BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU ===")

print("Kiểm tra missing values:")
print(df.isnull().sum())

Q1 = df["median_house_value"].quantile(0.25)
Q3 = df["median_house_value"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["median_house_value"] < lower_bound) | (df["median_house_value"] > upper_bound)]
print(f"so diem bat thuong {len(outliers)}")

features = ["median_income", "total_rooms"]

X = df[features]

y = df["median_house_value"]

print(f"Features sử dụng: {features}")
print(f"Số features: {len(features)}")


# --- BƯỚC 4: CHIA DỮ LIỆU VÀ CHUẨN HÓA ---
print("\n=== BƯỚC 4: CHIA DỮ LIỆU VÀ CHUẨN HÓA ===")

#chia du lieu 80 train 20 test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Chia dữ liệu:")
print(f"- Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"- Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)


y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()



# --- BƯỚC 5: HUẤN LUYỆN MÔ HÌNH ---
print("\n=== BƯỚC 5: HUẤN LUYỆN MÔ HÌNH ===")


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"\nĐang huấn luyện {name}...")
    
    # Huấn luyện
    model.fit(X_train_scaled, y_train_scaled)
    
    # Dự đoán
    y_pred_scaled = model.predict(X_test_scaled)
    # Chuyển ngược về đơn vị gốc (kW)

    #reshape(-1 , so cot ).ravel de ep ve mang 1 chieu vi [[1], [2], [3]] thanh [1, 2, 3]

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Lưu predictions
    predictions[name] = y_pred
    
    # Tính metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    print(f"  MAE: {mae:.2f} kW")
    print(f"  RMSE: {rmse:.2f} kW")
    print(f"  R2: {r2:.4f}")






# --- BƯỚC 6: SO SÁNH MÔ HÌNH ---
print("\n=== BƯỚC 6: SO SÁNH MÔ HÌNH ===")

# Tạo DataFrame kết quả
results_df = pd.DataFrame(results).T
print("\nSo sánh hiệu suất các mô hình:")
print(results_df.round(3))

# Tìm mô hình tốt nhất dựa trên R2
best_model_name = results_df['R2'].idxmax()
best_result = results_df.loc[best_model_name]

print(f"\n⭐ Mô hình tốt nhất: {best_model_name}")
print(f"  R²: {best_result['R2']:.4f}")
print(f"  MAE: {best_result['MAE']:.2f} kW")
print(f"  Sai số tương đối: {(best_result['MAE']/np.mean(y_test))*100:.2f}%")


print("\n=== BƯỚC 7: TRỰC QUAN HÓA KẾT QUẢ ===")

fig, axes = plt.subplots(2, 2, figsize=(9, 5))
models_names = list(results.keys())
y_pred_best = predictions[best_model_name]

# 1. So sánh dự đoán vs thực tế (mô hình tốt nhất)
axes[0][0].scatter(range(len(y_test[:100])), y_test[:100], label='Thực tế', marker='o', s=40)
axes[0][0].plot(range(len(y_test[:100])), y_pred_best[:100], label='Dự đoán', color='r', linestyle='--', marker='s', markersize=4)
axes[0][0].set_title(f'So sánh dự đoán và thực tế\n({best_model_name})')
axes[0][0].set_xlabel('Mẫu thử nghiệm')
axes[0][0].set_ylabel('gia nha')
axes[0][0].legend()
axes[0][0].legend()
axes[0][0].grid(True)


r2_scores = [results[m]['R2'] for m in models_names]
bars = axes[1][0].bar(models_names, r2_scores, color=['blue', 'green', 'orange', 'red'])
axes[1][0].set_title('So sánh R² Score')
axes[1][0].set_ylabel('R² Score')
axes[1][0].set_ylim(0, 1.1)
axes[1][0].tick_params(axis='x', rotation=45)


mae_scores = [results[m]['MAE'] for m in models_names]
bars = axes[0][1].bar(models_names, mae_scores, color=['blue', 'green', 'orange', 'red'])
axes[0][1].set_title('So sánh MAE (Mean Absolute Error)')
axes[0][1].set_ylabel('MAE (kW)')
axes[0][1].tick_params(axis='x', rotation=45)

for bar in bars:
    height = bar.get_height()
    axes[0][1].text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', ha='center', va='bottom')




axes[1, 1].scatter(y_test, y_pred_best, alpha=0.5)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Lý tưởng')
axes[1, 1].set_title(f'Dự đoán vs Thực tế\n({best_model_name})')
axes[1, 1].set_xlabel('Thực tế (kW)')
axes[1, 1].set_ylabel('Dự đoán (kW)')
axes[1, 1].legend()
axes[1, 1].grid(True)


plt.show()











debug = 1