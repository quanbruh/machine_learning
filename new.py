import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# Thiết lập style cho biểu đồ
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- BƯỚC 1: TẠO DỮ LIỆU MÔ PHỎNG ---
print("=== BƯỚC 1: TẠO DỮ LIỆU MÔ PHỎNG ===")

np.random.seed(42)
n_samples = 720 # 30 ngày * 24 giờ

# Tạo timestamp mỗi giờ
timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='H')

# Tạo features thời gian
hours = np.array([ts.hour for ts in timestamps])
day_of_week = np.array([ts.weekday() for ts in timestamps])
day_of_year = np.array([ts.dayofyear for ts in timestamps])

# Nhiệt độ: có tính thời vụ và ngày/đêm
base_temp = 25
temp_variation = 10 * np.sin(2 * np.pi * hours / 24) # Ngày/đêm
seasonal_variation = 8 * np.sin(2 * np.pi * day_of_year / 365) # Mùa
temperatures = base_temp + temp_variation + seasonal_variation + np.random.normal(0, 2, n_samples)

# Tốc độ sản xuất: cao ban ngày, thấp ban đêm
production_day = np.random.poisson(60, n_samples) # Ban ngày cao
production_night = np.random.poisson(20, n_samples) # Ban đêm thấp
production_rates = np.where(hours < 18, production_day, production_night)

# Số máy hoạt động: ít hơn vào cuối tuần
base_machines = 50
weekend_factor = np.where(day_of_week >= 5, 0.3, 1.0) # Cuối tuần chỉ 30% máy chạy
active_machines = (base_machines * weekend_factor).astype(int)

# Điện năng tiêu thụ (Mục tiêu cần dự đoán)
# Công thức: Điện năng = f(nhiệt độ, sản xuất, số máy, giờ)
base_power = 1000 # kW
power_from_temp = 20 * (temperatures - 25) # Nhiệt độ cao -> dùng nhiều điều hòa
power_from_production = 5 * production_rates
power_from_machines = 10 * active_machines
hourly_pattern = 100 * np.sin(2 * np.pi * (hours - 6) / 24) # Cao điểm ban ngày

power_consumption = (
    base_power + 
    power_from_temp + 
    power_from_production + 
    power_from_machines + 
    hourly_pattern + 
    np.random.normal(0, 50, n_samples) # Nhiễu
)

# Tạo DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'hour': hours,
    'day_of_week': day_of_week,
    'temperature': temperatures,
    'production_rate': production_rates,
    'active_machines': active_machines,
    'power_consumption': power_consumption
})

# Thêm các features phái sinh
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

print("Dữ liệu đã tạo:")
print(df.head())
print(f"\nKích thước: {df.shape}")
print("\nMô tả thống kê:")
print(df.describe().round(2))

# --- BƯỚC 2: PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU ---
print("\n=== BƯỚC 2: PHÂN TÍCH DỮ LIỆU ===")

fig, axes = plt.subplots(2, 3, figsize=(9, 5))

# 1. Điện năng theo thời gian
axes[0, 0].plot(df['timestamp'], df['power_consumption'], linewidth=1)
axes[0, 0].set_title('Điện năng tiêu thụ theo thời gian')
axes[0, 0].set_ylabel('Power (kW)')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Nhiệt độ vs Điện năng
scatter = axes[0, 1].scatter(df['temperature'], df['power_consumption'], c=df['hour'], cmap='viridis', alpha=0.6)
axes[0, 1].set_title('Nhiệt độ vs Điện năng')
axes[0, 1].set_xlabel('Temperature (°C)')
axes[0, 1].set_ylabel('Power (kW)')
plt.colorbar(scatter, ax=axes[0, 1], label='Giờ trong ngày')

# 3. Sản xuất vs Điện năng
axes[0, 2].scatter(df['production_rate'], df['power_consumption'], alpha=0.6)
axes[0, 2].set_title('Tốc độ sản xuất vs Điện năng')
axes[0, 2].set_xlabel('Production Rate')
axes[0, 2].set_ylabel('Power (kW)')

# 4. Phân bổ điện năng
axes[1, 0].hist(df['power_consumption'], bins=50, edgecolor='black')
axes[1, 0].set_title('Phân bố điện năng tiêu thụ')
axes[1, 0].set_xlabel('Power (kW)')
axes[1, 0].set_ylabel('Frequency')

# 5. Correlation matrix
corr_cols = ['temperature', 'production_rate', 'active_machines', 'power_consumption', 'is_daytime']
corr_matrix = df[corr_cols].corr()
im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 1].set_title('Ma trận tương quan')
axes[1, 1].set_xticks(range(len(corr_cols)))
axes[1, 1].set_xticklabels(corr_cols, rotation=45)
axes[1, 1].set_yticks(range(len(corr_cols)))
axes[1, 1].set_yticklabels(corr_cols)

# Thêm giá trị lên ô của ma trận tương quan
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = axes[1, 1].text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black", fontsize=8)

# 6. Điện năng trung bình theo giờ
hourly_avg = df.groupby('hour')['power_consumption'].mean()
axes[1, 2].plot(hourly_avg.index, hourly_avg.values, marker='o')
axes[1, 2].set_title('Điện năng trung bình theo giờ')
axes[1, 2].set_xlabel('Giờ trong ngày')
axes[1, 2].set_ylabel('Power (kW)')
axes[1, 2].set_xticks(range(0, 24, 3))

plt.tight_layout()
plt.show()

print("\nNhận xét từ phân tích:")
print("1. Điện năng có tính chu kỳ ngày/đêm rõ rệt")
print("2. Tương quan mạnh với nhiệt độ và sản xuất")
print("3. Cao điểm vào ban ngày, thấp điểm ban đêm")


#######################################################


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU ---
print("\n=== BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU ===")

# Kiểm tra missing values
print("Kiểm tra missing values:")
print(df.isnull().sum())

# Kiểm tra outliers bằng IQR cho power_consumption
Q1 = df['power_consumption'].quantile(0.25)
Q3 = df['power_consumption'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['power_consumption'] < lower_bound) | (df['power_consumption'] > upper_bound)]
print(f"Số outliers phát hiện: {len(outliers)}")

# Chuẩn bị features và target
features = ['temperature', 'production_rate', 'active_machines', 
            'is_weekend', 'is_daytime', 'hour_sin', 'hour_cos']
X = df[features]
y = df['power_consumption']

print(f"Features sử dụng: {features}")
print(f"Số features: {len(features)}")

# --- BƯỚC 4: CHIA DỮ LIỆU VÀ CHUẨN HÓA ---
print("\n=== BƯỚC 4: CHIA DỮ LIỆU VÀ CHUẨN HÓA ===")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print(f"Chia dữ liệu:")
print(f"- Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"- Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Chuẩn hóa dữ liệu
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Reshape y để phù hợp với scaler (cần mảng 2 chiều)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# --- BƯỚC 5: HUẤN LUYỆN MÔ HÌNH ---
print("\n=== BƯỚC 5: HUẤN LUYỆN MÔ HÌNH ===")

# Danh sách mô hình
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# Huấn luyện và đánh giá từng mô hình
results = {}
predictions = {}

for name, model in models.items():
    print(f"\nĐang huấn luyện {name}...")
    
    # Huấn luyện
    model.fit(X_train_scaled, y_train_scaled)
    
    # Dự đoán
    y_pred_scaled = model.predict(X_test_scaled)
    # Chuyển ngược về đơn vị gốc (kW)
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


#########################################



# --- BƯỚC 7: TRỰC QUAN HÓA KẾT QUẢ ---
print("\n=== BƯỚC 7: TRỰC QUAN HÓA KẾT QUẢ ===")

fig, axes = plt.subplots(2, 3, figsize=(9, 5))
models_names = list(results.keys())
y_pred_best = predictions[best_model_name]

# 1. So sánh dự đoán vs thực tế (mô hình tốt nhất)
axes[0, 0].scatter(range(len(y_test[:100])), y_test[:100], label='Thực tế', marker='o', s=40)
axes[0, 0].plot(range(len(y_test[:100])), y_pred_best[:100], label='Dự đoán', color='r', linestyle='--', marker='s', markersize=4)
axes[0, 0].set_title(f'So sánh dự đoán và thực tế\n({best_model_name})')
axes[0, 0].set_xlabel('Mẫu thử nghiệm')
axes[0, 0].set_ylabel('Power (kW)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Scatter plot dự đoán vs thực tế
axes[0, 1].scatter(y_test, y_pred_best, alpha=0.5)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Lý tưởng')
axes[0, 1].set_title(f'Dự đoán vs Thực tế\n({best_model_name})')
axes[0, 1].set_xlabel('Thực tế (kW)')
axes[0, 1].set_ylabel('Dự đoán (kW)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Phân bổ sai số
error = y_pred_best - y_test
axes[0, 2].hist(error, bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 2].axvline(x=error.mean(), color='g', linestyle='--', label=f'Mean: {error.mean():.2f}')
axes[0, 2].set_title('Phân bổ sai số')
axes[0, 2].set_xlabel('Sai số (kW)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].legend()

# 4. So sánh R2 của các mô hình
r2_scores = [results[m]['R2'] for m in models_names]
bars = axes[1, 0].bar(models_names, r2_scores, color=['blue', 'green', 'orange', 'red'])
axes[1, 0].set_title('So sánh R² Score')
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].tick_params(axis='x', rotation=45)

# Thêm giá trị lên đầu cột
for bar in bars:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom')

# 5. So sánh MAE
mae_scores = [results[m]['MAE'] for m in models_names]
bars = axes[1, 1].bar(models_names, mae_scores, color=['blue', 'green', 'orange', 'red'])
axes[1, 1].set_title('So sánh MAE (Mean Absolute Error)')
axes[1, 1].set_ylabel('MAE (kW)')
axes[1, 1].tick_params(axis='x', rotation=45)

for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', ha='center', va='bottom')

# 6. Feature Importance (nếu có)
best_model = models[best_model_name]
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=True)
    axes[1, 2].barh(feat_imp['feature'], feat_imp['importance'])
    axes[1, 2].set_title(f'Độ quan trọng tính năng\n({best_model_name})')
elif hasattr(best_model, 'coef_'):
    coef = np.abs(best_model.coef_)
    feat_imp = pd.DataFrame({'feature': features, 'coefficient': coef}).sort_values('coefficient', ascending=True)
    axes[1, 2].barh(feat_imp['feature'], feat_imp['coefficient'])
    axes[1, 2].set_title(f'Hệ số tính năng\n({best_model_name})')
else:
    axes[1, 2].text(0.5, 0.5, 'Không có feature importance\ncho mô hình này', ha='center', va='center')

plt.tight_layout()
plt.show()

# --- BƯỚC 8: DỰ ĐOÁN TRÊN DỮ LIỆU MỚI ---
print("\n=== BƯỚC 8: DỰ ĐOÁN CHO DỮ LIỆU MỚI ===")

new_data = pd.DataFrame({
    'temperature': [28, 30, 25, 32],
    'production_rate': [55, 60, 45, 65],
    'active_machines': [45, 50, 30, 50],
    'is_weekend': [0, 0, 1, 0],
    'is_daytime': [1, 1, 0, 1],
    'hour_sin': [np.sin(2*np.pi*10/24), np.sin(2*np.pi*14/24), np.sin(2*np.pi*20/24), np.sin(2*np.pi*16/24)],
    'hour_cos': [np.cos(2*np.pi*10/24), np.cos(2*np.pi*14/24), np.cos(2*np.pi*20/24), np.cos(2*np.pi*16/24)]
})

print("Dữ liệu mới cần dự đoán:")
print(new_data)

# Chuẩn hóa và dự đoán
new_data_scaled = scaler_X.transform(new_data)
new_predictions_scaled = best_model.predict(new_data_scaled)
new_predictions = scaler_y.inverse_transform(new_predictions_scaled.reshape(-1, 1)).ravel()

print("\nKết quả dự đoán điện năng:")
for i, pred in enumerate(new_predictions):
    conditions = []
    if new_data.loc[i, 'is_weekend'] == 1: conditions.append("cuối tuần")
    if new_data.loc[i, 'is_daytime'] == 1: conditions.append("ban ngày")
    else: conditions.append("ban đêm")
    
    # Tính lại giờ từ sin/cos (xấp xỉ)
    hour = int(np.arctan2(new_data.loc[i, 'hour_sin'], new_data.loc[i, 'hour_cos']) * 24 / (2 * np.pi))
    if hour < 0: hour += 24
    
    print(f"Scenario {i+1}: {pred:.1f} kW (Nhiệt độ {new_data.loc[i, 'temperature']}°C, "
          f"Sản xuất {new_data.loc[i, 'production_rate']}, {', '.join(conditions)}, Giờ {hour}:00)")
    



#####################################

import joblib
import os

# --- BƯỚC 9: TRIỂN KHAI ĐƠN GIẢN ---
print("\n=== BƯỚC 9: GỢI Ý TRIỂN KHAI ===")

# Lưu mô hình và scaler
# Tạo thư mục lưu model
os.makedirs('model', exist_ok=True)

# Lưu các thành phần quan trọng
joblib.dump(models[best_model_name], 'model/power_predictor.pkl')
joblib.dump(scaler_X, 'model/scaler_X.pkl')
joblib.dump(scaler_y, 'model/scaler_y.pkl')

print("Đã lưu mô hình và scaler vào thư mục 'model/'")

# --- Tạo file inference mẫu ---
inference_code = f"""
# inference.py - Mẫu code để sử dụng mô hình đã huấn luyện
import joblib
import numpy as np
import pandas as pd

class PowerPredictor:
    def __init__(self, 
                 model_path='model/power_predictor.pkl',
                 scaler_x_path='model/scaler_X.pkl',
                 scaler_y_path='model/scaler_y.pkl'):
        self.model = joblib.load(model_path)
        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
    def predict(self, temperature, production_rate, active_machines, is_weekend, is_daytime, hour):
        # Chuẩn bị features (tính toán lại sin/cos của giờ)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        features = [[temperature, production_rate, active_machines, 
                     is_weekend, is_daytime, hour_sin, hour_cos]]
        
        # Chuẩn hóa và dự đoán
        features_scaled = self.scaler_X.transform(features)
        prediction_scaled = self.model.predict(features_scaled)
        
        # Chuyển ngược về đơn vị gốc
        prediction = self.scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
        
        return prediction

if __name__ == "__main__":
    predictor = PowerPredictor()
    power = predictor.predict(
        temperature=28,
        production_rate=55,
        active_machines=45,
        is_weekend=0,
        is_daytime=1,
        hour=10
    )
    print(f"Dự đoán điện năng: {{power:.1f}} kW")
"""

# Ghi file inference_demo.py
with open('inference_demo.py', 'w', encoding='utf-8') as f:
    f.write(inference_code)

print("Đã tạo file inference_demo.py để demo triển khai")

print("\n" + "="*60)
print("KẾT THÚC BÀI THỰC HÀNH")
print("="*60)
print("\nTóm tắt những gì đã làm:")
print("1. ✅ Tạo dữ liệu mô phỏng từ nhà máy")
print("2. ✅ Phân tích và trực quan hóa dữ liệu")
print("3. ✅ Tiền xử lý và chuẩn hóa dữ liệu")
print("4. ✅ Huấn luyện 4 mô hình khác nhau")
print("5. ✅ So sánh và chọn mô hình tốt nhất")
print("6. ✅ Dự đoán cho dữ liệu mới")
print("7. ✅ Lưu mô hình và tạo inference code")