
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
    print(f"Dự đoán điện năng: {power:.1f} kW")
