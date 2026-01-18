# import numpy as np 
# import matplotlib.pyplot as plt 

# np.random.seed(42)

# hours = np.arange(0, 100)

# normal_temp = 70 + 5 * np.sin(hours/10) + np.random.normal(0, 2, 100)

# abnormal_indices = [40, 41, 60, 61, 62, 85, 86]

# normal_temp [abnormal_indices] = [85, 90, 88, 92, 89, 95, 96]

# plt.figure(figsize=(12,6))

# plt.plot(hours, normal_temp, 'b-', linewidth = 2 , label = 'nhiet do dong co')

# plt.scatter(abnormal_indices, normal_temp[abnormal_indices], color='red', s = 100, zorder =5, label = 'bathuong')

# threshold = 80

# plt.axhline(y=threshold, color='orange', linestyle='--',
#             label=f"nguong canh bao {threshold} do C")


# plt.fill_between(hours, threshold, 100, alpha = 0.2, color = 'red')

# plt.xlabel('thoi gian(gio)')

# plt.ylabel('nhiet do(C)')

# plt.title('phat hien bat thuong nhiet do dong co')

# plt.legend()

# plt.grid(True)

# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# 1. TẠO DỮ LIỆU GIẢ LẬP (SINH VIÊN NĂM 3)
def generate_student_data(n=200):
    np.random.seed(42)
    
    # Giả lập dữ liệu bình thường
    data = {
        'MSSV': [f'SV{i:03d}' for i in range(1, n + 1)],
        # GPA tích lũy các kỳ trước (thang 4.0)
        'GPA_TichLuy_Cu': np.random.uniform(2.2, 3.8, n),
        # GPA kỳ hiện tại
        'GPA_Ky_Nay': np.random.uniform(2.0, 3.9, n)
    }
    
    df = pd.DataFrame(data)
    
    # Gây nhiễu: Tạo một số trường hợp bất thường thực tế
    # Trường hợp 1: Sa sút nghiêm trọng
    df.loc[10, 'GPA_Ky_Nay'] = 0.8
    df.loc[50, 'GPA_Ky_Nay'] = 1.1
    
    # Trường hợp 2: Tiến bộ vượt bậc
    df.loc[100, 'GPA_TichLuy_Cu'] = 2.1
    df.loc[100, 'GPA_Ky_Nay'] = 3.9
    
    return df

# 2. PHÂN TÍCH VÀ PHÁT HIỆN BẤT THƯỜNG
def analyze_anomalies(df):
    # Lấy các đặc trưng để đưa vào mô hình
    features = ['GPA_TichLuy_Cu', 'GPA_Ky_Nay']
    X = df[features]
    
    # Khởi tạo mô hình Isolation Forest
    # contamination=0.05 nghĩa là dự đoán khoảng 5% dữ liệu là bất thường
    model = IsolationForest(contamination=0.05, random_state=42)
    
    # Dự đoán (1: bình thường, -1: bất thường)
    df['Anomaly_Code'] = model.fit_predict(X)
    
    # Gán nhãn tư vấn dựa trên sự thay đổi điểm số
    def get_advice(row):
        diff = row['GPA_Ky_Nay'] - row['GPA_TichLuy_Cu']
        
        if row['Anomaly_Code'] == -1:
            if diff < -1.0:
                return "Cảnh báo Đỏ: Sa sút nghiêm trọng - Cần tham vấn tâm lý/học tập ngay"
            elif diff > 1.0:
                return "Khen ngợi: Tiến bộ vượt bậc - Cần phỏng vấn chia sẻ kinh nghiệm"
            else:
                return "Theo dõi: Có biến động lạ so với mặt bằng chung"
        return "Ổn định: Tiếp tục duy trì"

    df['Tu_Van'] = df.apply(get_advice, axis=1)
    return df

# 3. TRỰC QUAN HÓA DỮ LIỆU
def visualize_results(df):
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Vẽ các điểm dữ liệu
    scatter = sns.scatterplot(
        data=df,
        x='GPA_TichLuy_Cu',
        y='GPA_Ky_Nay',
        hue='Anomaly_Code',
        palette={1: 'dodgerblue', -1: 'crimson'},
        style='Anomaly_Code',
        markers={1: 'o', -1: 'X'},
        s=100
    )
    
    plt.title('BIỂU ĐỒ PHÁT HIỆN ĐIỂM HỌC TẬP BẤT THƯỜNG\nSINH VIÊN NĂM 3', fontsize=14)
    plt.xlabel('GPA Tích lũy (Năm 1 & 2)', fontsize=12)
    plt.ylabel('GPA Kỳ hiện tại (Kỳ 5)', fontsize=12)
    plt.legend(title='Phân loại', labels=['Bất thường', 'Bình thường'])
    
    # Vẽ đường chéo tham chiếu (y=x)
    plt.plot([1, 4], [1, 4], color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# 4. CHƯƠNG TRÌNH CHÍNH
if __name__ == "__main__":
    # Bước 1: Chuẩn bị dữ liệu
    student_df = generate_student_data()
    
    # Bước 2: Phân tích
    result_df = analyze_anomalies(student_df)
    
    # Bước 3: Xuất kết quả ra màn hình
    print("-" * 30)
    print("DANH SÁCH SINH VIÊN CẦN TƯ VẤN ĐẶC BIỆT")
    print("-" * 30)
    
    anomalies = result_df[result_df['Anomaly_Code'] == -1].sort_values(by='GPA_Ky_Nay')
    print(anomalies[['MSSV', 'GPA_TichLuy_Cu', 'GPA_Ky_Nay', 'Tu_Van']])
    
    # Bước 4: Lưu file CSV (tùy chọn)
    # result_df.to_csv('Bao_Cao_Tu_Van_Hoc_Tap.csv', index=False)
    
    # Bước 5: Xem biểu đồ
    visualize_results(result_df)