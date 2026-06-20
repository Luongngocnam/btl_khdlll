import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import re

print("--- Đang xử lý dữ liệu MUA BÁN BẤT ĐỘNG SẢN TOÀN QUỐC ---")

# 1. Đọc dữ liệu từ file MUA BÁN (Hãy đổi tên file vừa tải về thành bds_mua_ban.csv)
try:
    df = pd.read_csv('sale_real_estate.csv')
except FileNotFoundError:
    print("LỖI: Bạn chưa đổi tên file tải về thành 'sale_real_estate.csv' hoặc chưa để vào thư mục D:\\ktmh")
    exit()

# Hàm bóc tách Tỉnh, Quận, Phường
def split_address_deep(address):
    parts = [p.strip() for p in str(address).split(',') if p.strip()]
    province, district, sub_district = "Không Rõ", "Không Rõ", "Không Rõ"
    if len(parts) >= 3:
        province, district, sub_district = parts[-1].title(), parts[-2].title(), parts[-3].title()
    elif len(parts) == 2:
        province, district = parts[-1].title(), parts[-2].title()
    elif len(parts) == 1:
        province = parts[0].title()
    return province, district, sub_district

print("-> Đang phân tách cấu trúc địa lý...")
extracted = df['address'].apply(split_address_deep)
df['province'] = [e[0] for e in extracted]
df['district'] = [e[1] for e in extracted]
df['sub_district'] = [e[2] for e in extracted]

df = df.rename(columns={'bedrooms_num': 'bedroom', 'bathrooms_num': 'toilet'})

# Hàm thông minh biến đổi mọi định dạng chữ (3.5 tỷ, 800 triệu...) về số thực dạng TỶ VNĐ
def clean_price_to_billion(price_str, area):
    price_str = str(price_str).lower().strip()
    if 'thỏa thuận' in price_str or price_str == 'nan':
        return np.nan
    
    # Trích xuất số
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', price_str)
    if not numbers:
        return np.nan
    value = float(numbers[0])
    
    if 'tỷ' in price_str:
        return value
    elif 'triệu/m²' in price_str and pd.notna(area):
        return (value * area) / 1000  # Quy đổi triệu/m2 sang tổng giá tỷ
    elif 'triệu' in price_str:
        return value / 1000           # 500 triệu -> 0.5 tỷ
    
    # Nếu chỉ có số thuần túy, giả định người dùng nhập đơn vị tỷ (hoặc triệu nếu quá nhỏ)
    if value > 1000: 
        return value / 1000000000     # Nếu gõ hẳn số 3.500.000.000
    return value

print("-> Đang quy đổi giá về đơn vị Tỷ VNĐ...")
df['area'] = pd.to_numeric(df['area'].astype(str).str.replace(r'[^\d\.]', '', regex=True), errors='coerce')
df['price'] = df.apply(lambda row: clean_price_to_billion(row['price'], row['area']), axis=1)

required_cols = ['province', 'district', 'sub_district', 'area', 'bedroom', 'toilet', 'price']
df = df[required_cols].dropna()

# Lọc bỏ Outliers quá phi lý (Ví dụ: nhà < 5m2 hoặc giá bán < 50 triệu)
df = df[(df['area'] >= 10) & (df['area'] <= 500)]
df = df[(df['price'] >= 0.05) & (df['price'] <= 50.0)]

# Tạo bản đồ phân cấp địa lý
geo_map = {}
for prov in df['province'].unique():
    geo_map[prov] = {}
    districts_in_prov = df[df['province'] == prov]['district'].unique()
    for dist in districts_in_prov:
        subs_in_dist = df[(df['province'] == prov) & (df['district'] == dist)]['sub_district'].unique()
        geo_map[prov][dist] = sorted(list(subs_in_dist))

# Mã hóa nhãn
encoders = {}
for col in ['province', 'district', 'sub_district']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Huấn luyện mô hình XGBoost
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

with open('real_estate_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'encoders': encoders, 'geo_map': geo_map}, f)

print(f"--> Đã hoàn thành! Độ chính xác mô hình mua bán: {model.score(X_test, y_test)*100:.2f}%")