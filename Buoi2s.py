import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 📌 Đọc dữ liệu từ URL
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

# 🔹 Xử lý dữ liệu bị thiếu (NaN)
df['Age'] = df['Age'].fillna(df['Age'].median())
df.dropna(subset=['Embarked'], inplace=True)
df['Cabin'] = df['Cabin'].fillna('Unknown')

# 🔹 Chuyển đổi dữ liệu dạng chuỗi thành số
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 🔹 Loại bỏ các cột không cần thiết
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# 🔹 Xử lý Outliers (ngoại lệ)
Q1, Q3 = df['Fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

# 🔹 Chuẩn hóa dữ liệu
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 🔹 Chia tập dữ liệu
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# 📌 Hiển thị Dashboard trên Streamlit
st.title("🚢 Titanic Data Preprocessing Dashboard")

# 📊 Hiển thị DataFrame sau xử lý
st.subheader("🔹 Dữ liệu sau khi tiền xử lý")
st.dataframe(df.head())

# 📌 Hiển thị thông tin tập dữ liệu
st.subheader("📊 Thông tin kích thước tập dữ liệu")
st.write(f"**🔹 Training size:** {X_train.shape[0]}")
st.write(f"**🔸 Validation size:** {X_val.shape[0]}")
st.write(f"**🔹 Test size:** {X_test.shape[0]}")

# 📊 Vẽ biểu đồ tỷ lệ tập dữ liệu
sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
labels = ["Train", "Validation", "Test"]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=['#3498db', '#f39c12', '#2ecc71'], startangle=90)
ax.set_title("📊 Tỉ lệ tập dữ liệu")
st.pyplot(fig)

# ✅ Kết thúc ứng dụng
st.success("🎉 Dữ liệu đã được xử lý và hiển thị thành công!")
