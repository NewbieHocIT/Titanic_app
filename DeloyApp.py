import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from joblib import load, dump
import mlflow
import mlflow.sklearn
import os
from mlflow.models.signature import infer_signature
import numpy as np


# Đọc dữ liệu đã xử lý
X_train = pd.read_csv("processed_data/X_train.csv")
X_valid = pd.read_csv("processed_data/X_valid.csv")
X_test = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_valid = pd.read_csv("processed_data/y_valid.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

# Cấu hình MLflow
mlflow.set_experiment("RandomForest_Classification")

# Khởi tạo mô hình Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_results = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')

# ========================== HÀM HỖ TRỢ ==========================

def load_model_from_file(model_path):
    """ Load mô hình từ file, kiểm tra nếu tồn tại. """
    if os.path.exists(model_path):
        return load(model_path)
    else:
        st.error(f"🚨 Không tìm thấy file mô hình: {model_path}")
        return None

def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    """ Tiền xử lý đầu vào cho mô hình. """
    sex = 0 if sex == "Male" else 1
    embarked = {"S": 0, "C": 1, "Q": 2}.get(embarked, -1)
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]], dtype=np.float64)
    return input_data

def get_mlflow_runs():
    """ Lấy danh sách các lần chạy từ MLflow """
    try:
        runs = mlflow.search_runs()
        return runs
    except Exception as e:
        st.error(f"🚨 Lỗi khi lấy dữ liệu từ MLflow: {e}")
        return None

def main():
    st.set_page_config(layout="wide")  
    st.title("📊 Ứng dụng Phân Tích Dữ Liệu & Dự Đoán")

    tab1, tab2, tab3 = st.tabs(["📂 Xử lý dữ liệu", "📈 Huấn luyện mô hình", "🤖 Dự đoán"])

    # ========================== TAB 1: XỬ LÝ DỮ LIỆU ==========================
    with tab1:
        st.header("🔄 Quá trình Xử lý Dữ liệu")

        # Tải dữ liệu Titanic từ URL
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)

        # **1️⃣ Hiển thị Dữ liệu Gốc**
        st.subheader("📋 Dữ liệu gốc")
        st.write(df.head())

        # **2️⃣ Thông tin về dữ liệu**
        st.subheader("📊 Thông tin dữ liệu")
        st.write(f"🔹 **Số dòng và cột**: {df.shape}")
        st.write(f"🔹 **Các cột**: {df.columns}")
        st.write(f"🔹 **Dữ liệu thiếu** (NaN):")
        missing_values = df.isna().sum()
        st.write(missing_values[missing_values > 0])

        # **3️⃣ Hiển thị từng bước xử lý dữ liệu**
        st.subheader("🛠️ Các bước xử lý dữ liệu")

        with st.expander("📌 Bước 1: Loại bỏ cột không cần thiết"):
            drop_cols = ['Name', 'Ticket', 'Cabin']  # Các cột không cần thiết
            df = df.drop(columns=drop_cols)
            st.write(f"🔹 Cột đã loại bỏ: {drop_cols}")
            st.write(df.head())

        with st.expander("📌 Bước 2: Xử lý dữ liệu thiếu"):
            # Hiển thị số lượng dữ liệu thiếu trước khi xử lý
            st.write(f"🔹 Dữ liệu thiếu trước khi xử lý: {df.isna().sum()}")
            
            # Cụ thể xử lý các cột có dữ liệu thiếu
            st.write("🔹 **Xử lý cột 'Age'**: Điền giá trị thiếu bằng giá trị trung bình (median).")
            df['Age'] = df['Age'].fillna(df['Age'].median())
            
            st.write("🔹 **Xử lý cột 'Embarked'**: Điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất (mode).")
            df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

            # Hiển thị số lượng dữ liệu thiếu sau khi xử lý
            st.write(f"🔹 Dữ liệu thiếu sau khi xử lý: {df.isna().sum()}")
            st.write(df.head())

        with st.expander("📌 Bước 3: Chuyển đổi One-Hot Encoding và Mã hóa"):
            # Mã hóa "Sex" theo cách mong muốn
            df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

            # Mã hóa "Embarked" theo cách mong muốn
            df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
            
            # Log các tham số vào MLflow
            mlflow.log_param("encoded_columns", ['Sex', 'Embarked'])

            # Hiển thị kết quả sau khi mã hóa
            st.write(f"🔹 Sau khi mã hóa: ")
            st.write(df.head())


        with st.expander("📌 Bước 4: Chuẩn hóa dữ liệu"):
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            numerical_cols = ['Age', 'Fare']
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            st.write(f"🔹 Dữ liệu sau khi chuẩn hóa: ")
            st.write(df.head())

        # **4️⃣ Hiển thị Dữ liệu Sau Xử Lý**
        st.subheader("✅ Dữ liệu sau xử lý")
        st.write(df.head())

        # **5️⃣ Chia tập dữ liệu**
        from Buoi2_processing import split_data
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df, target_column="Survived")

        # Hiển thị kích thước tập train/val/test
        st.subheader("📊 Kích thước tập dữ liệu sau khi chia")
        st.write(f"🔹 **Train:** {X_train.shape} mẫu")
        st.write(f"🔹 **Validation:** {X_valid.shape} mẫu")
        st.write(f"🔹 **Test:** {X_test.shape} mẫu")

        # Cho phép tải về dữ liệu sau xử lý
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Tải xuống dữ liệu đã xử lý", csv, "processed_data.csv", "text/csv")

    # ========================== TAB 2: HUẤN LUYỆN MÔ HÌNH ==========================
    import yaml
    import os

    # ========================== TAB 2: HUẤN LUYỆN MÔ HÌNH ==========================
    with tab2:

        # Hàm lấy thông tin từ MLflow
        def get_mlflow_runs():
            tracking_uri = "file:./mlruns"
            mlflow.set_tracking_uri(tracking_uri)

            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()

            runs_data = []
            for exp in experiments:
                runs = client.search_runs(exp.experiment_id)
                for run in runs:
                    run_data = run.data.to_dictionary()
                    run_info = {
                        "run_id": run.info.run_id,
                        "experiment_id": run.info.experiment_id,
                        "model_type": run_data["params"].get("model_type", "Unknown"),
                        "params": run_data["params"],
                        "metrics": run_data["metrics"],
                    }
                    runs_data.append(run_info)

            return runs_data if runs_data else None

        st.header("📈 Kết quả Huấn luyện Mô hình")

        # Hiển thị kích thước của tập dữ liệu
        st.subheader("📊 Kích thước tập dữ liệu")
        st.write(f"🔹 **Train:** {X_train.shape[0]} mẫu, {X_train.shape[1]} features")
        st.write(f"🔹 **Validation:** {X_valid.shape[0]} mẫu, {X_valid.shape[1]} features")
        st.write(f"🔹 **Test:** {X_test.shape[0]} mẫu, {X_test.shape[1]} features")

        # Chọn mô hình để hiển thị kết quả
        model_choice = st.selectbox("Chọn mô hình để hiển thị kết quả", ["Random Forest", "Linear Regression", "Polynomial Regression"])

        # Lấy thông tin từ MLflow
        runs = get_mlflow_runs()
        if runs is not None:
            filtered_runs = [run for run in runs if run["model_type"] == model_choice]

            if filtered_runs:
                latest_run = filtered_runs[0]  # Lấy kết quả mới nhất

                st.subheader("📌 Parameters")
                params_df = pd.DataFrame(list(latest_run["params"].items()), columns=["Parameter", "Value"])
                st.dataframe(params_df, width=800)

                st.subheader("📊 Metrics")
                metrics_dict = latest_run["metrics"]

                if model_choice == "Random Forest":
                    selected_metrics = [
                        "class_0_f1-score", "class_0_precision", "class_0_recall", "class_0_support",
                        "class_1_f1-score", "class_1_precision", "class_1_recall", "class_1_support",
                        "class_macro avg_f1-score", "class_macro avg_precision", "class_macro avg_recall", "class_macro avg_support",
                        "class_weighted avg_f1-score", "class_weighted avg_precision", "class_weighted avg_recall", "class_weighted avg_support",
                        "fold_1_accuracy", "fold_2_accuracy", "fold_3_accuracy", "fold_4_accuracy", "fold_5_accuracy", "test_accuracy"
                    ]
                elif model_choice == "Linear Regression":
                    selected_metrics = ["fold_1_r2", "fold_2_r2", "fold_3_r2", "fold_4_r2", "fold_5_r2", "test_mse", "test_r2"]
                elif model_choice == "Polynomial Regression":
                    selected_metrics = ["fold_1_r2", "fold_2_r2", "fold_3_r2", "fold_4_r2", "fold_5_r2", "test_mse", "test_r2"]
                else:
                    selected_metrics = []

                metrics_df = pd.DataFrame(
                    [(metric, value) for metric, value in metrics_dict.items() if metric in selected_metrics],
                    columns=["Metric", "Value"]
                )
                st.dataframe(metrics_df, width=1000)

                # ================== Nút xem file meta.yaml ==================
                meta_path = f"mlruns/{latest_run['experiment_id']}/{latest_run['run_id']}/meta.yaml"

                if st.button("📄 Xem Chi tiết của mô hình này"):
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as file:
                            meta_data = yaml.safe_load(file)
                        
                        st.subheader(f"📑 Chi Tiết của  {model_choice}")
                        st.json(meta_data)
                    else:
                        st.error("🚨 Không tìm thấy file `meta.yaml`. Hãy kiểm tra lại!")
            else:
                st.warning(f"Không tìm thấy dữ liệu cho mô hình {model_choice}.")


    # Hiển thị kết quả dự đoán    
    # ========================== TAB 3: DỰ ĐOÁN ==========================
    with tab3:
        st.header("🚢 Titanic Survival Prediction")
        st.subheader("🔢 Nhập Thông Tin Hành Khách")
        col1, col2 = st.columns(2)

        with col1:
            pclass = st.selectbox("🔹 Pclass", [1, 2, 3])
            sex = st.selectbox("🔹 Sex", ["Male", "Female"])
            age = st.number_input("🔹 Age", min_value=0, max_value=100, value=30)
            embarked = st.selectbox("🔹 Embarked", ["S", "C", "Q"])

        with col2:
            sibsp = st.number_input("🔹 SibSp", min_value=0, max_value=10, value=0)
            parch = st.number_input("🔹 Parch", min_value=0, max_value=10, value=0)
            fare = st.number_input("🔹 Fare", min_value=0.0, max_value=500.0, value=32.0)

        # Tiền xử lý đầu vào
        input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)

        # Load mô hình
        model_paths = {
            "Random Forest": "models/random_forest.pkl",
            "Multiple Regression": "models/linear_regression.pkl",
            "Polynomial Regression": "models/polynomial_regression.pkl"
        }

        models = {
            "Random Forest": load_model_from_file(model_paths["Random Forest"]),
            "Multiple Regression": load_model_from_file(model_paths["Multiple Regression"]),
        }

        # Load riêng Polynomial Regression
        poly_model, poly = load_model_from_file(model_paths["Polynomial Regression"])  # Tách riêng model & PolynomialFeatures
        models["Polynomial Regression"] = poly_model  # Lưu model vào dictionary
        input_data_poly = poly.transform(input_data)  # Chuyển đổi đầu vào

        # Nút dự đoán
        if st.button("🚀 Predict Survival"):
            results = {}
            for model_name, model in models.items():
                if model_name == "Polynomial Regression":
                    pred = model.predict(input_data_poly)[0]  # Dự đoán với input đã biến đổi
                else:
                    pred = model.predict(input_data)[0]
                results[model_name] = "🟢 Survived" if pred == 1 else "🔴 Did Not Survive"

            # Hiển thị kết quả từ cả 3 mô hình
            st.subheader("📊 Kết quả Dự đoán")
            for model_name, result in results.items():
                st.write(f"🔹 **{model_name}**: {result}")


if __name__ == "__main__":
    main()
