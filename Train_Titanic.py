import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from mlflow.models.signature import infer_signature

# 📌 Đọc dữ liệu đã xử lý
X_train = pd.read_csv("processed_data/X_train.csv")
X_valid = pd.read_csv("processed_data/X_valid.csv")
X_test = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_valid = pd.read_csv("processed_data/y_valid.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

# 📌 Cấu hình MLflow
mlflow.set_experiment("RandomForest_Classification")

# 📌 Khởi tạo mô hình Random Forest
n_estimators = 100
random_state = 42
rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

# 📌 Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
cross_val_results = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')

# 📌 Bắt đầu ghi log vào MLflow
with mlflow.start_run():
    print("🔹 Bắt đầu logging vào MLflow...")

    # 🔹 Log tham số mô hình
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # 🔹 Log kết quả Cross Validation
    for i, acc in enumerate(cross_val_results):
        print(f"Fold {i+1}: Accuracy = {acc:.4f}")
        mlflow.log_metric(f"fold_{i+1}_accuracy", acc)

    # 🔹 Log trung bình accuracy từ Cross Validation
    mean_cv_acc = np.mean(cross_val_results)
    mlflow.log_metric("mean_cv_accuracy", mean_cv_acc)

    # 🔹 Huấn luyện mô hình trên tập train + valid
    print("🔹 Đang huấn luyện mô hình...")
    rf_model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
    print("✅ Huấn luyện hoàn tất!")

    # 🔹 Dự đoán trên tập test
    print("🔹 Đang dự đoán trên tập test...")
    y_pred = rf_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    # 🔹 Log kết quả test
    mlflow.log_metric("test_accuracy", test_acc)

    # 🔹 Log classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):  # Chỉ log các giá trị dạng số
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"class_{label}_{metric_name}", value)

    # 🔹 Kiểm tra và tạo thư mục lưu mô hình nếu chưa có
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest.pkl")

    # 🔹 Lưu mô hình
    joblib.dump(rf_model, model_path)

    # 🔹 Tạo input_example và signature để log model đúng cách
    input_example = X_test.iloc[:5]  # Lấy 5 dòng làm ví dụ đầu vào
    signature = infer_signature(X_test, y_pred)  # Tự động tạo signature từ dữ liệu test

    # 🔹 Log model vào MLflow với input_example và signature
    mlflow.sklearn.log_model(
        rf_model,
        "random_forest_model",
        signature=signature,
        input_example=input_example
    )

    print("✅ Mô hình đã được lưu và ghi log vào MLflow thành công!")
