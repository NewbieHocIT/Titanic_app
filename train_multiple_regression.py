import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# Đọc dữ liệu đã xử lý
X_train = pd.read_csv("processed_data/X_train.csv")
X_valid = pd.read_csv("processed_data/X_valid.csv")
X_test = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_valid = pd.read_csv("processed_data/y_valid.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

# Gộp tập train và valid để huấn luyện
X_train_full = pd.concat([X_train, X_valid])
y_train_full = pd.concat([y_train, y_valid])

# Cấu hình MLflow
mlflow.set_experiment("Multiple_vs_Polynomial_Regression")

# ✅ Tạo thư mục models để lưu mô hình
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

### ================== Linear Regression ==================
with mlflow.start_run() as run:
    print("\nTraining Linear Regression...")

    # Khởi tạo mô hình
    linear_regression_model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross Validation
    cross_val_results = cross_val_score(linear_regression_model, X_train_full, y_train_full, cv=cv, scoring='r2')
    
    # Log các tham số
    mlflow.log_param("model_type", "Linear Regression")
    
    # Log kết quả Cross Validation
    for i, score in enumerate(cross_val_results):
        mlflow.log_metric(f"fold_{i+1}_r2", score)
    
    # Train mô hình
    linear_regression_model.fit(X_train_full, y_train_full)
    
    # Dự đoán trên tập test
    y_pred = linear_regression_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # Log kết quả test
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_r2", test_r2)
    
    # ✅ Lưu mô hình vào thư mục models/
    model_path = os.path.join(model_dir, "linear_regression.pkl")
    joblib.dump(linear_regression_model, model_path)

    # ✅ Log mô hình vào MLflow
    input_example = X_test.iloc[:5]
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(linear_regression_model, "linear_regression_model", signature=signature, input_example=input_example)

    print(f"✅ Linear Regression model saved to {model_path} and logged to MLflow.")

### ================== Polynomial Regression ==================
with mlflow.start_run() as run:
    print("\nTraining Polynomial Regression...")

    # Khởi tạo mô hình
    degree = 2
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_full)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()

    # Cross Validation
    cross_val_results_poly = cross_val_score(poly_model, X_train_poly, y_train_full, cv=cv, scoring='r2')

    # Log các tham số
    mlflow.log_param("model_type", "Polynomial Regression")
    mlflow.log_param("degree", degree)
    
    # Log kết quả Cross Validation
    for i, score in enumerate(cross_val_results_poly):
        mlflow.log_metric(f"fold_{i+1}_r2", score)
    
    # Train mô hình
    poly_model.fit(X_train_poly, y_train_full)
    
    # Dự đoán trên tập test
    y_pred_poly = poly_model.predict(X_test_poly)
    test_mse_poly = mean_squared_error(y_test, y_pred_poly)
    test_r2_poly = r2_score(y_test, y_pred_poly)
    
    # Log kết quả test
    mlflow.log_metric("test_mse", test_mse_poly)
    mlflow.log_metric("test_r2", test_r2_poly)
    
    # ✅ Lưu mô hình vào thư mục models/
    model_path_poly = os.path.join(model_dir, "polynomial_regression.pkl")
    joblib.dump((poly_model, poly), model_path_poly)

    # ✅ Log mô hình vào MLflow
    input_example_poly = poly.transform(X_test.iloc[:5])
    signature_poly = infer_signature(X_test_poly, y_pred_poly)
    mlflow.sklearn.log_model(poly_model, "polynomial_regression_model", signature=signature_poly, input_example=input_example_poly)

    print(f"✅ Polynomial Regression model saved to {model_path_poly} and logged to MLflow.")

print("\n🎯 Training Completed!")
