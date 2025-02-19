import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("C:/TraThanhTri/PYthon/TriTraThanh/MLvsPython/processed_data.csv")

# Split data
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# MLflow Tracking
mlflow.set_experiment("Titanic_RF_Experiment")

# Kiểm tra nếu model đã tồn tại trong MLflow
runs = mlflow.search_runs(order_by=["start_time desc"])
model_uri = None
cv_accuracy = None
test_acc = None

if not runs.empty:
    last_run = runs.iloc[0]
    last_run_id = last_run.run_id
    model_uri = f"runs:/{last_run_id}/random_forest_model"
    # Lấy giá trị metric từ lần chạy trước
    cv_accuracy = last_run.get("metrics.cv_accuracy")
    test_acc = last_run.get("metrics.test_accuracy")

if model_uri is None:  # Chỉ train nếu chưa có model
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Training & Validation
        scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        # Logging to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("cv_accuracy", np.mean(scores))
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(model, "random_forest_model")

        model_uri = f"runs:/{run_id}/random_forest_model"  # Cập nhật model_uri
        cv_accuracy = np.mean(scores)  # Cập nhật giá trị để hiển thị

# Streamlit UI
st.title("Titanic Survival Prediction with MLflow")
model_path = "random_forest_model.pkl"

# Hiển thị kích thước tập dữ liệu
st.write("## Dataset Split Information")
st.write(f"Train set size: {X_train.shape[0]} samples")
st.write(f"Validation set size: {X_valid.shape[0]} samples")
st.write(f"Test set size: {X_test.shape[0]} samples")

st.write("## Model Performance")
st.write(f"Cross-validation Accuracy: {cv_accuracy:.4f}")
st.write(f"Test Accuracy: {test_acc:.4f}")

# User Input for Prediction
st.sidebar.header("Enter Passenger Details")
pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.sidebar.number_input("SibSp", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Parch", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=500.0, value=32.0)

# Convert inputs
sex = 0 if sex == "Male" else 1
input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])

# Make Prediction
if st.sidebar.button("Predict Survival"):
    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
        prediction = loaded_model.predict(input_data)[0]
        result = "Survived" if prediction == 1 else "Did Not Survive"
        st.write(f"### Prediction: {result}")
    except Exception as e:
        st.error(f"Error loading model: {e}")