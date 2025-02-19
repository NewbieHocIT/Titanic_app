import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import mlflow

mlflow.set_experiment("ML DataProcessing Experienced")
mlflow.set_tracking_uri('http://127.0.0.1:5000')
with mlflow.start_run(run_name= 'Data Processing'):

    
    data = pd.read_csv('data.csv')
    #Kich thuoc Data
    mlflow.log_param("Original Shape", data.shape)

    #dem gia tri null trong moi column
    mlflow.log_param("Count null", data.isnull().sum()) 

    #dien gia tri NaN
    data = data.copy()  # Tạo bản sao an toàn
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Cabin'].fillna(data['Cabin'].mode()[0], inplace=True)
    # xoa cac hang NaN con lai
    data.dropna(inplace=True)
    mlflow.log_param("Kich thuoc sau khi dropNa", data.shape)
    
    #xoa trung lap
    data.drop_duplicates(inplace=True)
    mlflow.log_param("Kich thuoc sau khi drop_duplicates", data.shape)

    # encode
    print(data.nunique())
    categorical_col = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
    encoder = LabelEncoder()
    for col in categorical_col:
        if data[col].dtype == 'object':
            data[col] = encoder.fit_transform(data[col])
        else:
            data = pd.get_dummies(data, columns=[col], drop_first=True)


    # Z-score Norm

    # pre-processed data
    mlflow.log_param("Count null after Pre-processing", data.isnull().sum()) 

    processed_file = "processed_data.csv"
    data.to_csv(processed_file, index=False)
    mlflow.log_artifact(processed_file)


    #Train-Test split
    X = data.drop(columns=['Survived'])
    y = data['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42, shuffle=True)

    #Traing 90% - Validation 10%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)

    #log_metric
    mlflow.log_metric("Test Size", 20)
    mlflow.log_metric("Train Size", 70)
    mlflow.log_metric("Validation Size", 10)

    mlflow.end_run()