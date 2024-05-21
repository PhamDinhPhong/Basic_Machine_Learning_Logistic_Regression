from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')

# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# def train_logistic_regression(X, y, lr=0.01, iterations=1000):
#     m, n = X.shape
#     X = np.hstack((np.ones((m, 1)), X))  # Thêm cột bias (cột các giá trị 1) vào X
#     weights = np.zeros(n + 1)
    
#     for _ in range(iterations):
#         z = np.dot(X, weights)
#         predictions = sigmoid(z)
#         errors = y - predictions
#         gradient = np.dot(X.T, errors)
#         weights += lr * gradient / m
    
#     return weights

# def predict_logistic_regression(weights, X):
#     X = np.hstack((np.ones((X.shape[0], 1)), X))  # Thêm cột bias vào X
#     z = np.dot(X, weights)
#     probabilities = sigmoid(z)
#     return [1 if prob >= 0.5 else 0 for prob in probabilities]

def result(request):
    scaler = MinMaxScaler()
    def process_categorical_feature(value, options):
        encoded_values = [0] * len(options)
        if value in options:
            encoded_values[options.index(value)] = 1
        return encoded_values

    def replace_no_service(data):
        data.replace({'No internet service': 0, 'No phone service': 0}, inplace=True)
        return data

    def replace_yes_no(data):
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = data[column].replace({'Yes': 1, 'No': 0})
        return data

    def replace_gender(data):
        data['gender'] = data['gender'].replace({'Female': 0, 'Male': 1})
        return data

    # def replace_boolean(data):
    #     data = data.replace({True: 1, False: 0})
    #     return data

    dataset = pd.read_csv('./data/customer_churn.csv')
    df1 = dataset.drop('customerID', axis='columns')
    df1['TotalCharges'] = pd.to_numeric(df1['TotalCharges'], errors='coerce')
    df1 = df1.dropna()
    df1 = pd.get_dummies(df1, columns=['InternetService', 'Contract', 'PaymentMethod'])
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])
    df1 = replace_no_service(df1)
    df1 = replace_yes_no(df1)
    df1 = replace_gender(df1)
    # df1 = replace_boolean(df1)
    

    # Train test split
    X = df1.drop("Churn", axis=1)
    Y = df1['Churn']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    
    # acc = model.score(X_test, Y_test)
    

    internet_service_options = ['DSL', 'Fiber optic', 'No']
    contract_options = ['Month-to-month', 'One year', 'Two year']
    payment_method_options = ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']

    
    gender = request.GET.get('gender')
    senior_citizen = int(request.GET.get('senior_citizen'))
    partner = request.GET.get('partner')
    dependents = request.GET.get('dependents')
    phone_service = request.GET.get('phone_service')
    multiple_lines = request.GET.get('multiple_lines')
    online_security = request.GET.get('online_security')
    online_backup = request.GET.get('online_backup')
    device_protection = request.GET.get('device_protection')
    tech_support = request.GET.get('tech_support')
    streaming_tv = request.GET.get('streaming_tv')
    streaming_movies = request.GET.get('streaming_movies')
    paperless_billing = request.GET.get('paperless_billing')
    payment_method_value = request.GET.get('payment_method')
    internet_service_value = request.GET.get('internet_service')
    contract_value = request.GET.get('contract')
    payment_method_value = request.GET.get('payment_method')

    internet_service_encoded = process_categorical_feature(internet_service_value, internet_service_options)
    contract_encoded = process_categorical_feature(contract_value, contract_options)
    payment_method_encoded = process_categorical_feature(payment_method_value, payment_method_options)

    # Chuyển các giá trị số về dạng float
    
    

    # Chuyển đổi giá trị của tenure, monthly_charges, và total_charges
    tenure_scaled = scaler.fit_transform(np.array(request.GET.get('tenure')).reshape(-1, 1))
    monthly_charges_scaled = scaler.fit_transform(np.array(request.GET.get('monthly_charges')).reshape(-1, 1))
    total_charges_scaled = scaler.fit_transform(np.array(request.GET.get('total_charges')).reshape(-1, 1))

    # Lấy giá trị sau khi chuyển đổi
    tenure_scaled_value = tenure_scaled[0][0]
    monthly_charges_scaled_value = monthly_charges_scaled[0][0]
    total_charges_scaled_value = total_charges_scaled[0][0]

    
    # Tạo mảng đầu vào để dự đoán
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure_scaled_value],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'PaperlessBilling': [paperless_billing],
        'MonthlyCharges': [monthly_charges_scaled_value],
        'TotalCharges': [total_charges_scaled_value],
        'InternetService_DSL': internet_service_encoded[0],
        'InternetService_Fiber optic': internet_service_encoded[1],
        'InternetService_No': internet_service_encoded[2],
        'Contract_Month-to-month': contract_encoded[0],
        'Contract_One year': contract_encoded[1],
        'Contract_Two year': contract_encoded[2],
        'PaymentMethod_Bank transfer (automatic)': payment_method_encoded[0],
        'PaymentMethod_Credit card (automatic)': payment_method_encoded[1],
        'PaymentMethod_Electronic check': payment_method_encoded[2],
        'PaymentMethod_Mailed check': payment_method_encoded[3]
    })
    # input_data = np.array([[
    #     gender,
    #     senior_citizen,
    #     partner,
    #     dependents,
    #     tenure_scaled_value,
    #     phone_service,
    #     multiple_lines,
    #     online_security,
    #     online_backup,
    #     device_protection,
    #     tech_support,
    #     streaming_tv,
    #     streaming_movies,
    #     paperless_billing,
    #     monthly_charges_scaled_value,
    #     total_charges_scaled_value,
    #     internet_service_encoded[0],
    #     internet_service_encoded[1],
    #     internet_service_encoded[2],
    #     contract_encoded[0],
    #     contract_encoded[1],
    #     contract_encoded[2],
    #     payment_method_encoded[0],
    #     payment_method_encoded[1],
    #     payment_method_encoded[2],
    #     payment_method_encoded[3]
    # ]])
    
    # weights = train_logistic_regression(X_train, Y_train)
    # input_data_array = np.array(input_data)
    
    # Dự đoán
    # model = LogisticRegression(lr=0.01)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    pred = model.predict(input_data)
    # pred = predict_logistic_regression(weights, input_data)
    
    result1 = "Không rời" if pred == 1 else "Rời"
    
    return render(request, 'predict.html', {"result": result1})
