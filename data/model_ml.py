import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv('/content/gdrive/MyDrive/Project_ML/customer_churn.csv')
data.sample(5)

data.drop('customerID', axis='columns', inplace=True)

pd.to_numeric(data.TotalCharges, errors='coerce').isnull()

data[pd.to_numeric(data.TotalCharges, errors='coerce').isnull()]

df1 = data[data.TotalCharges!=' ']
df1.shape

df1 = df1[df1['gender'].isin(['Female', 'Male'])].copy()
df1['gender'] = df1['gender'].replace({'Female': 1, 'Male': 0}).astype('int64')
print(df1['gender'].dtype)

df1.loc[:, 'TotalCharges'] = pd.to_numeric(df1['TotalCharges'])

def print_unique_col_values(data):
  for column in data:
    if data[column].dtypes=='object':
      print(f'{column}: {data[column].unique()}')

print_unique_col_values(df1)

df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)

print_unique_col_values(df1)

yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in yes_no_columns:
  df1[col].replace({'Yes': 1, 'No': 0}, inplace=True)

for col in df1:
  print(f'{col}: {df1[col].unique()}')

df2 = pd.get_dummies(data=df1, columns=['InternetService', 'Contract', 'PaymentMethod'])
df2 = df2.replace({True: 1, False: 0})
df2.columns

cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

for col in df2:
  print(f'{col}: {df2[col].unique()}')

X = df2.drop('Churn', axis='columns')
y = testLabels = df2.Churn.astype(np.float32)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

input_data = pd.DataFrame({
    'gender': [1],  # Giả sử khách hàng là nữ (Female), có thể thay đổi thành 0 nếu là nam (Male)
    'SeniorCitizen': [0],  # Không phải là người cao tuổi
    'Partner': [1],  # Có đối tác
    'Dependents': [0],  # Không có người phụ thuộc
    'tenure': [0],  # Giả sử thời gian sử dụng dịch vụ là 0.5
    'PhoneService': [0],  # Có dịch vụ điện thoại
    'MultipleLines': [0],  # Không sử dụng dịch vụ nhiều dòng
    'OnlineSecurity': [0],  # Không sử dụng bảo mật trực tuyến
    'OnlineBackup': [1],  # Có sao lưu trực tuyến
    'DeviceProtection': [0],  # Không có bảo vệ thiết bị
    'TechSupport': [0],  # Không sử dụng hỗ trợ kỹ thuật
    'StreamingTV': [0],  # Có dịch vụ xem TV trực tuyến
    'StreamingMovies': [0],  # Có dịch vụ xem phim trực tuyến
    'PaperlessBilling': [1],  # Sử dụng hóa đơn không giấy tờ
    'MonthlyCharges': [0.115423],  # Giả sử hóa đơn hàng tháng là 0.5
    'TotalCharges': [0.0012751],  # Giả sử tổng chi phí là 0.3
    'InternetService_DSL': [1],  # Sử dụng dịch vụ DSL
    'InternetService_Fiber optic': [0],  # Không sử dụng dịch vụ cáp quang
    'InternetService_No': [0],  # Không sử dụng dịch vụ Internet
    'Contract_Month-to-month': [1],  # Hợp đồng hàng tháng
    'Contract_One year': [0],  # Không có hợp đồng một năm
    'Contract_Two year': [0],  # Không có hợp đồng hai năm
    'PaymentMethod_Bank transfer (automatic)': [0],  # Không sử dụng chuyển khoản ngân hàng tự động
    'PaymentMethod_Credit card (automatic)': [0],  # Sử dụng thẻ tín dụng tự động
    'PaymentMethod_Electronic check': [1],  # Không sử dụng thanh toán điện tử
    'PaymentMethod_Mailed check': [0]  # Không sử dụng thanh toán bằng th
})

def log_reg(X_train, y_train, X_test, y_test, input_data):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy", acc, "\n")

    y_pred = model.predict(input_data)
    print(y_pred)
    if y_pred == 1:
        return "Khách hàng rời đi"
    else:
        return "Khách hàng không rời đi"

log_reg(X_train, y_train, X_test, y_test, input_data)

# def log_reg1(X_train, y_train, X_test, y_test, input_data):
#     model = LogisticRegression()
#     model.fit(X_train, y_train)
#     acc = model.score(X_test, y_test)
#     print("Accuracy", acc, "\n")

#     predicted_result = model.predict(input_data)
#     # print("Predicted result:", predicted_result)

#     # In kết quả dự đoán
#     if predicted_result == 0:
#         print("Khách hàng không rời đi")
#     else:
#         print("Khách hàng rời đi")
#     print(predicted_result)
# log_reg1(X_train, y_train, X_test, y_test, input_data)

# model = LogisticRegression()
# def predict_churn(model, input_data):
#     model.fit(X_train, y_train)
#     input_df = pd.DataFrame(input_data)

#     input_features = input_df

#     # Dự đoán kết quả churn
#     churn_prediction = model.predict(input_features)
#     print(churn_prediction)
#     # Trả về kết quả dự đoán
#     if churn_prediction == 0:
#         return "Khách hàng không rời đi"
#     else:
#         return "Khách hàng rời đi"


# # Sử dụng hàm predict_churn để dự đoán trên dữ liệu mới
# predict_churn(model, input_data)