# ==============================>> IMPORTS <<==============================
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================>> TITLE <<==============================
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Telco Customer Churn Analysis & Prediction App")
st.write("This app analyzes customer data and predicts churn using KNN model.")

# ==============================>> LOAD DATA <<==============================
df = pd.read_csv("Telco-Customer-Churn.csv")
df = df.drop("customerID", axis=1)

text_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "Churn"
]

for col in text_cols:
    df[col] = df[col].str.lower().str.strip()

df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df.loc[df["tenure"] == 0, "TotalCharges"] = 0

df["MultipleLines"] = df["MultipleLines"].replace("no phone service", "no")

replace_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

for col in replace_cols:
    df[col] = df[col].replace("no internet service", "no")

# ==============================>> DATASET OVERVIEW <<==============================
st.header("📁 Dataset Overview")

st.write("Shape of Dataset:", df.shape)
st.write("Sample Records:")
st.dataframe(df.head())

# ============================== VISUALIZATIONS ==============================
st.title("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="gender", hue="Churn", ax=ax)
    ax.set_title("Gender vs Churn")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
    ax.set_title("Monthly Charges vs Churn")
    st.pyplot(fig)

col3, col4 = st.columns(2)
with col3:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Contract", ax=ax)
    # plt.xticks(rotation=15)
    ax.set_title("Contract Distribution")
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, ax=ax)
    ax.set_title("Tenure Distribution")
    st.pyplot(fig)

# ==============================>> MODEL TRAINING <<==============================
# =================>> Encoding <<=================
# Label Encoding
le = LabelEncoder()

encode_cols = [
    "gender","Partner","Dependents","PhoneService","MultipleLines",
    "OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies",
    "PaperlessBilling","Churn"
]

for col in encode_cols:
    df[col] = le.fit_transform(df[col])

# One-hot Encoding
ohe = OneHotEncoder(drop=None, sparse_output=False, dtype=int)

ohe_encoded = ohe.fit_transform(df[["InternetService", "Contract", "PaymentMethod"]])

ohe_encoded_df = pd.DataFrame(
    ohe_encoded,
    columns=ohe.get_feature_names_out(["InternetService", "Contract", "PaymentMethod"])
)

df.drop(columns=["InternetService", "Contract", "PaymentMethod"], inplace=True)

df = pd.concat([df, ohe_encoded_df], axis=1)

joblib.dump(ohe, "ohe_encoder.pkl")

# =================>> Training Process <<=================
# Define Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Training
model = KNeighborsClassifier(n_neighbors=39)
model.fit(X_train, y_train)

#===========>> Testing <<===========
st.subheader("📊 Model Performance")

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Show accuracy as metric
st.metric(
    label="Model Accuracy",
    value=f"{acc*100:.2f}%"
)

# detailed report
with st.expander("📋 Show Classification Report"):
    st.dataframe(report_df.style.format("{:.2f}"))


# =================>> USER INPUT <<=================
st.subheader("🧑 Enter Customer Details")

st.header("Customer Info")
gender = st.selectbox("Gender", ["male", "female"])
senior_citizen = st.selectbox("SeniorCitizen", ["yes", "no"])
partner = st.selectbox("Partner", ["yes", "no"])
dependents = st.selectbox("Dependents", ["yes", "no"])
phone = st.selectbox("Phone Service", ["yes", "no"])
multiple_lines = st.selectbox("Multiple Lines", ["yes", "no"])
online_security = st.selectbox("Online Security", ["yes", "no"])
online_backup = st.selectbox("Online Backup", ["yes", "no"])
device_protection = st.selectbox("Device Protection", ["yes", "no"])
tech_support = st.selectbox("Tech Support", ["yes", "no"])
streaming_tv = st.selectbox("Streaming TV", ["yes", "no"])
streaming_movies = st.selectbox("Streaming Movies", ["yes", "no"])
paperless = st.selectbox("Paperless Billing", ["yes", "no"])
contract = st.selectbox("Contract", ["month-to-month", "one year", "two year"])
internet = st.selectbox("Internet Service", ["dsl", "fiber optic", "no"])
payment_method = st.selectbox("PaymentMethod", ["electronic check", "mailed check", "bank transfer (automatic)", "credit card (automatic)"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

if st.button("🔮 Predict Churn"):
    
    other_df = pd.DataFrame({
        "gender":[1 if gender=="male" else 0],
        "SeniorCitizen":[1 if senior_citizen=="yes" else 0],
        "Partner":[1 if partner=="yes" else 0],
        "Dependents":[1 if dependents=="yes" else 0],
        "PhoneService":[1 if phone=="yes" else 0],
        "MultipleLines":[1 if multiple_lines=="yes" else 0],
        "OnlineSecurity":[1 if online_security=="yes" else 0],
        "OnlineBackup":[1 if online_backup=="yes" else 0],
        "DeviceProtection":[1 if device_protection=="yes" else 0],
        "TechSupport":[1 if tech_support=="yes" else 0],
        "StreamingTV":[1 if streaming_tv=="yes" else 0],
        "StreamingMovies":[1 if streaming_movies=="yes" else 0],
        "PaperlessBilling":[1 if paperless=="yes" else 0],
    })
    
    ohe_df = pd.DataFrame({
        "InternetService": [internet],
        "Contract": [contract],
        "PaymentMethod": [payment_method]
    })
    encoder = joblib.load("ohe_encoder.pkl")
    user_encoded = encoder.transform(ohe_df)
    user_encoded_df = pd.DataFrame(
        user_encoded,
        columns=encoder.get_feature_names_out(
            ["InternetService", "Contract", "PaymentMethod"]
        )
    )

    sacling_data = np.array([[tenure, monthly, total]])
    user_data_scaled = scaler.transform(sacling_data)
    scale_df = pd.DataFrame(
        user_data_scaled,
        columns=["tenure", "MonthlyCharges", "TotalCharges"]
    )

    final_df = pd.concat(
        [other_df, user_encoded_df, scale_df], 
        axis=1
    )

    final_df = final_df[X.columns]

    # =====================>> PREDICTION <<=====================
    pred = model.predict(final_df)
    prob = model.predict_proba(final_df)[0][1]

    st.subheader("📌 Prediction Result")
    st.write(f"Churn Probability: **{prob*100:.2f}%**")

    if pred == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")