import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Custom CSS for light theme and button color changes
st.markdown("""
    <style>
    body, html, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-color: #F7F9FC;
        color: #333333;
    }

    /* Headers */
    h1, h2, h3 {
        color: #305F72;
    }

    /* Predict button - changed to blue */
    .stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 5px;
        font-weight: 500;
        padding: 0.6rem;
    }

    .stButton > button:hover {
        background-color: #0056b3;
    }

    /* Inputs, selections, and sliders */
    .stSelectbox > div > div, 
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stSlider > div > div {
        background-color: white;
        color: #333;
        border: 1px solid #CCCCCC;
        border-radius: 5px;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #007BFF;
    }

    /* Footer */
    footer {
        text-align: center;
        margin-top: 50px;
        color: #888888;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Load the encoders and scalers
@st.cache_resource
def load_encoders_and_scalers():
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open("scalar.pkl", "rb") as file:
        scalar = pickle.load(file)
    with open("onehot_geo.pkl", "rb") as file:
        label_encoder_geo = pickle.load(file)
    return label_encoder_gender, scalar, label_encoder_geo

label_encoder_gender, scalar, label_encoder_geo = load_encoders_and_scalers()

# Streamlit app layout
st.title("🚀 Customer Churn Prediction")

st.markdown("### Input the customer's details below to predict the churn probability.")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", label_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5)

with col2:
    balance = st.number_input("Balance", min_value=0.0, format="%.2f")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, format="%.2f")
    num_of_products = st.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Predict button
if st.button("Predict Churn Probability"):
    # Preparing input data
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # One-hot encoding geography
    geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(["Geography"]))

    # Combining data frames
    input_data = pd.concat([geo_encoded_df, input_data.reset_index(drop=True)], axis=1)

    # Scaling the data
    input_data_scaled = scalar.transform(input_data)

    # Predict churn probability
    predict = model.predict(input_data_scaled)
    pred_prob = predict[0][0]

    # Display prediction result
    st.subheader("Prediction Result")

    pred_prob_percentage = int(pred_prob * 100)
    st.progress(pred_prob_percentage)

    st.metric(label="Churn Probability (%)", value=f"{pred_prob * 100:.2f}")

    if pred_prob > 0.5:
        st.markdown("**⚠️ High risk of churn detected.**")
    else:
        st.markdown("**✅ Low risk of churn detected.**")

st.markdown("""
    <footer>
        <p>Customer Churn Prediction | Powered by Streamlit</p>
    </footer>
""", unsafe_allow_html=True)
