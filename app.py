
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")

loaded_pipeline = load_model()

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the service.")

st.divider()

st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    col3, col4 = st.columns(2)

    with col3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        monthly = st.number_input(
            "Monthly Charges ($)", min_value=0.0,
            max_value=200.0, value=70.0
        )

    with col4:
        total = st.number_input(
            "Total Charges ($)", min_value=0.0,
            value=float(monthly * tenure)
        )

    submitted = st.form_submit_button(
        "Predict Churn", type="primary"
    )

if submitted:
    service_values = [
        phone, multiple, security, backup,
        protection, support, tv, movies
    ]

    number_of_services = sum(
        value == "Yes" for value in service_values
    )

    if tenure <= 12:
        tenure_group = "0-1 Year"
    elif tenure <= 24:
        tenure_group = "1-2 Years"
    elif tenure <= 48:
        tenure_group = "2-4 Years"
    else:
        tenure_group = "4-6 Years"

    average_monthly_cost = total / tenure if tenure > 0 else monthly

    new_customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "NumberOfServices": number_of_services,
        "AverageMonthlyCost": average_monthly_cost,
        "TenureGroup": tenure_group
    }])

    try:
        probability = loaded_pipeline.predict_proba(
            new_customer
        )[0][1]

        prediction = int(probability >= 0.58)

        st.divider()
        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{probability:.1%}")

        if prediction == 1:
            st.error("⚠️ Likely to Churn")
            st.write(
                "The model predicts this customer is at risk of leaving."
            )
        else:
            st.success("✅ Likely to Stay")
            st.write(
                "The model predicts this customer is less likely to leave."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the saved pipeline was trained with the "
            "same feature columns and category values."
        )
