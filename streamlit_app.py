import json
import os
import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

FEATURE_ORDER = [
    "Month", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
    "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour",
    "Monthly_Balance"
]

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(features: list) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

st.set_page_config(page_title="Credit Score Prediction", layout="centered")
st.title("Customer Credit Score Prediction")
st.write("Isi data nasabah di bawah untuk memprediksi kategori credit score (Poor / Standard / Good).")

with st.form("credit_form"):
    month = st.selectbox("Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        occupation = st.text_input("Occupation", value="Engineer")
        annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0)
        num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, value=3)
        num_credit_card = st.number_input("Num Credit Card", min_value=0, value=4)
        interest_rate = st.number_input("Interest Rate", min_value=0.0, value=12.0)
        num_of_loan = st.number_input("Num of Loan", min_value=0.0, value=2.0)
        delay_from_due_date = st.number_input("Delay from Due Date", min_value=0, value=5)
        num_delayed_payment = st.number_input("Num of Delayed Payment", min_value=0.0, value=3.0)
    with col2:
        changed_credit_limit = st.number_input("Changed Credit Limit", value=5.5)
        num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0.0, value=2.0)
        credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1200.0)
        credit_utilization = st.number_input("Credit Utilization Ratio", min_value=0.0, value=30.5)
        credit_history_age = st.text_input("Credit History Age", value="10 Years and 5 Months")
        payment_min_amount = st.selectbox("Payment of Min Amount", ["Yes", "No"])
        total_emi = st.number_input("Total EMI per month", min_value=0.0, value=150.0)
        amount_invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=300.0)
        payment_behaviour = st.selectbox("Payment Behaviour", [
            "High_spent_Small_value_payments", "Low_spent_Small_value_payments",
            "High_spent_Medium_value_payments", "Low_spent_Medium_value_payments",
            "High_spent_Large_value_payments", "Low_spent_Large_value_payments"
        ])
        monthly_balance = st.number_input("Monthly Balance", value=400.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    features = [
        month, age, occupation, annual_income, monthly_salary,
        num_bank_accounts, num_credit_card, interest_rate, num_of_loan,
        delay_from_due_date, num_delayed_payment, changed_credit_limit,
        num_credit_inquiries, credit_mix, outstanding_debt, credit_utilization,
        credit_history_age, payment_min_amount, total_emi, amount_invested,
        payment_behaviour, monthly_balance
    ]
    try:
        result = invoke_endpoint(features)
        label = result["labels"][0]
        color_map = {"Good": "success", "Standard": "warning", "Poor": "error"}
        getattr(st, color_map[label])(f"Predicted Credit Score: **{label}**")
    except NoCredentialsError:
        st.error("No AWS credentials found.")
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")