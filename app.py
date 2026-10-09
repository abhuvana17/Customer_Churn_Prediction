import streamlit as st          # Streamlit: builds the website
import pandas as pd             # to build a one-row table for the model
import joblib                   # to load the saved model

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
# Sets the browser tab title and icon. Must be the first Streamlit command.

@st.cache_resource              # decorator: load the model only once, then reuse it (faster)
def load_model():               # a function that loads the model
    return joblib.load("model/churn_model.joblib")   # read the saved pipeline from disk

model = load_model()            # call the function; model is now ready

st.title("📉 Customer Churn Predictor")   # big heading on the page
st.write("Enter customer details to see how likely they are to leave.")  # normal text

# ---------- INPUT WIDGETS ----------
# Each widget returns the value the user selected.
tenure = st.slider("Tenure (months)", 0, 72, 12)
# slider(label, min, max, default)

monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=70.0)
total = st.number_input("Total charges", min_value=0.0, max_value=10000.0, value=1000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
# selectbox(label, list of choices): a dropdown menu

internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"])
tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])

# ---------- PREDICTION ----------
if st.button("Predict"):                 # runs the code below only when the button is clicked
    row = pd.DataFrame([{                # build a table with ONE row from the user's inputs
        "tenure": tenure,                # column names MUST match training exactly
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "TechSupport": tech,
    }])

    prob = model.predict_proba(row)[0][1]
    # predict_proba gives [[P(stay), P(churn)]]; [0] = first row, [1] = churn probability

    st.metric("Churn probability", f"{prob:.1%}")   # show it as a percentage, e.g. 63.2%

    if prob >= 0.5:                      # decision threshold
        st.error("⚠️ High risk: this customer is likely to churn.")
    else:
        st.success("✅ Low risk: this customer is likely to stay.")