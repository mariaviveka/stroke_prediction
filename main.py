# main.py
import streamlit as st
import pandas as pd
import joblib
import traceback

# Import custom encoder so joblib can unpickle the pipeline
try:
    from encoder import MultiLabelEncoder
except Exception:
    pass

MODEL_PATH = "Model/stroke_model.joblib"  # adjust path if needed

st.set_page_config(page_title="Stroke Risk Prediction", layout="centered")
st.title("🧠 Stroke Risk Prediction")

# Load model
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        raise

model = load_model(MODEL_PATH)

# ---------------- User Inputs (limited) ---------------- #
st.subheader("Patient Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=55, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])

with col2:
    avg_glucose = st.number_input("Avg. Glucose Level (mg/dL)", min_value=40.0, max_value=400.0, value=110.0, step=0.1)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=26.5, step=0.1)
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col3:
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=126, step=1)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80, step=1)
    smoking_status = st.selectbox("Smoking Status", ["Formerly smoked", "Never smoked", "Smokes", "Unknown"])

family_history = st.selectbox("Family History of Stroke", ["No", "Yes"])

# Symptoms multiselect
symptom_options = [
    "Blurred vision", "Confusion", "Difficulty speaking", "Dizziness",
    "Headache", "Loss of balance", "Numbness", "Seizures",
    "Severe fatigue", "Weakness"
]
selected_symptoms = st.multiselect("Symptoms (select all that apply)", symptom_options)

# Stress Levels numeric with tooltip
stress_levels = st.slider(
    "Stress Level (1-10)",
    min_value=1, max_value=10, value=5,
    help="Stress Levels were numeric in training (1–10)."
)

# ---------------- Hidden placeholders (auto-filled) ---------------- #
hidden_defaults = {
    "Stroke History": 0,
    "HDL": 50,
    "LDL": 120,
    "Marital Status": "Single",
    "Work Type": "Private",
    "Residence Type": "Urban",
    "Alcohol Intake": "Never",
    "Physical Activity": "Moderate",
    "Dietary Habits": "Balanced"
}

# ---------------- Build Input Data ---------------- #
symptom_list_for_model = [s.strip().lower() for s in selected_symptoms]

input_dict = {
    # User-facing inputs
    "Age": int(age),
    "Gender": gender,
    "Hypertension": 1 if hypertension == "Yes" else 0,
    "Heart Disease": 1 if heart_disease == "Yes" else 0,
    "Average Glucose Level": float(avg_glucose),
    "Body Mass Index (BMI)": float(bmi),
    "Systolic_BP": int(systolic_bp),
    "Diastolic_BP": int(diastolic_bp),
    "Smoking Status": smoking_status,
    "Family History of Stroke": family_history,
    "symptom_list": symptom_list_for_model,
    "Stress Levels": int(stress_levels),
    # Hidden defaults
    **hidden_defaults
}

input_df = pd.DataFrame([input_dict])

st.subheader("Input Preview")
st.write(input_df)

# ---------------- Prediction ---------------- #
numeric_cols = ["Age", "Hypertension", "Heart Disease", "Average Glucose Level",
                "Body Mass Index (BMI)", "Systolic_BP", "Diastolic_BP",
                "Stroke History", "HDL", "LDL", "Stress Levels"]

for c in numeric_cols:
    input_df[c] = pd.to_numeric(input_df[c], errors="coerce")

nan_cols = input_df[numeric_cols].columns[input_df[numeric_cols].isna().any()].tolist()
if nan_cols:
    st.error(f"Numeric conversion failed for columns: {nan_cols}. Please check inputs.")
else:
    if st.button("Predict Stroke Risk"):
        try:
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(input_df)[0][1])
            else:
                pred = model.predict(input_df)[0]
                prob = float(pred)

            pred_label = 1 if prob >= 0.5 else 0

            if pred_label == 1:
                st.error(f"⚠️ High Risk of Stroke (probability = {prob:.2f})")
            else:
                st.success(f"✅ Low Risk of Stroke (probability = {prob:.2f})")

        except Exception:
            st.error("Error during prediction — see details below.")
            st.text(traceback.format_exc())











