import streamlit as st
import pandas as pd
import joblib

model = joblib.load("diabetes_model.pkl")
encoders = joblib.load("all_encoders.pkl")
scaler = joblib.load("sd.pkl")

st.title("DIABETES PREDICTION")
st.write("Diabetes prediction using patient health details.")


def text_select(col):
    """Show text labels from the saved encoder, return the encoded number."""
    label = st.selectbox(col, list(encoders[col].classes_))
    return encoders[col].transform([label])[0]


data = {}
data["age"] = st.number_input("age", 0, 120, 50, step=1)
data["sex"] = text_select("sex")
data["gender"] = text_select("gender")
data["ethnicity"] = text_select("ethnicity")

data["bmi"] = st.number_input("bmi", 10.0, 70.0, 27.88)
data["waist_circumference_cm"] = st.number_input("waist_circumference_cm", 40.0, 200.0, 85.2)
data["waist_to_hip_ratio"] = st.number_input("waist_to_hip_ratio", 0.4, 1.5, 0.84)
data["body_fat_percentage"] = st.number_input("body_fat_percentage", 3.0, 70.0, 27.2)
data["fasting_glucose_mg_dl"] = st.number_input("fasting_glucose_mg_dl", 40.0, 400.0, 105.6)
data["post_meal_glucose_mg_dl"] = st.number_input("post_meal_glucose_mg_dl", 40.0, 500.0, 155.6)
data["two_hour_glucose_mg_dl"] = st.number_input("two_hour_glucose_mg_dl", 40.0, 500.0, 159.1)
data["hba1c_percent"] = st.number_input("hba1c_percent", 3.0, 15.0, 4.75)
data["fasting_insulin_uIU_ml"] = st.number_input("fasting_insulin_uIU_ml", 0.0, 300.0, 14.74)
data["c_peptide_ng_ml"] = st.number_input("c_peptide_ng_ml", 0.0, 20.0, 2.27)
data["insulin_resistance_index"] = st.number_input("insulin_resistance_index", 0.0, 50.0, 3.76)
data["systolic_bp"] = st.number_input("systolic_bp", 60.0, 250.0, 124.0)
data["diastolic_bp"] = st.number_input("diastolic_bp", 40.0, 150.0, 78.0)
data["ldl_mg_dl"] = st.number_input("ldl_mg_dl", 20.0, 300.0, 119.8)
data["hdl_mg_dl"] = st.number_input("hdl_mg_dl", 10.0, 150.0, 50.0)
data["triglycerides_mg_dl"] = st.number_input("triglycerides_mg_dl", 20.0, 1000.0, 162.9)
data["egfr_ml_min_1_73m2"] = st.number_input("egfr_ml_min_1_73m2", 5.0, 200.0, 90.4)
data["urine_albumin_creatinine_ratio"] = st.number_input("urine_albumin_creatinine_ratio", 0.0, 3000.0, 22.5)

data["family_history_diabetes"] = text_select("family_history_diabetes")
data["smoking_status"] = text_select("smoking_status")
data["physical_activity_level"] = text_select("physical_activity_level")

data["diet_quality_score"] = st.number_input("diet_quality_score", 0.0, 100.0, 58.5)
data["sugar_grams_per_day"] = st.number_input("sugar_grams_per_day", 0.0, 500.0, 106.7)

data["prediabetes_history"] = text_select("prediabetes_history")
data["gestational_diabetes_history"] = text_select("gestational_diabetes_history")
data["diabetes_risk_score"] = st.number_input("diabetes_risk_score", 0.0, 100.0, 6.3)

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([data])

        # same column order as training
        if hasattr(scaler, "feature_names_in_"):
            input_df = input_df[list(scaler.feature_names_in_)]

        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            st.error("Predicted: Diabetes")
        else:
            st.success("Predicted: No Diabetes")

    except Exception as e:
        st.error(f"Prediction Error: {e}")