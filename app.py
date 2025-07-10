import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('model_xgboost.pkl')
scaler = joblib.load('scaler.pkl')

# Judul aplikasi
st.title("Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi kemungkinan diabetes.")

# Input user
pregnancies = st.number_input("Jumlah Kehamilan", min_value=0)
glucose = st.number_input("Kadar Glukosa", min_value=0)
blood_pressure = st.number_input("Tekanan Darah Diastolik", min_value=0)
skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0)
insulin = st.number_input("Kadar Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Usia", min_value=0)

# Tombol prediksi
if st.button("Prediksi"):
    data_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data_input)
    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("Pasien berisiko Diabetes 😟")
    else:
        st.success("Pasien tidak berisiko Diabetes 😌")
