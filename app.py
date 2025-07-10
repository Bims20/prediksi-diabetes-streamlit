import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('model_xgboost.pkl')
scaler = joblib.load('scaler.pkl')

# Judul Aplikasi
st.title("Prediksi Diabetes Pasien Wanita")
st.write("Masukkan data pasien untuk memprediksi kemungkinan diabetes.")

st.info("Jika Anda tidak mengetahui nilai pasti, gunakan nilai default yang disediakan. Hasil lebih akurat bila menggunakan data asli")

# ===== Input User =====
pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=0)
st.caption("Rentang normal: 0 – 17 | Default: 0")

glucose = st.number_input("Kadar Glukosa (mg/dL)", min_value=0, max_value=250, value=100)
st.caption("Rentang normal: 70 – 140 | Default: 100")

blood_pressure = st.number_input("Tekanan Darah Diastolik (mm Hg)", min_value=0, max_value=140, value=80)
st.caption("Rentang normal: 60 – 100 | Default: 80")

skin_thickness = st.number_input("Ketebalan Lipatan Kulit (mm)", min_value=0, max_value=100, value=20)
st.caption("Rentang normal: 10 – 60 | Default: 20")

insulin = st.number_input("Kadar Insulin (μU/mL)", min_value=0, max_value=900, value=80)
st.caption("Rentang normal: 15 – 276 | Default: 80")

bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=0.0, max_value=60.0, value=25.0)
st.caption("Rentang normal: 18.5 – 24.9 | Default: 25.0")

# DPF dropdown
dpf_option = st.selectbox(
    "Riwayat Diabetes dalam Keluarga",
    [
        "Tidak ada riwayat diabetes",
        "Satu anggota keluarga",
        "Lebih dari satu keluarga",
        "3+ anggota keluarga memiliki diabetes"
    ]
)
st.caption("pilih salah satu")

# Mapping DPF
dpf_mapping = {
    "Tidak ada riwayat diabetes": 0.2,
    "Satu anggota keluarga": 0.5,
    "Lebih dari satu keluarga": 0.8,
    "3+ anggota keluarga memiliki diabetes": 1.5
}
dpf = dpf_mapping[dpf_option]

age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=30)
st.caption("Rentang umum: 21 – 80 | Default: 30")

# ===== Tombol Prediksi =====
if st.button("Prediksi"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Hasil: Pasien berisiko Diabetes 😟")
    else:
        st.success("Hasil: Pasien tidak berisiko Diabetes 😌")

# ===== Penjelasan Lengkap Kolom =====
st.markdown("### ℹ️ Penjelasan Setiap Kolom Input")
st.markdown("""
- **Pregnancies**: Jumlah kehamilan yang pernah dialami pasien wanita.
- **Glucose**: Kadar glukosa darah setelah 2 jam minum larutan glukosa. Diukur dengan tes darah (OGTT) atau alat glukometer.
- **Blood Pressure**: Tekanan darah diastolik (angka bawah). Diukur dengan tensimeter.
- **Skin Thickness**: Ketebalan lipatan kulit triceps. Diukur menggunakan skinfold caliper oleh tenaga medis.
- **Insulin**: Kadar insulin darah setelah puasa. Hanya bisa diperoleh melalui tes darah di laboratorium.
- **BMI**: Indeks massa tubuh dihitung dari berat dan tinggi badan. Bisa dihitung mandiri.
- **Diabetes Pedigree Function**: Nilai statistik yang menunjukkan riwayat diabetes dalam keluarga. Diisi melalui pilihan kategori.
- **Age**: Usia pasien saat pemeriksaan. Diisi langsung oleh pengguna.
""")
