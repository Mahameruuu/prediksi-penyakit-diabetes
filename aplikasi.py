import streamlit as st
import numpy as np
import joblib

def aplikasi():
    try:
        decision_tree_model = joblib.load('D:/Prediksi Diabetes/models/decision_tree_model.pkl')
        scaler = joblib.load('D:/Prediksi Diabetes/models/scaler.pkl')
    except FileNotFoundError:
        st.error("Model atau scaler tidak ditemukan. Pastikan file 'decision_tree_model.pkl' dan 'scaler.pkl' berada di direktori 'models'.")

    # calculate BMI
    def calculate_bmi(height_cm, weight_kg):
        if height_cm == 0:
            return 0
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return bmi

    st.title('Aplikasi Prediksi Diabetes')

    # Sidebar
    st.sidebar.title('Menu')
    menu = st.sidebar.radio('', ['Dashboard', 'Visualisasi', 'About'])

    if menu == 'Dashboard':
        # Field input
        height = st.number_input('Tinggi Badan (cm)', min_value=0.0, step=0.1)
        weight = st.number_input('Berat Badan (kg)', min_value=0.0, step=0.1)
        gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        had_hypertension = st.selectbox('Riwayat Hipertensi', ['Ya', 'Tidak'])
        blood_glucose = st.number_input('Tingkat Glukosa Darah', min_value=0.0)
        hemoglobin = st.number_input('Hemoglobin (HbA1c)', min_value=0.0)
        had_heart_disease = st.selectbox('Riwayat Penyakit Jantung', ['Ya', 'Tidak'])
        smoking_history = st.selectbox('Riwayat Merokok', ['Tidak Pernah', 'Bekas Perokok', 'Perokok Aktif'])
        age = st.number_input('Usia', min_value=0)

        # Hitung BMI
        bmi = calculate_bmi(height, weight)
        if bmi == 0:
            st.warning("Tinggi badan tidak boleh nol.")
        else:
            st.write(f'BMI yang Dihitung: {bmi:.2f}')

        # Konversi variabel kategorikal menjadi numerik
        gender = 1 if gender == 'Laki-laki' else 0
        had_hypertension = 1 if had_hypertension == 'Ya' else 0
        had_heart_disease = 1 if had_heart_disease == 'Ya' else 0
        smoking_history_map = {'Tidak Pernah': 0, 'Bekas Perokok': 1, 'Perokok Aktif': 2}
        smoking_history = smoking_history_map[smoking_history]

        # Fitur
        input_features = np.array([[age, had_hypertension, had_heart_disease, smoking_history, bmi, hemoglobin, blood_glucose, gender]])

        # Prediksi diabetes
        if st.button('Prediksi'):
            if height == 0:
                st.error("Tinggi badan tidak boleh nol.")
            else:
                try:
                    scaled_features = scaler.transform(input_features)
                    st.write(f'Scaled features: {scaled_features}')
                    prediction = decision_tree_model.predict(scaled_features)
                    result = 'Diabetes' if prediction[0] == 1 else 'Tidak Diabetes'
                    st.write(f'Prediksi: {result}')
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

    elif menu == 'Visualisasi':
        st.write("Ini adalah halaman Visualisasi")

    elif menu == 'About':
        st.write("Ini adalah halaman About")
