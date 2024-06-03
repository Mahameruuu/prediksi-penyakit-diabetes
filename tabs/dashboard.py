import streamlit as st
import numpy as np
import pandas as pd
import os

def calculate_bmi(height_cm, weight_kg):
    if height_cm == 0:
        return 0
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return bmi

def save_prediction(data):
    if not os.path.exists('predictions.csv'):
        data.to_csv('predictions.csv', index=False)
    else:
        data.to_csv('predictions.csv', mode='a', header=False, index=False)

def dashboard(decision_tree_model, scaler):
    pass_validation = True
    errorMsg = []

    col1, col2, col3 = st.columns(3)

    with col1:
        height = st.number_input('Tinggi Badan (cm)', min_value=15.0, max_value=300.0, step=0.1)
        gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        blood_glucose = st.number_input('Tingkat Glukosa Darah', min_value=80.0, max_value=300.0)

    with col2:
        weight = st.number_input('Berat Badan (kg)', min_value=3.0, max_value=200.0, step=0.1)
        had_hypertension = st.selectbox('Riwayat Hipertensi', ['Ya', 'Tidak'])
        hemoglobin = st.number_input('Hemoglobin (HbA1c)', min_value=3.0, max_value=10.0)

    with col3:
        had_heart_disease = st.selectbox('Riwayat Penyakit Jantung', ['Ya', 'Tidak'])
        smoking_history = st.selectbox('Riwayat Merokok', ['Tidak Pernah', 'Bekas Perokok', 'Perokok Aktif'])
        age = st.number_input('Usia', min_value=0.0, max_value=170.0)

    # Checking for minimum value inputs
    if height < 15.0:
        pass_validation = False
        errorMsg.append("Tinggi badan berada pada nilai minimum.")
    if weight < 3.0:
        pass_validation = False
        errorMsg.append("Berat badan berada pada nilai minimum.")
    if blood_glucose < 80.0:
        pass_validation = False
        errorMsg.append("Tingkat glukosa darah berada pada nilai minimum.")
    if hemoglobin < 3.0:
        pass_validation = False
        errorMsg.append("Hemoglobin berada pada nilai minimum.")
    if age < 0.0:
        pass_validation = False
        errorMsg.append("Usia berada pada nilai minimum.")

    # Checking for maximum value inputs
    if height > 300.0:
        pass_validation = False
        errorMsg.append("Tinggi badan melebihi nilai maksimum yang diizinkan.")
    if weight > 200.0:
        pass_validation = False
        errorMsg.append("Berat badan melebihi nilai maksimum yang diizinkan.")
    if blood_glucose > 300.0:
        pass_validation = False
        errorMsg.append("Tingkat glukosa darah melebihi nilai maksimum yang diizinkan.")
    if hemoglobin > 10.0:
        pass_validation = False
        errorMsg.append("Hemoglobin melebihi nilai maksimum yang diizinkan.")
    if age > 170.0:
        pass_validation = False
        errorMsg.append("Usia melebihi nilai maksimum yang diizinkan.")

    bmi = calculate_bmi(height, weight)
    if bmi == 0:
        pass_validation = False
        st.warning("Tinggi badan tidak boleh nol.")
    else:
        st.write(f'BMI yang Dihitung: {bmi:.2f}')

    gender = 1 if gender == 'Laki-laki' else 0
    had_hypertension = 1 if had_hypertension == 'Ya' else 0
    had_heart_disease = 1 if had_heart_disease == 'Ya' else 0
    smoking_history_map = {'Tidak Pernah': 0, 'Bekas Perokok': 2, 'Perokok Aktif': 1}
    smoking_history = smoking_history_map[smoking_history]

    input_features = np.array([[age, had_hypertension, had_heart_disease, smoking_history, bmi, hemoglobin, blood_glucose, gender]])

    if pass_validation:
        if st.button('Prediksi'):
            try:
                scaled_features = scaler.transform(input_features)
                prediction = decision_tree_model.predict(scaled_features)
                result = 'Diabetes' if prediction[0] == 1 else 'Tidak Diabetes'
                st.write(f'Prediksi: {result}')
                
                # Save prediction to history
                prediction_data = pd.DataFrame([[age, had_hypertension, had_heart_disease, smoking_history, bmi, hemoglobin, blood_glucose, gender, result]], 
                                               columns=['age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender', 'result'])
                save_prediction(prediction_data)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        for msg in errorMsg:
            st.error(msg)
