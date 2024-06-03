import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle

def load_model(model_name):
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    return model

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

def dashboard(model, scaler):
    pass_validation = True
    errorMsg = []

    col1, col2, col3 = st.columns(3)

    with col1:
        height = st.number_input('Tinggi Badan (cm)', min_value=0.0, max_value=300.0, step=1.0)
        gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        blood_glucose = st.number_input('Tingkat Glukosa Darah', min_value=0.0, max_value=300.0, step=1.0)

    with col2:
        weight = st.number_input('Berat Badan (kg)', min_value=0.0, max_value=200.0, step=1.0)
        had_hypertension = st.selectbox('Riwayat Hipertensi', ['Ya', 'Tidak'])
        hemoglobin = st.number_input('Hemoglobin (HbA1c)', min_value=0.0, max_value=10.0, step=0.1)

    with col3:
        had_heart_disease = st.selectbox('Riwayat Penyakit Jantung', ['Ya', 'Tidak'])
        smoking_history = st.selectbox('Riwayat Merokok', ['Tidak Pernah', 'Bekas Perokok', 'Perokok Aktif'])
        age = st.number_input('Usia', min_value=0.0, max_value=170.0, step=1.0)

    gender = 1 if gender == 'Laki-laki' else 0
    had_hypertension = 1 if had_hypertension == 'Ya' else 0
    had_heart_disease = 1 if had_heart_disease == 'Ya' else 0
    smoking_history_map = {'Tidak Pernah': 0, 'Bekas Perokok': 2, 'Perokok Aktif': 1}
    smoking_history = smoking_history_map[smoking_history]

    if st.button('Prediksi'):
        # Validasi input untuk nilai 0 dan nilai di bawah minimum yang diizinkan
        if height == 0:
            pass_validation = False
            errorMsg.append("Tinggi badan tidak boleh nol.")
        elif height < 15.0:
            pass_validation = False
            errorMsg.append("Tinggi badan berada di bawah nilai minimum yang diizinkan (15.0 cm).")

        if weight == 0:
            pass_validation = False
            errorMsg.append("Berat badan tidak boleh nol.")
        elif weight < 3.0:
            pass_validation = False
            errorMsg.append("Berat badan berada di bawah nilai minimum yang diizinkan (3.0 kg).")

        if blood_glucose == 0:
            pass_validation = False
            errorMsg.append("Tingkat glukosa darah tidak boleh nol.")
        elif blood_glucose < 80.0:
            pass_validation = False
            errorMsg.append("Tingkat glukosa darah berada di bawah nilai minimum yang diizinkan (80.0 mg/dL).")

        if hemoglobin == 0:
            pass_validation = False
            errorMsg.append("Hemoglobin tidak boleh nol.")
        elif hemoglobin < 3.0:
            pass_validation = False
            errorMsg.append("Hemoglobin berada di bawah nilai minimum yang diizinkan (3.0 g/dL).")

        if age == 0:
            pass_validation = False
            errorMsg.append("Usia tidak boleh nol.")
        elif age < 0.0:
            pass_validation = False
            errorMsg.append("Usia tidak boleh kurang dari nol.")

        # Validasi untuk nilai maksimal
        if height > 300.0:
            pass_validation = False
            errorMsg.append("Tinggi badan melebihi nilai maksimum yang diizinkan (300.0 cm).")
        if weight > 200.0:
            pass_validation = False
            errorMsg.append("Berat badan melebihi nilai maksimum yang diizinkan (200.0 kg).")
        if blood_glucose > 300.0:
            pass_validation = False
            errorMsg.append("Tingkat glukosa darah melebihi nilai maksimum yang diizinkan (300.0 mg/dL).")
        if hemoglobin > 10.0:
            pass_validation = False
            errorMsg.append("Hemoglobin melebihi nilai maksimum yang diizinkan (10.0 g/dL).")
        if age > 170.0:
            pass_validation = False
            errorMsg.append("Usia melebihi nilai maksimum yang diizinkan (170.0 tahun).")

        if pass_validation:
            bmi = calculate_bmi(height, weight)
            if bmi == 0:
                st.warning("Tinggi badan tidak boleh nol.")
            else:
                st.write(f'BMI yang Dihitung: {bmi:.2f}')

            input_features = np.array([[age, had_hypertension, had_heart_disease, smoking_history, bmi, hemoglobin, blood_glucose, gender]])

            try:
                scaled_features = scaler.transform(input_features)
                prediction = model.predict(scaled_features)
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

if __name__ == '__main__':
    st.title('Aplikasi Prediksi Diabetes')
    
    # Load the model and scaler once when the app starts
    model = load_model('decision_tree_model.pkl')
    scaler = load_model('scaler.pkl')
    
    dashboard(model, scaler)
