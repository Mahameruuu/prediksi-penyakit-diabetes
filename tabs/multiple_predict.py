import streamlit as st
import pandas as pd
import numpy as np

def preprocess_data(df, scaler):
    df['smoking_history'] = df['smoking_history'].replace('No Info', np.NaN)
    mode_value = df['smoking_history'].mode()[0]
    df['smoking_history'].fillna(mode_value, inplace=True)
    df = df[df['gender'].isin(['Female', 'Male'])]
    smoking_history_mapping = {'never': 0, 'current': 1, 'former': 2, 'ever': 3, 'not current': 4}
    gender_mapping = {'Female': 0, 'Male': 1}
    df['smoking_history'] = df['smoking_history'].map(smoking_history_mapping)
    df['gender'] = df['gender'].map(gender_mapping)
    columns_to_normalize = ['age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender']
    df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
    return df

def multiple_predict(decision_tree_model, scaler):
    st.write("Ini adalah halaman Multiple Predict")
    uploaded_file = st.file_uploader("Unggah file CSV untuk prediksi", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            original_length = len(data)
            preprocessed_data = preprocess_data(data, scaler)
            if len(preprocessed_data) != original_length:
                st.warning(f"Beberapa baris dihapus selama preprocessing. Baris awal: {original_length}, Baris setelah preprocessing: {len(preprocessed_data)}")
            if 'diabetes' in preprocessed_data.columns:
                preprocessed_data = preprocessed_data.drop('diabetes', axis=1)
            predictions = decision_tree_model.predict(preprocessed_data)
            data = data.iloc[:len(preprocessed_data)]
            data['Prediksi Diabetes'] = predictions
            data['Prediksi Diabetes'] = data['Prediksi Diabetes'].map({0: 'Tidak Diabetes', 1: 'Diabetes'})
            st.write("Hasil Prediksi:")
            st.dataframe(data)
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(label="Unduh Hasil Prediksi sebagai CSV", data=csv, file_name='prediksi_diabetes.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
