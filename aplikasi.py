import streamlit as st
import pandas as pd
import numpy as np
import joblib

def aplikasi():
    try:
        decision_tree_model = joblib.load('models/decision_tree_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        st.error("Model atau scaler tidak ditemukan. Pastikan file 'decision_tree_model.pkl' dan 'scaler.pkl' berada di direktori 'models'.")

    # calculate BMI
    def calculate_bmi(height_cm, weight_kg):
        if height_cm == 0:
            return 0
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return bmi

    def preprocess_data(df):
        # Replace 'No Info' with mode in 'smoking_history'
        df['smoking_history'] = df['smoking_history'].replace('No Info', np.NaN)
        mode_value = df['smoking_history'].mode()[0]
        df['smoking_history'].fillna(mode_value, inplace=True)
        
        # Drop rows with 'gender' as 'Other'
        df = df[df['gender'].isin(['Female', 'Male'])]
        
        # Map values to integers
        smoking_history_mapping = {'never': 0, 'current': 1, 'former': 2, 'ever': 3, 'not current': 4}
        gender_mapping = {'Female': 0, 'Male': 1}
        df['smoking_history'] = df['smoking_history'].map(smoking_history_mapping)
        df['gender'] = df['gender'].map(gender_mapping)
        
        # Normalize specified columns
        columns_to_normalize = ['age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender']
        df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
        
        return df

    st.title('Aplikasi Prediksi Diabetes')

    # Sidebar
    st.sidebar.title('Menu')
    menu = st.sidebar.radio('', ['Dashboard', 'Visualisasi', 'Multiple Predict', 'About'])

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
        smoking_history_map = {'Tidak Pernah': 0, 'Bekas Perokok': 2, 'Perokok Aktif': 1}
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
                    prediction = decision_tree_model.predict(scaled_features)
                    result = 'Diabetes' if prediction[0] == 1 else 'Tidak Diabetes'
                    st.write(f'Prediksi: {result}')
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

    elif menu == 'Visualisasi':
        st.write("Ini adalah halaman Visualisasi")
    
    elif menu == 'Multiple Predict':
        st.write("Ini adalah halaman Multiple Predict")

        # Upload CSV file
        uploaded_file = st.file_uploader("Unggah file CSV untuk prediksi", type=["csv"])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                original_length = len(data)
                
                # Preprocess the data
                preprocessed_data = preprocess_data(data)
                
                # Check if the preprocessing removed any rows
                if len(preprocessed_data) != original_length:
                    st.warning(f"Beberapa baris dihapus selama preprocessing. Baris awal: {original_length}, Baris setelah preprocessing: {len(preprocessed_data)}")
                
                # Drop the target column if it exists
                if 'diabetes' in preprocessed_data.columns:
                    preprocessed_data = preprocessed_data.drop('diabetes', axis=1)
                
                # Make predictions
                predictions = decision_tree_model.predict(preprocessed_data)
                
                # Add predictions to the original data
                data = data.iloc[:len(preprocessed_data)]
                data['Prediksi Diabetes'] = predictions
                data['Prediksi Diabetes'] = data['Prediksi Diabetes'].map({0: 'Tidak Diabetes', 1: 'Diabetes'})
                
                st.write("Hasil Prediksi:")
                st.dataframe(data)
                
                # Provide option to download the predictions
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(label="Unduh Hasil Prediksi sebagai CSV", data=csv, file_name='prediksi_diabetes.csv', mime='text/csv')
            
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

    elif menu == 'About':
        st.write("Ini adalah halaman About")

if __name__ == '__main__':
    aplikasi()