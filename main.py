import streamlit as st
from tabs import dashboard, visualisasi, multiple_predict, history, about
import joblib

def main():
    try:
        decision_tree_model = joblib.load('models/decision_tree_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        st.error("Model atau scaler tidak ditemukan. Pastikan file 'decision_tree_model.pkl' dan 'scaler.pkl' berada di direktori 'models'.")
        return

    st.title('Aplikasi Prediksi Diabetes')

    st.sidebar.title('Menu')
    menu = st.sidebar.radio('', ['Dashboard', 'Visualisasi', 'Multiple Predict', 'History', 'About'])

    if menu == 'Dashboard':
        dashboard.dashboard(decision_tree_model, scaler)
    elif menu == 'Visualisasi':
        visualisasi.visualisasi()
    elif menu == 'Multiple Predict':
        multiple_predict.multiple_predict(decision_tree_model, scaler)
    elif menu == 'History':
        history.history()
    elif menu == 'About':
        about.about()

if __name__ == '__main__':
    main()