import streamlit as st
import pandas as pd
import os

def history():
    st.title("History")
    if os.path.exists('predictions.csv'):
        history_data = pd.read_csv('predictions.csv')
        st.dataframe(history_data)
    else:
        st.write("Belum ada data prediksi yang disimpan.")
