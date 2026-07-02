import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo y las columnas entrenadas desde la misma carpeta
ruta_base = '/content/drive/MyDrive/proyecto_final_estud-ia/'
modelo = joblib.load(ruta_base + 'modelo_arriendos.pkl')
columnas_modelo = joblib.load(ruta_base + 'columnas_modelo.pkl')

st.title("Predicción de Precios de Arriendo")
st.write("Esta aplicación determina si un inmueble es de **Precio Alto** o **Precio Bajo**.")

# Inputs para el usuario
bedrooms = st.number_input("Habitaciones", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Baños", min_value=1, max_value=5, value=1)
surface = st.number_input("Superficie Total (m2)", min_value=20, max_value=500, value=50)

if st.button("Predecir Precio"):
    # Lógica de predicción
    input_data = pd.DataFrame([[bedrooms, bathrooms, surface]], 
                              columns=['bedrooms', 'bathrooms', 'surface_total'])
    st.success("El modelo está listo para realizar la clasificación.")
