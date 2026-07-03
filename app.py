import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Definimos los nombres de las columnas que tu modelo espera
# (Estos nombres deben coincidir exactamente con los del entrenamiento)
nombres_columnas = ['bedrooms', 'bathrooms', 'surface_total']

# 2. Cargar el modelo
try:
    modelo = joblib.load('modelo_arriendos.pkl')
except FileNotFoundError:
    st.error("No se encontró el archivo 'modelo_arriendos.pkl'.")
    st.stop()

st.title("Predicción de Precios de Arriendo")

# 3. Inputs del usuario
bedrooms = st.number_input("Habitaciones", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Baños", min_value=1, max_value=5, value=1)
surface = st.number_input("Superficie Total (m2)", min_value=20, max_value=500, value=50)

# 4. Predicción
if st.button("Predecir Precio"):
    # Creamos el DataFrame usando la variable que definimos arriba
    input_data = pd.DataFrame([[bedrooms, bathrooms, surface]], columns=nombres_columnas)
    
    prediccion = modelo.predict(input_data)
    
    resultado = "Precio Alto" if prediccion[0] == 1 else "Precio Bajo"
    st.write(f"### El resultado de la clasificación es: **{resultado}**")
