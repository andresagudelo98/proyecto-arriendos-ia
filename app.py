import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo y las columnas directamente desde la carpeta del repositorio
# Al no poner ruta absoluta, Streamlit los buscará en la misma carpeta que app.py
try:
    modelo = joblib.load('modelo_arriendos.pkl')
    columnas_modelo = joblib.load('columnas_modelo.pkl')
except FileNotFoundError:
    st.error("No se encontraron los archivos del modelo en la carpeta del repositorio.")
    st.stop()

st.title("Predicción de Precios de Arriendo")
st.write("Esta aplicación determina si un inmueble es de **Precio Alto** o **Precio Bajo**.")

# Inputs para el usuario
bedrooms = st.number_input("Habitaciones", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Baños", min_value=1, max_value=5, value=1)
surface = st.number_input("Superficie Total (m2)", min_value=20, max_value=500, value=50)

if st.button("Predecir Precio"):
    # 1. Crear el DataFrame con los nombres correctos que espera tu modelo
    input_data = pd.DataFrame([[bedrooms, bathrooms, surface]], columns=columnas_modelo)
    
    # 2. Realizar la predicción
    prediccion = modelo.predict(input_data)
    
    # 3. Mostrar el resultado
    resultado = "Precio Alto" if prediccion[0] == 1 else "Precio Bajo"
    st.write(f"### El resultado de la clasificación es: **{resultado}**")
