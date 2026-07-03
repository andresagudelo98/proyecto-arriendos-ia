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

if st.button("Predecir Precio"):
    # 1. Definimos los nombres que sabemos que tu modelo espera
    # (Asegúrate de que estos 3 nombres coincidan EXACTAMENTE con los que usaste al entrenar)
    nombres_columnas = ['bedrooms', 'bathrooms', 'surface_total']
    
    # 2. Creamos el DataFrame
    input_data = pd.DataFrame([[bedrooms, bathrooms, surface]], columns=nombres_columnas)
    
    # 3. Realizamos la predicción
    prediccion = modelo.predict(input_data)
    
    # 4. Resultado
    resultado = "Precio Alto" if prediccion[0] == 1 else "Precio Bajo"
    st.write(f"### El resultado de la clasificación es: **{resultado}**")
