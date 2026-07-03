import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# 1. Cargar modelo y columnas (usando ruta relativa para que funcione en cualquier lado)
# Si el archivo está en la misma carpeta que app.py, no necesita ruta larga.
modelo = joblib.load('modelo_arriendos.pkl')
columnas_modelo = joblib.load('columnas_modelo.pkl')

st.title("Predicción de Precios de Arriendo")

# 2. Inputs
bedrooms = st.number_input("Habitaciones", 1, 10, 2)
bathrooms = st.number_input("Baños", 1, 5, 1)
surface = st.number_input("Superficie Total (m2)", 20, 500, 50)
l3 = st.selectbox("Ciudad/Sector", ["Medellín", "Bogotá D.C", "Envigado"])


if st.button("Predecir Precio"):
    # 1. Crear un diccionario con ceros para TODAS las columnas del modelo
    # Esto asegura que el DataFrame tenga exactamente las 151 columnas
    input_data = {col: 0 for col in columnas_modelo}
    
    # 2. Llenar los valores básicos
    input_data['bedrooms'] = bedrooms
    input_data['bathrooms'] = bathrooms
    input_data['surface_total'] = surface
    
    # 3. Llenar la columna de la ciudad seleccionada
    # IMPORTANTE: El nombre debe coincidir exactamente con el que está en columnas_modelo
    # Probemos con el formato que vimos en tu archivo: 'l3_Medellín'
    col_ciudad = f"l3_{l3}" 
    if col_ciudad in input_data:
        input_data[col_ciudad] = 1
    
    # 4. Crear el DataFrame y ORDENARLO exactamente como el modelo espera
    input_df = pd.DataFrame([input_data])
    input_df = input_df[columnas_modelo] # Ahora sí tiene las 151 columnas
    
    # 5. Predecir
    prediccion = modelo.predict(input_df)
    
    resultado = "Precio Alto" if prediccion[0] == 1 else "Precio Bajo"
    st.write(f"### El resultado es: **{resultado}**")
