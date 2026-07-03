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
    # 3. Preparación de datos con Dummies
    # Creamos la estructura base
    input_dict = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'surface_total': surface,
        'l3_Medellín': 1 if l3 == "Medellín" else 0,
        'l3_Bogotá D.C': 1 if l3 == "Bogotá D.C" else 0,
        'l3_Envigado': 1 if l3 == "Envigado" else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # IMPORTANTE: Aseguramos que las columnas estén en el orden exacto que el modelo espera
    input_df = input_df[columnas_modelo]
    
    # 4. Predicción
    prediccion = modelo.predict(input_df)
    
    resultado = "Precio Alto" if prediccion[0] == 1 else "Precio Bajo"
    st.write(f"### El resultado es: **{resultado}**")
