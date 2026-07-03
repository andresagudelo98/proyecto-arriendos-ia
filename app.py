import streamlit as st
import joblib
import pandas as pd

# 1. Cargar el modelo
modelo = joblib.load('modelo_arriendos.pkl')

# 2. EXTRAER LOS NOMBRES REALES DEL MODELO
# Esto nos dirá exactamente qué nombres espera (ej: 'bedrooms', 'bathrooms', 'surface_total')
columnas_reales = modelo.feature_names_in_

st.title("Predicción de Precios de Arriendo")

# Inputs
bedrooms = st.number_input("Habitaciones", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Baños", min_value=1, max_value=5, value=1)
surface = st.number_input("Superficie Total (m2)", min_value=20, max_value=500, value=50)

if st.button("Predecir Precio"):
    # 3. Crear el DataFrame usando los nombres que el modelo nos acaba de dar
    input_data = pd.DataFrame([[bedrooms, bathrooms, surface]], columns=columnas_reales)
    
    # 4. Predecir
    prediccion = modelo.predict(input_data)
    
    resultado = "Precio Alto" if prediccion[0] == 1 else "Precio Bajo"
    st.write(f"### El resultado de la clasificación es: **{resultado}**")
    
    # Debug opcional: esto te mostrará en pantalla qué nombres está usando el modelo
    st.write("Columnas detectadas por el modelo:", list(columnas_reales))
