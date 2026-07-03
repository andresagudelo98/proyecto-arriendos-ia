import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Cargar modelo y columnas
# Asegúrate de que estos archivos estén en la misma carpeta que app.py en GitHub
modelo = joblib.load('modelo_arriendos.pkl')
columnas_modelo = joblib.load('columnas_modelo.pkl')

st.title("Predicción de Precios de Arriendo")
st.write("Ingresa los datos para clasificar el inmueble.")

# 2. Inputs para el usuario
bedrooms = st.number_input("Habitaciones", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Baños", min_value=1, max_value=5, value=1)
surface = st.number_input("Superficie Total (m2)", min_value=20, max_value=500, value=50)

# Agrega aquí todas las ciudades que estaban en tu dataset original
lista_ciudades = ["Medellín", "Bogotá D.C", "Envigado", "Manizales", "Cali", "Barranquilla", "Pereira"]
l3 = st.selectbox("Ciudad/Sector", lista_ciudades)

# 3. Lógica de predicción
if st.button("Predecir Precio"):
    # Crear un diccionario inicializado con ceros para TODAS las columnas del modelo
    # Esto evita el KeyError y garantiza que el DataFrame sea correcto
    input_data = {col: 0 for col in columnas_modelo}
    
    # Llenar valores numéricos
    input_data['bedrooms'] = bedrooms
    input_data['bathrooms'] = bathrooms
    input_data['surface_total'] = surface
    
    # Llenar la ubicación seleccionada dinámicamente
    # Busca la columna que contenga el nombre de la ciudad
    encontrado = False
    for col in columnas_modelo:
        if l3.lower() in col.lower():
            input_data[col] = 1
            encontrado = True
            break
    
    # Crear el DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ordenar columnas según lo que espera el modelo
    input_df = input_df[columnas_modelo]
    
    # Predicción
    prediccion = modelo.predict(input_df)
    
    # Mostrar resultado
    resultado = "Precio Alto" if prediccion[0] == 1 else "Precio Bajo"
    st.write(f"### El resultado de la clasificación es: **{resultado}**")
