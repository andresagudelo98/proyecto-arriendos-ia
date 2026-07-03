import streamlit as st
import joblib
import os

# Esta línea busca el modelo exactamente en la misma carpeta donde está app.py
# Esto funciona tanto en tu PC como en la nube de Streamlit.
nombre_archivo_modelo = 'modelo_arriendos.pkl'

try:
    # Carga el modelo
    modelo = joblib.load(nombre_archivo_modelo)
except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo '{nombre_archivo_modelo}'. Asegúrate de que esté en la raíz del repositorio.")
    st.stop()
