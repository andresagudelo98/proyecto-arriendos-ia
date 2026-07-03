import streamlit as st
import joblib
import pandas as pd

modelo = joblib.load('modelo_arriendos.pkl')

# DEBUG: Esto imprimirá en la pantalla de la app lo que el modelo espera
st.write("---")
st.write("CANTIDAD DE COLUMNAS ESPERADAS:", len(modelo.feature_names_in_))
st.write("NOMBRES DE COLUMNAS ESPERADAS:", list(modelo.feature_names_in_))
st.write("---")

# Inputs...
bedrooms = st.number_input("Habitaciones", 1, 10, 2)
bathrooms = st.number_input("Baños", 1, 5, 1)
surface = st.number_input("Superficie Total (m2)", 20, 500, 50)

if st.button("Predecir Precio"):
    # Si la lista de arriba NO tiene 3 elementos, aquí está el problema.
    # Por ahora, ignoremos las columnas y pasemos los datos como una lista simple
    # para ver si el modelo al menos recibe los datos:
    data = pd.DataFrame([[bedrooms, bathrooms, surface]])
    
    st.write("Dataframe creado correctamente")
    prediccion = modelo.predict(data)
    st.write("Predicción:", prediccion)
