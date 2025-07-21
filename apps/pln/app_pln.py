# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:03:54 2025

@author: SciData
"""

# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuración de rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_sentimiento.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
DICT_PATH = os.path.join(BASE_DIR, "diccionario_clases.pkl")

# --- Verificación de archivos ---
st.set_page_config(page_title="Clasificador de Sentimientos")

# Verificación silenciosa para producción
try:
    if not all(os.path.exists(path) for path in [MODEL_PATH, TOKENIZER_PATH, DICT_PATH]):
        st.error("Error crítico: Archivos del modelo no encontrados. Contacte al administrador.")
        st.stop()
except Exception as e:
    st.error(f"Error verificando archivos: {str(e)}")
    st.stop()

# --- Carga de recursos con caché ---
@st.cache_resource
def cargar_recursos():
    try:
        modelo = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        with open(DICT_PATH, "rb") as f:
            diccionario = pickle.load(f)
        return modelo, tokenizer, diccionario
    except Exception as e:
        st.error(f"Error cargando recursos: {str(e)}")
        st.stop()

modelo, tokenizer, diccionario_clases = cargar_recursos()

# --- Interfaz de usuario ---
st.title("Clasificación de Frases por Sentimiento (LSTM)")
archivo = st.file_uploader("Sube un CSV con una columna de frases", type=["csv"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo)
        st.write("Vista previa del archivo:")
        st.dataframe(df.head())

        if df.shape[1] != 1:
            st.error("El archivo debe tener solo una columna de texto.")
        else:
            columna_texto = df.columns[0]
            frases = df[columna_texto].astype(str).values

            secuencias = tokenizer.texts_to_sequences(frases)
            X = pad_sequences(secuencias, maxlen=42)

            predicciones = modelo.predict(X)
            clases_predichas = [diccionario_clases[np.argmax(p)] for p in predicciones]

            df["Prediccion"] = clases_predichas
            df["Confianza"] = [f"{np.max(p)*100:.1f}%" for p in predicciones]

            st.write("### Frases clasificadas")
            st.dataframe(df)

            st.download_button(
                label="Descargar resultados",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="clasificacion_frases.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")

st.sidebar.markdown("""
### Instrucciones:
1. Sube un CSV con una columna de frases
2. Se usará un modelo LSTM previamente entrenado
3. Los resultados incluyen nivel de confianza
4. Puedes descargar los resultados
""")
