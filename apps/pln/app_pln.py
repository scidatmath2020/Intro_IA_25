# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 18:03:54 2025

@author: SciData
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Cargar modelo, tokenizer y clases ---
modelo = load_model("modelo_sentimiento.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("diccionario_clases.pkl", "rb") as f:
    diccionario_clases = pickle.load(f)

# --- Configuración de app ---
st.title("Clasificación de Frases por Sentimiento (LSTM)")
archivo = st.file_uploader("Sube un CSV con una columna de frases", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    st.write("Vista previa del archivo:")
    st.dataframe(df.head())

    if df.shape[1] != 1:
        st.error("El archivo debe tener solo una columna de texto.")
    else:
        columna_texto = df.columns[0]
        frases = df[columna_texto].astype(str).values

        secuencias = tokenizer.texts_to_sequences(frases)
        X = pad_sequences(secuencias, maxlen=tokenizer.document_count)

        predicciones = modelo.predict(X)
        clases_predichas = [diccionario_clases[np.argmax(p)] for p in predicciones]

        df["Prediccion"] = clases_predichas

        st.write("### Frases clasificadas")
        st.dataframe(df)

        st.download_button(
            label="Descargar resultados",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="clasificacion_frases.csv",
            mime="text/csv"
        )

st.sidebar.markdown("""
### Instrucciones:
1. Sube un CSV con una columna de frases
2. Se usará un modelo LSTM previamente entrenado para predecir su clase
3. Puedes descargar los resultados
""")
