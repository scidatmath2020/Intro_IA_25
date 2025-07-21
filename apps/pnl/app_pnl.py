# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 22:07:42 2025

@author: SciData
"""

import os
import streamlit as st
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# --- Configuración de página ---
st.set_page_config(page_title="Clasificador de Sentimientos (spaCy)", layout="wide")

# --- Carga y configuración de recursos ---
@st.cache_resource
def cargar_pipeline():
    try:
        nlp = spacy.load("es_core_news_sm")
        if "spacytextblob" not in nlp.pipe_names:
            nlp.add_pipe("spacytextblob")
        return nlp
    except Exception as e:
        st.error(f"Error al cargar el modelo spaCy: {str(e)}")
        st.stop()

nlp = cargar_pipeline()

# --- Interfaz de usuario ---
st.title("Clasificación de Frases por Sentimiento (spaCy)")

archivo = st.file_uploader("Sube un archivo CSV con una sola columna de texto", type=["csv"])

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

            def obtener_sentimiento(texto):
                doc = nlp(texto)
                return doc._.blob.polarity

            df["Polaridad"] = [obtener_sentimiento(f) for f in frases]
            df["Clasificacion"] = df["Polaridad"].apply(
                lambda x: "Positivo" if x > 0.1 else ("Negativo" if x < -0.1 else "Neutral")
            )

            st.write("### Frases clasificadas por sentimiento")
            st.dataframe(df)

            st.download_button(
                label="Descargar resultados",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="resultados_sentimiento.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")

# --- Instrucciones en barra lateral ---
st.sidebar.markdown("""
### Instrucciones:
1. Sube un archivo **.csv** con una sola columna de frases en español.
2. Se usa un modelo de lenguaje spaCy con análisis de sentimiento.
3. Los resultados incluyen:
   - **Polaridad**: valor entre -1 (negativo) y 1 (positivo).
   - **Clasificación**: positiva, negativa o neutral.
4. Puedes descargar los resultados como archivo `.csv`.
""")
