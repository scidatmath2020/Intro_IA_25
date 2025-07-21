# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:30:00 2025

@author: SciData
"""

import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Función para transformar etiquetas de estrellas a sentimiento ---
def estrellas_a_sentimiento(label):
    estrellas = int(label[0])
    etiquetas = {
        1: "muy negativo",
        2: "negativo",
        3: "neutro",
        4: "positivo",
        5: "muy positivo"
    }
    return etiquetas.get(estrellas, "desconocido")

# --- Carga del modelo con caché ---
@st.cache_resource
def cargar_modelo():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classifier = cargar_modelo()

# --- Interfaz principal ---
st.title("Clasificador de Sentimientos con BERT (Multilingüe)")
st.write("Esta aplicación clasifica frases de un archivo CSV según su polaridad emocional.")

# 1. Subida del archivo
archivo = st.file_uploader("Sube tu archivo CSV con frases", type=["csv"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo)
        st.success("✅ Archivo cargado correctamente.")

        # 2. Selección de columna
        columna = st.selectbox("Selecciona la columna que contiene las frases:", df.columns)
        frases = df[columna].astype(str).tolist()

        # 3. Clasificación
        if st.button("Ejecutar análisis de sentimientos"):
            st.info("Clasificando, espera un momento...")

            resultados = classifier(frases)

            df["estrellas"] = [r["label"] for r in resultados]
            df["confianza"] = [round(r["score"], 4) for r in resultados]
            df["sentimiento"] = df["estrellas"].apply(estrellas_a_sentimiento)

            st.write("### Resultados:")
            st.dataframe(df[[columna, "sentimiento", "estrellas", "confianza"]])

            st.download_button(
                label="Descargar resultados como CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="resultados_sentimientos.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")

else:
    st.info("Carga un archivo CSV para comenzar.")

# --- Instrucciones en la barra lateral ---
st.sidebar.markdown("""
### Instrucciones:
1. Sube un archivo CSV con una columna de frases
2. Selecciona la columna que contiene el texto
3. Haz clic en **Ejecutar análisis de sentimientos**
4. Descarga los resultados como archivo CSV
""")
