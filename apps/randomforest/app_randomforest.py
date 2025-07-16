# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:14:39 2025

@author: SciData
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# --- Título ---
st.title("Clasificación supervisada con Random Forest")

# --- Subir archivo de entrenamiento ---
archivo_entrenamiento = st.file_uploader("Sube tu archivo CSV con datos etiquetados (con columna objetivo)", type=["csv"])

if archivo_entrenamiento:
    df = pd.read_csv(archivo_entrenamiento)
    columnas = df.select_dtypes(include=['number', 'float']).columns.tolist()
    st.write("Vista previa del archivo:")
    st.dataframe(df.head())

    # --- Seleccionar columna objetivo y predictoras ---
    col_objetivo = st.selectbox("Selecciona la columna objetivo (target)", df.columns)
    columnas_pred = st.multiselect("Selecciona las columnas predictoras (features)", columnas, default=[c for c in columnas if c != col_objetivo])

    if columnas_pred and col_objetivo:
        X = df[columnas_pred]
        y = df[col_objetivo]

        # Escalado opcional
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        modelo = RandomForestClassifier(random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        st.subheader("Resultados del modelo")

        # Métricas
        st.write(f"**Exactitud (accuracy):** {accuracy_score(y_test, y_pred):.2f}")
        st.text("Reporte de clasificación:")
        st.text(classification_report(y_test, y_pred))

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)

        # Guardar modelo y scaler internamente para usarlo en predicciones
        st.session_state['modelo_entrenado'] = modelo
        st.session_state['scaler'] = scaler
        st.session_state['columnas_esperadas'] = columnas_pred

        # Descargar modelo entrenado (opcionalmente puedes usar joblib para persistencia real)

# --- Predicción de nuevos datos ---
st.markdown("---")
st.subheader("¿Tienes nuevos datos para predecir la clase?")

archivo_nuevos = st.file_uploader("Sube un archivo CSV con nuevas observaciones (sin la columna objetivo)", type=["csv"], key="nuevos")

if archivo_nuevos and 'modelo_entrenado' in st.session_state:
    df_nuevos = pd.read_csv(archivo_nuevos)

    # Verificación
    if set(st.session_state['columnas_esperadas']) <= set(df_nuevos.columns):
        X_nuevos = df_nuevos[st.session_state['columnas_esperadas']]
        X_nuevos_scaled = st.session_state['scaler'].transform(X_nuevos)
        predicciones = st.session_state['modelo_entrenado'].predict(X_nuevos_scaled)

        df_resultado = df_nuevos.copy()
        df_resultado['Prediccion'] = predicciones

        st.write("### Nuevos datos con predicción")
        st.dataframe(df_resultado)

        st.download_button(
            "Descargar resultados como CSV",
            data=df_resultado.to_csv(index=False).encode('utf-8'),
            file_name="predicciones.csv",
            mime="text/csv"
        )
    else:
        st.error("❌ Las columnas del archivo no coinciden con las que usaste para entrenar el modelo.")
