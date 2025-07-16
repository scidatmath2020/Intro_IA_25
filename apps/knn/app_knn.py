# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:29:31 2025

@author: SciData
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# --- Función de clasificación supervisada ---
def clasificar_knn(df, columna_objetivo, columnas_predictoras, n_vecinos=5):
    """
    Aplica PCA + KNN a un DataFrame con columnas numéricas y una etiqueta.
    """
    X = df[columnas_predictoras]
    y = df[columna_objetivo]

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # KNN
    modelo_knn = KNeighborsClassifier(n_neighbors=n_vecinos)
    modelo_knn.fit(X_pca, y)
    predicciones = modelo_knn.predict(X_pca)

    # Visualización
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=pd.factorize(y)[0],
        cmap='tab10',
        alpha=0.7,
        label=y
    )
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Clasificación con KNN (reducción PCA)')
    legend_labels = pd.Series(y.unique()).sort_values().tolist()
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Clases")
    ax.grid(True)

    # DataFrame con resultados
    df_result = df.copy()
    df_result['Prediccion'] = predicciones

    return df_result, fig, modelo_knn, scaler, pca

# --- Interfaz de Streamlit ---
st.title("Clasificación con KNN + PCA")

# 1. Subir archivo
archivo = st.file_uploader("Sube tu CSV con columna objetivo (etiquetas)", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.write("Vista previa del archivo:")
    st.dataframe(df.head())

    # 2. Seleccionar columna objetivo
    columna_objetivo = st.selectbox("Selecciona la columna objetivo", df.columns)

    # 3. Seleccionar columnas numéricas
    columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
    columnas_predictoras = st.multiselect(
        "Selecciona columnas numéricas para la clasificación",
        [col for col in columnas_numericas if col != columna_objetivo],
        default=[col for col in columnas_numericas if col != columna_objetivo]
    )

    if columnas_predictoras:
        # 4. Número de vecinos
        n_vecinos = st.slider(
            "Número de vecinos (k)",
            min_value=1,
            max_value=15,
            value=5
        )

        # 5. Ejecutar clasificación
        if st.button("Ejecutar clasificación"):
            st.write("### Resultados de la clasificación")

            df_resultado, figura, modelo_knn, scaler, pca = clasificar_knn(
                df, columna_objetivo, columnas_predictoras, n_vecinos
            )

            st.pyplot(figura)

            st.write("### Datos con predicción (entrenamiento)")
            st.dataframe(df_resultado)

            st.download_button(
                label="Descargar resultados como CSV",
                data=df_resultado.to_csv(index=False).encode('utf-8'),
                file_name='resultados_knn.csv',
                mime='text/csv'
            )

            # Guardar modelos para predicciones futuras
            st.session_state["modelo_knn"] = modelo_knn
            st.session_state["scaler"] = scaler
            st.session_state["pca"] = pca
            st.session_state["columnas_predictoras"] = columnas_predictoras

# 6. Predicción sobre nuevos datos
st.markdown("---")
st.subheader("¿Tienes nuevos datos sin etiqueta? Haz predicciones aquí:")

archivo_nuevos = st.file_uploader("Sube CSV sin columna objetivo", type=["csv"], key="nuevos")

if archivo_nuevos and "modelo_knn" in st.session_state:
    df_nuevos = pd.read_csv(archivo_nuevos)

    columnas_requeridas = st.session_state["columnas_predictoras"]

    if set(columnas_requeridas).issubset(set(df_nuevos.columns)):
        X_nuevos = df_nuevos[columnas_requeridas]
        X_nuevos_scaled = st.session_state["scaler"].transform(X_nuevos)
        X_nuevos_pca = st.session_state["pca"].transform(X_nuevos_scaled)
        predicciones = st.session_state["modelo_knn"].predict(X_nuevos_pca)

        df_predicciones = df_nuevos.copy()
        df_predicciones["Prediccion"] = predicciones

        st.write("### Predicciones para nuevos datos")
        st.dataframe(df_predicciones)

        st.download_button(
            label="Descargar predicciones como CSV",
            data=df_predicciones.to_csv(index=False).encode('utf-8'),
            file_name="predicciones_nuevos_knn.csv",
            mime="text/csv"
        )
    else:
        st.error("Las columnas del archivo nuevo no coinciden con las utilizadas para entrenar el modelo.")

# --- Instrucciones ---
st.sidebar.markdown("""
### Instrucciones:
1. Sube un archivo CSV con una columna objetivo (etiquetas)
2. Selecciona las columnas numéricas predictoras
3. Ajusta el número de vecinos (k)
4. Haz clic en **Ejecutar clasificación**
5. Opcional: Sube nuevos datos sin etiqueta para predecir sus clases
""")
