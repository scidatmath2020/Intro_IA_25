# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 09:51:23 2025

@author: SciData
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Función de entrenamiento del modelo ---
def entrenar_segmentador(df, n_clusters=5):
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    datos_reducidos = pca.fit_transform(datos_escalados)

    modelo_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    etiquetas = modelo_kmeans.fit_predict(datos_reducidos)

    centroides = modelo_kmeans.cluster_centers_

    df_result = df.copy()
    df_result['Cluster'] = etiquetas

    return df_result, scaler, pca, modelo_kmeans, datos_reducidos, etiquetas, centroides

# --- Streamlit App ---
st.title("Segmentación con K-Means + PCA")

# Subir archivo base
archivo = st.file_uploader("Sube tu archivo CSV para segmentación (ej. clientes_segmentacion.csv)", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    columnas_numericas = df.select_dtypes(include='number').columns.tolist()

    columnas_seleccionadas = st.multiselect(
        "Selecciona columnas numéricas para clustering",
        columnas_numericas,
        default=columnas_numericas
    )

    if columnas_seleccionadas:
        df_numerico = df[columnas_seleccionadas]

        n_clusters = st.slider("Número de clusters", 2, 10, value=5)

        if st.button("Ejecutar segmentación"):
            st.write("### Resultados del clustering")

            df_resultado, scaler, pca, modelo_kmeans, datos_reducidos, etiquetas, centroides = entrenar_segmentador(df_numerico, n_clusters)

            # --- Visualización base (datos originales) ---
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(n_clusters):
                ax.scatter(
                    datos_reducidos[etiquetas == i, 0],
                    datos_reducidos[etiquetas == i, 1],
                    label=f'Cluster {i+1}',
                    alpha=0.7
                )
            ax.scatter(
                centroides[:, 0], centroides[:, 1],
                s=250, c='black', marker='*', label='Centroides'
            )
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.set_title('Segmentación con K-Means (reducción PCA)')
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)
            st.dataframe(df_resultado)

            st.download_button(
                label="Descargar resultados como CSV",
                data=df_resultado.to_csv(index=False).encode('utf-8'),
                file_name='resultados_clustering.csv',
                mime='text/csv'
            )

            # --- Predicción sobre nuevos datos ---
            st.markdown("---")
            st.write("### Predicción de clúster para nuevos datos")
            archivo_nuevo = st.file_uploader("Sube archivo CSV con nuevos datos (ej. clientes_nuevos.csv)", type=["csv"], key="nuevo")

            if archivo_nuevo is not None:
                df_nuevo = pd.read_csv(archivo_nuevo)

                if list(df_nuevo.columns) != columnas_seleccionadas:
                    st.error("❌ Las columnas del nuevo archivo no coinciden exactamente con las seleccionadas.")
                else:
                    datos_nuevos_escalados = scaler.transform(df_nuevo)
                    datos_nuevos_reducidos = pca.transform(datos_nuevos_escalados)
                    etiquetas_nuevas = modelo_kmeans.predict(datos_nuevos_reducidos)

                    df_nuevo_resultado = df_nuevo.copy()
                    df_nuevo_resultado['Cluster_Predicho'] = etiquetas_nuevas

                    st.write("### Nuevos datos con clústeres asignados")
                    st.dataframe(df_nuevo_resultado)

                    # --- Nueva visualización con ambos ---
                    fig2, ax2 = plt.subplots(figsize=(10, 6))

                    # Originales
                    for i in range(n_clusters):
                        ax2.scatter(
                            datos_reducidos[etiquetas == i, 0],
                            datos_reducidos[etiquetas == i, 1],
                            label=f'Cluster {i+1}',
                            alpha=0.5
                        )

                    # Nuevos puntos
                    for i in range(n_clusters):
                        puntos = datos_nuevos_reducidos[etiquetas_nuevas == i]
                        ax2.scatter(
                            puntos[:, 0],
                            puntos[:, 1],
                            marker='x',
                            s=120,
                            label=f'Nuevos - Cluster {i+1}'
                        )

                    ax2.scatter(
                        centroides[:, 0], centroides[:, 1],
                        s=250, c='black', marker='*', label='Centroides'
                    )
                    ax2.set_xlabel('PCA 1')
                    ax2.set_ylabel('PCA 2')
                    ax2.set_title('Segmentación con nuevos datos añadidos')
                    ax2.legend()
                    ax2.grid(True)

                    st.pyplot(fig2)

                    st.download_button(
                        label="Descargar predicciones como CSV",
                        data=df_nuevo_resultado.to_csv(index=False).encode('utf-8'),
                        file_name='predicciones_clusters.csv',
                        mime='text/csv'
                    )

# Instrucciones
st.sidebar.markdown("""
### Instrucciones:
1. Sube un archivo CSV con datos numéricos (ej. clientes_segmentacion.csv)
2. Selecciona columnas y número de clústeres
3. Ejecuta la segmentación
4. Después, sube nuevos datos (ej. clientes_nuevos.csv) para predecir su clúster
""")
