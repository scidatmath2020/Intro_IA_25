# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 09:51:23 2025

@author: Usuario
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Función de segmentación (la tuya, con pequeña modificación) ---
def segmentar_kmeans(df, n_clusters=5):
    """
    Aplica PCA + KMeans a un DataFrame numérico.
    """
    # Escalado (añadido para mejor rendimiento de K-Means)
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df)
    
    # PCA
    pca = PCA(n_components=2)
    datos_reducidos = pca.fit_transform(datos_escalados)
    
    # K-Means
    modelo_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    etiquetas = modelo_kmeans.fit_predict(datos_reducidos)
    centroides = modelo_kmeans.cluster_centers_
    
    # Visualización
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
    
    # DataFrame con resultados
    df_result = df.copy()
    df_result['Cluster'] = etiquetas
    
    return df_result, fig

# --- Interfaz de Streamlit ---
st.title("Segmentación con K-Means + PCA")

# 1. Subir archivo
archivo = st.file_uploader("Sube tu CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    
    # 2. Seleccionar columnas numéricas
    columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
    columnas_seleccionadas = st.multiselect(
        "Selecciona columnas numéricas para el clustering",
        columnas_numericas,
        default=columnas_numericas
    )
    
    if columnas_seleccionadas:
        df_numerico = df[columnas_seleccionadas]
        
        # 3. Seleccionar número de clusters
        n_clusters = st.slider(
            "Número de clusters",
            min_value=2,
            max_value=10,
            value=5
        )
        
        # 4. Aplicar clustering
        if st.button("Ejecutar segmentación"):
            st.write("### Resultados del clustering")
            
            df_resultado, figura = segmentar_kmeans(df_numerico, n_clusters)
            
            # Mostrar gráfico
            st.pyplot(figura)
            
            # Mostrar dataframe con clusters
            st.write("### Datos con clusters asignados")
            st.dataframe(df_resultado)
            
            # Descargar resultados
            st.download_button(
                label="Descargar resultados como CSV",
                data=df_resultado.to_csv(index=False).encode('utf-8'),
                file_name='resultados_clustering.csv',
                mime='text/csv'
            )

# --- Instrucciones ---
st.sidebar.markdown("""
### Instrucciones:
1. Sube un archivo CSV con datos numéricos
2. Selecciona las columnas para el clustering
3. Elige el número de clusters
4. Haz clic en **Ejecutar segmentación**
""")