import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Segmentación de datos conocidos
def entrenar_modelo(df, n_clusters):
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    datos_pca = pca.fit_transform(datos_escalados)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    etiquetas = kmeans.fit_predict(datos_pca)

    return etiquetas, scaler, pca, kmeans, datos_pca

# Predicción de datos nuevos
def predecir_nuevos(df_nuevos, scaler, pca, kmeans):
    escalados = scaler.transform(df_nuevos)
    pca_reducidos = pca.transform(escalados)
    predicciones = kmeans.predict(pca_reducidos)
    return predicciones, pca_reducidos

# App Streamlit
st.title("Segmentación con KMeans + PCA")

# 1. Subir datos base
archivo = st.file_uploader("Sube tu archivo CSV con datos conocidos", type=["csv"])

if archivo:
    df_base = pd.read_csv(archivo)
    columnas = df_base.select_dtypes(include='number').columns.tolist()

    st.write("Columnas numéricas detectadas:")
    st.write(columnas)

    columnas_uso = st.multiselect("Selecciona columnas para segmentación", columnas, default=columnas)
    n_clusters = st.slider("Selecciona número de clusters", 2, 10, 5)

    if st.button("Segmentar datos conocidos"):
        df_base_uso = df_base[columnas_uso]
        etiquetas, scaler, pca, kmeans, datos_pca = entrenar_modelo(df_base_uso, n_clusters)

        df_base_resultado = df_base.copy()
        df_base_resultado["Cluster"] = etiquetas

        # Gráfico base
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(n_clusters):
            ax.scatter(
                datos_pca[etiquetas == i, 0],
                datos_pca[etiquetas == i, 1],
                label=f"Cluster {i+1}",
                alpha=0.6
            )
        ax.set_title("Clustering de datos conocidos")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()
        st.pyplot(fig)

        st.write("### Datos conocidos con cluster asignado")
        st.dataframe(df_base_resultado)
        st.download_button(
            "Descargar CSV con clusters",
            data=df_base_resultado.to_csv(index=False).encode("utf-8"),
            file_name="clientes_segmentados.csv",
            mime="text/csv"
        )

        # 2. Subir datos nuevos
        st.markdown("---")
        st.subheader("¿Quieres predecir clústeres para nuevos clientes?")
        archivo_nuevos = st.file_uploader("Sube CSV con datos nuevos", type=["csv"], key="nuevos")

        if archivo_nuevos:
            df_nuevos = pd.read_csv(archivo_nuevos)

            # Validación
            if list(df_nuevos.columns) != columnas_uso:
                st.error("❌ Las columnas del archivo nuevo no coinciden con las seleccionadas.")
            else:
                etiquetas_nuevas, nuevos_pca = predecir_nuevos(df_nuevos, scaler, pca, kmeans)
                df_nuevos_resultado = df_nuevos.copy()
                df_nuevos_resultado["Cluster_Predicho"] = etiquetas_nuevas

                # Gráfico combinado
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                for i in range(n_clusters):
                    ax2.scatter(
                        datos_pca[etiquetas == i, 0],
                        datos_pca[etiquetas == i, 1],
                        label=f"Cluster {i+1} (conocidos)",
                        alpha=0.4
                    )
                    ax2.scatter(
                        nuevos_pca[etiquetas_nuevas == i, 0],
                        nuevos_pca[etiquetas_nuevas == i, 1],
                        marker='x',
                        s=100,
                        label=f"Nuevos - Cluster {i+1}"
                    )

                ax2.set_title("Datos conocidos y nuevos clientes")
                ax2.set_xlabel("PCA 1")
                ax2.set_ylabel("PCA 2")
                ax2.legend()
                st.pyplot(fig2)

                st.write("### Nuevos clientes con cluster asignado")
                st.dataframe(df_nuevos_resultado)

                st.download_button(
                    "Descargar predicciones",
                    data=df_nuevos_resultado.to_csv(index=False).encode("utf-8"),
                    file_name="clientes_nuevos_clasificados.csv",
                    mime="text/csv"
                )
