# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:29:31 2025

@author: SciData
"""

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # --- Título ---
# st.title("Clasificación supervisada con KNN + PCA (2 componentes)")

# # --- Subir archivo de entrenamiento ---
# archivo_entrenamiento = st.file_uploader("Sube tu CSV con datos etiquetados", type=["csv"])

# if archivo_entrenamiento:
#     df = pd.read_csv(archivo_entrenamiento)
#     columnas_numericas = df.select_dtypes(include=['number', 'float']).columns.tolist()

#     st.write("Vista previa del archivo:")
#     st.dataframe(df.head())

#     col_objetivo = st.selectbox("Selecciona la columna objetivo", df.columns)
#     columnas_predictoras = st.multiselect(
#         "Selecciona columnas predictoras", 
#         columnas_numericas, 
#         default=[col for col in columnas_numericas if col != col_objetivo]
#     )

#     if columnas_predictoras and col_objetivo:
#         X = df[columnas_predictoras]
#         y = df[col_objetivo]

#         # Escalado
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         # PCA (2 componentes)
#         pca = PCA(n_components=2)
#         X_pca = pca.fit_transform(X_scaled)

#         # División
#         X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

#         # Entrenar modelo KNN
#         n_neighbors = st.slider("Número de vecinos (k)", 1, 15, 5)
#         modelo = KNeighborsClassifier(n_neighbors=n_neighbors)
#         modelo.fit(X_train, y_train)
#         y_pred = modelo.predict(X_test)

#         # Métricas
#         st.subheader("Evaluación del modelo")
#         st.write(f"**Exactitud (accuracy):** {accuracy_score(y_test, y_pred):.2f}")
#         st.text("Reporte de clasificación:")
#         st.text(classification_report(y_test, y_pred))

#         # Matriz de confusión
#         cm = confusion_matrix(y_test, y_pred)
#         fig, ax = plt.subplots()
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
#         ax.set_xlabel("Predicción")
#         ax.set_ylabel("Real")
#         st.pyplot(fig)

#         # Guardar para uso posterior
#         st.session_state["modelo_knn"] = modelo
#         st.session_state["scaler"] = scaler
#         st.session_state["pca"] = pca
#         st.session_state["columnas_x"] = columnas_predictoras

# # --- Subir archivo de predicción ---
# st.markdown("---")
# st.subheader("Predicción sobre nuevos datos")

# archivo_nuevos = st.file_uploader("Sube un CSV con nuevos datos (sin columna objetivo)", type=["csv"], key="nuevos")

# if archivo_nuevos and "modelo_knn" in st.session_state:
#     df_nuevos = pd.read_csv(archivo_nuevos)

#     if set(st.session_state["columnas_x"]).issubset(set(df_nuevos.columns)):
#         X_nuevos = df_nuevos[st.session_state["columnas_x"]]
#         X_nuevos_scaled = st.session_state["scaler"].transform(X_nuevos)
#         X_nuevos_pca = st.session_state["pca"].transform(X_nuevos_scaled)
#         predicciones = st.session_state["modelo_knn"].predict(X_nuevos_pca)

#         df_resultado = df_nuevos.copy()
#         df_resultado["Prediccion"] = predicciones

#         st.write("### Nuevas predicciones")
#         st.dataframe(df_resultado)

#         st.download_button(
#             "Descargar predicciones como CSV",
#             data=df_resultado.to_csv(index=False).encode('utf-8'),
#             file_name="predicciones_pca_knn.csv",
#             mime="text/csv"
#         )
#     else:
#         st.error("❌ Las columnas del nuevo archivo no coinciden con las usadas para entrenar el modelo.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuración de la página
st.set_page_config(page_title="KNN + PCA Classifier", layout="wide")

# --- Título ---
st.title("🧠 Clasificación con KNN + PCA")

# --- Subir archivo de entrenamiento ---
with st.expander("🔽 PASO 1: Subir datos de entrenamiento", expanded=True):
    archivo_entrenamiento = st.file_uploader("Sube tu CSV con datos etiquetados", type=["csv"], key="train")

    if archivo_entrenamiento:
        try:
            df = pd.read_csv(archivo_entrenamiento)
            columnas_numericas = df.select_dtypes(include=['number', 'float']).columns.tolist()

            st.success("✅ Archivo cargado correctamente")
            st.write("**Vista previa:**")
            st.dataframe(df.head(3))

            col_objetivo = st.selectbox("Selecciona la columna objetivo", df.columns)
            columnas_predictoras = st.multiselect(
                "Selecciona columnas predictoras", 
                columnas_numericas, 
                default=[col for col in columnas_numericas if col != col_objetivo]
            )

            if columnas_predictoras and col_objetivo:
                # Procesamiento de datos
                with st.spinner("Procesando datos..."):
                    X = df[columnas_predictoras]
                    y = df[col_objetivo]

                    # Escalado
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # PCA
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)

                    # División train-test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_pca, y, test_size=0.3, random_state=42
                    )

                # Entrenamiento del modelo
                with st.expander("⚙️ Configuración del modelo KNN"):
                    n_neighbors = st.slider("Número de vecinos (k)", 1, 15, 5)
                    modelo = KNeighborsClassifier(n_neighbors=n_neighbors)
                    
                    with st.spinner("Entrenando modelo..."):
                        modelo.fit(X_train, y_train)
                        y_pred = modelo.predict(X_test)

                # Resultados
                st.subheader("📊 Resultados del modelo")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Exactitud (accuracy)", f"{accuracy_score(y_test, y_pred):.2%}")
                
                with col2:
                    st.download_button(
                        "📥 Descargar reporte completo",
                        data=classification_report(y_test, y_pred, output_dict=True),
                        file_name="reporte_clasificacion.txt"
                    )

                # Visualización
                tab1, tab2 = st.tabs(["Matriz de confusión", "Componentes PCA"])
                
                with tab1:
                    fig1, ax1 = plt.subplots()
                    sns.heatmap(
                        confusion_matrix(y_test, y_pred), 
                        annot=True, fmt='d', 
                        cmap='Blues',
                        xticklabels=modelo.classes_, 
                        yticklabels=modelo.classes_
                    )
                    ax1.set_xlabel("Predicción")
                    ax1.set_ylabel("Real")
                    st.pyplot(fig1)
                
                with tab2:
                    fig2, ax2 = plt.subplots()
                    scatter = ax2.scatter(
                        X_pca[:, 0], X_pca[:, 1], 
                        c=y.astype('category').cat.codes, 
                        alpha=0.6
                    )
                    ax2.set_xlabel("Primer componente principal")
                    ax2.set_ylabel("Segundo componente principal")
                    plt.colorbar(scatter)
                    st.pyplot(fig2)

                # Guardar estado para predicciones
                st.session_state.update({
                    "modelo_knn": modelo,
                    "scaler": scaler,
                    "pca": pca,
                    "columnas_x": columnas_predictoras
                })

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")

# --- Predicción de nuevos datos ---
with st.expander("🔮 PASO 2: Predecir nuevos datos"):
    archivo_nuevos = st.file_uploader(
        "Sube un CSV con nuevos datos (sin columna objetivo)", 
        type=["csv"], 
        key="predict"
    )

    if archivo_nuevos and "modelo_knn" in st.session_state:
        try:
            df_nuevos = pd.read_csv(archivo_nuevos)
            
            if set(st.session_state["columnas_x"]).issubset(set(df_nuevos.columns)):
                with st.spinner("Realizando predicciones..."):
                    X_nuevos = df_nuevos[st.session_state["columnas_x"]]
                    X_nuevos_scaled = st.session_state["scaler"].transform(X_nuevos)
                    X_nuevos_pca = st.session_state["pca"].transform(X_nuevos_scaled)
                    predicciones = st.session_state["modelo_knn"].predict(X_nuevos_pca)

                    df_resultado = df_nuevos.copy()
                    df_resultado["Prediccion"] = predicciones

                st.success("✅ Predicciones completadas")
                st.dataframe(df_resultado)

                st.download_button(
                    "📤 Descargar predicciones",
                    data=df_resultado.to_csv(index=False).encode('utf-8'),
                    file_name="predicciones.csv",
                    mime="text/csv"
                )
            else:
                missing = set(st.session_state["columnas_x"]) - set(df_nuevos.columns)
                st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
                
        except Exception as e:
            st.error(f"Error en predicción: {str(e)}")
