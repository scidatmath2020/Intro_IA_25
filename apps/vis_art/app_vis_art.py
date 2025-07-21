# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 14:00:00 2025

@author: SciData
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import tempfile
import shutil
from PIL import Image
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- Función: descomprimir archivo ZIP ---
def descomprimir_zip(archivo_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

# --- Función: crear generadores ---
def crear_generadores(directorio, tamano=(224, 224), batch=32):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(directorio, target_size=tamano, batch_size=batch,
                                            class_mode='categorical', subset='training')
    val_gen = datagen.flow_from_directory(directorio, target_size=tamano, batch_size=batch,
                                          class_mode='categorical', subset='validation')
    return train_gen, val_gen

# --- Función: crear modelo CNN (MobileNetV2) ---
@st.cache_resource
def crear_modelo(num_clases):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_clases, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Función: clasificar imágenes individuales ---
def clasificar_imagenes(modelo, lista_rutas, diccionario_clases):
    resultados = []
    rev_map = {v: k for k, v in diccionario_clases.items()}
    for ruta in lista_rutas:
        img = Image.open(ruta).convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(img) / 255.0, 0)
        pred = modelo.predict(arr, verbose=0)
        idx = int(np.argmax(pred))
        resultados.append({
            "archivo": os.path.basename(ruta),
            "clase_predicha": rev_map[idx],
            "confianza": round(float(np.max(pred)), 4)
        })
    return resultados

# --- Interfaz principal ---
st.title("Clasificador de Imágenes Multiclase")
st.write("Esta aplicación permite entrenar un modelo con imágenes organizadas en carpetas y clasificar nuevas imágenes.")

# --- Paso 1: Subida y entrenamiento ---
st.header("Paso 1: Entrenamiento del modelo")

zip_entrenamiento = st.file_uploader("Sube un archivo ZIP con carpetas (una por clase)", type=["zip"])

if zip_entrenamiento:
    with st.spinner("Procesando carpetas..."):
        dir_entrenamiento = descomprimir_zip(zip_entrenamiento)
        train_gen, val_gen = crear_generadores(dir_entrenamiento)
        st.success(f"✅ Dataset cargado: {train_gen.samples} imágenes en {len(train_gen.class_indices)} clases.")
        if st.button("Entrenar modelo"):
            modelo = crear_modelo(num_clases=len(train_gen.class_indices))
            modelo.fit(train_gen, validation_data=val_gen, epochs=5)
            st.session_state["modelo"] = modelo
            st.session_state["clases"] = train_gen.class_indices
            st.success("✅ Modelo entrenado correctamente.")
            shutil.rmtree(dir_entrenamiento)

# --- Paso 2: Clasificación de nuevas imágenes ---
st.header("Paso 2: Clasificación de imágenes")

if "modelo" not in st.session_state:
    st.info("Primero debes entrenar un modelo en el Paso 1.")
else:
    zip_imagenes = st.file_uploader("Sube un ZIP con imágenes a clasificar", type=["zip"])
    if zip_imagenes:
        with st.spinner("Clasificando imágenes..."):
            dir_test = descomprimir_zip(zip_imagenes)
            rutas = [os.path.join(dirpath, f)
                     for dirpath, _, files in os.walk(dir_test)
                     for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            resultados = clasificar_imagenes(st.session_state["modelo"], rutas, st.session_state["clases"])
            df_resultados = pd.DataFrame(resultados)
            st.write("### Resultados de la clasificación")
            st.dataframe(df_resultados)

            st.download_button(
                label="Descargar resultados como TSV",
                data=df_resultados.to_csv(sep="\t", index=False).encode("utf-8"),
                file_name="resultados_imagenes.tsv",
                mime="text/tab-separated-values"
            )

# --- Instrucciones en la barra lateral ---
st.sidebar.markdown("""
### Instrucciones:
1. Crea un archivo `.zip` con subcarpetas. Cada carpeta debe contener imágenes de una clase.
2. Sube el archivo en el **Paso 1** y haz clic en **Entrenar modelo**.
3. Luego, en el **Paso 2**, sube otro `.zip` con imágenes sueltas para clasificarlas.
4. Descarga los resultados como archivo `.tsv`.
""")
