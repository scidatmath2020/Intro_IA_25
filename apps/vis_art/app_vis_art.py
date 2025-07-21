# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:30:00 2025

@author: SciData
"""

import streamlit as st
import zipfile
import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# --- Configuración de Streamlit ---
st.set_page_config(page_title="Clasificador de Imágenes", layout="wide")
st.title("📸 Clasificación de Imágenes por Carpeta (Transfer Learning)")
st.write("Sube un ZIP con imágenes organizadas en carpetas (una por clase). Luego entrena y prueba tu modelo.")

# --- Instrucciones ---
st.sidebar.markdown("""
### Instrucciones:
1. Sube un archivo `.zip` con carpetas para **entrenamiento** (una carpeta por clase)
2. Haz clic en **Entrenar modelo** para iniciar el entrenamiento
3. Una vez entrenado, sube otro `.zip` con imágenes para **clasificación**
4. Revisa las predicciones y descarga el archivo `.tsv` con los resultados
""")

# --- Función para descomprimir ZIP ---
def descomprimir_zip(zip_file, destino):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(destino)

# --- Cargar imágenes para entrenamiento ---
st.header("Paso 1: Entrenamiento del modelo")
zip_train = st.file_uploader("🔼 Sube un archivo ZIP con carpetas de imágenes (una por clase)", type=["zip"])
modelo = None
clases = []

if zip_train:
    path_train = "train_data"
    if os.path.exists(path_train): shutil.rmtree(path_train)
    os.makedirs(path_train, exist_ok=True)
    descomprimir_zip(zip_train, path_train)

    total = sum(len(files) for _, _, files in os.walk(path_train))
    carpetas = next(os.walk(path_train))[1]
    st.success(f"✅ Dataset cargado: {total} imágenes en {len(carpetas)} clases.")

    if st.button("🚀 Entrenar modelo"):
        st.info("⌛ Entrenando modelo, espera unos segundos...")

        datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
        train_gen = datagen.flow_from_directory(
            path_train, target_size=(224, 224), batch_size=32,
            class_mode='categorical', subset='training'
        )
        val_gen = datagen.flow_from_directory(
            path_train, target_size=(224, 224), batch_size=32,
            class_mode='categorical', subset='validation'
        )

        clases = list(train_gen.class_indices.keys())

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        output = Dense(len(clases), activation='softmax')(x)
        modelo = Model(inputs=base_model.input, outputs=output)

        for layer in base_model.layers:
            layer.trainable = False

        modelo.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        modelo.fit(train_gen, validation_data=val_gen, epochs=5)

        val_loss, val_acc = modelo.evaluate(val_gen, verbose=0)
        st.success(f"📈 Efectividad del modelo: **{val_acc:.2%}**")

# --- Clasificar nuevas imágenes ---
st.header("Paso 2: Clasificación de nuevas imágenes")
zip_pred = st.file_uploader("🔼 Sube un ZIP con imágenes para predecir", type=["zip"], key="pred")

if zip_pred and modelo is not None and clases:
    path_pred = "predict_data"
    if os.path.exists(path_pred): shutil.rmtree(path_pred)
    os.makedirs(path_pred, exist_ok=True)
    descomprimir_zip(zip_pred, path_pred)

    imagenes = []
    for root, _, files in os.walk(path_pred):
        for name in files:
            if name.lower().endswith(("png", "jpg", "jpeg")):
                img_path = os.path.join(root, name)
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                img_array = image.img_to_array(img) / 255.0
                imagenes.append((name, img_array))

    if not imagenes:
        st.warning("⚠️ No se encontraron imágenes válidas en el ZIP.")
    else:
        nombres, arrays = zip(*imagenes)
        arrays = np.stack(arrays)
        predicciones = modelo.predict(arrays)
        etiquetas = [clases[np.argmax(p)] for p in predicciones]

        resultados = pd.DataFrame({"archivo": nombres, "clase_predicha": etiquetas})
        st.write("### Resultados de la clasificación:")
        st.dataframe(resultados)

        # Botón para descargar resultados
        tsv = resultados.to_csv(sep="\t", index=False)
        st.download_button(
            label="📥 Descargar resultados como TSV",
            data=tsv,
            file_name="predicciones.tsv",
            mime="text/tab-separated-values"
        )
