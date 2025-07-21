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
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- FUNCIONES AUXILIARES ---
def descomprimir_zip(archivo_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

def crear_generadores(directorio, tamano=(224, 224), batch=16):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.3,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    train_gen = datagen.flow_from_directory(
        directorio, target_size=tamano, batch_size=batch,
        class_mode='categorical', subset='training', shuffle=True)
    val_gen = datagen.flow_from_directory(
        directorio, target_size=tamano, batch_size=batch,
        class_mode='categorical', subset='validation', shuffle=False)
    return train_gen, val_gen

@st.cache_resource
def crear_modelo(num_clases):
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_clases, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def clasificar_imagenes(modelo, lista_rutas, diccionario_clases):
    resultados = []
    rev_map = {v: k for k, v in diccionario_clases.items()}
    for ruta in lista_rutas:
        try:
            img = Image.open(ruta).convert("RGB").resize((224, 224))
            arr = np.expand_dims(np.array(img) / 255.0, 0)
            pred = modelo.predict(arr, verbose=0)
            idx = int(np.argmax(pred))
            resultados.append({
                "archivo": os.path.basename(ruta),
                "clase_predicha": rev_map[idx],
                "confianza": round(float(np.max(pred)), 4)
            })
        except Exception as e:
            st.warning(f"⚠️ No se pudo procesar {ruta}: {e}")
    return resultados

# --- INTERFAZ STREAMLIT ---
st.title("Clasificador de Imágenes Multiclase")
st.write("Entrena un modelo con imágenes organizadas en carpetas y luego clasifica nuevas imágenes.")

# Paso 1: Entrenamiento
st.header("Paso 1: Entrenamiento")
zip_ent = st.file_uploader("Sube ZIP con carpetas por clase", type=["zip"])
if zip_ent:
    with st.spinner("Procesando dataset..."):
        dir_ent = descomprimir_zip(zip_ent)
        train_gen, val_gen = crear_generadores(dir_ent)
        st.success(f"✅ Dataset: {train_gen.samples} imágenes en {len(train_gen.class_indices)} clases.")
        if st.button("Entrenar modelo"):
            modelo = crear_modelo(len(train_gen.class_indices))
            history = modelo.fit(train_gen, validation_data=val_gen, epochs=3, verbose=1)
            acc = history.history["val_accuracy"][-1]
            st.session_state["modelo"] = modelo
            st.session_state["clases"] = train_gen.class_indices
            st.success(f"✅ Modelo entrenado. Efectividad en validación: {round(acc*100, 2)}%")
            shutil.rmtree(dir_ent)

# Paso 2: Clasificación
st.header("Paso 2: Clasificación")
if "modelo" not in st.session_state:
    st.info("Primero entrena un modelo en el Paso 1.")
else:
    zip_imgs = st.file_uploader("Sube ZIP con imágenes a clasificar", type=["zip"])
    if zip_imgs:
        with st.spinner("Clasificando..."):
            dir_test = descomprimir_zip(zip_imgs)
            rutas = [
                os.path.join(dp, f)
                for dp, _, files in os.walk(dir_test)
                for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            resultados = clasificar_imagenes(st.session_state["modelo"], rutas, st.session_state["clases"])
            df_res = pd.DataFrame(resultados)
            st.write("### Resultados (primeras 50 filas)")
            st.dataframe(df_res.head(50))
            conteo = Counter(df_res["clase_predicha"])
            st.write("Distribución:", dict(conteo))
            tsv = df_res.to_csv(sep="\t", index=False).encode("utf-8")
            st.download_button("Descargar resultados (.tsv)", tsv, "resultados.tsv", "text/tab-separated-values")
            shutil.rmtree(dir_test)

# Instrucciones
st.sidebar.markdown("""
### Instrucciones:
1. ZIP con carpetas (una por clase) ➤ Paso 1 ➤ Entrenar
2. ZIP con imágenes (.jpg/.png) ➤ Paso 2 ➤ Clasificar
3. Revisa resultados y descarga el `.tsv`
""")
