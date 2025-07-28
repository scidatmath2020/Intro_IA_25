# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 18:22:12 2025

@author: Usuario
"""

# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import tempfile
import shutil
import gc
from PIL import Image
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# --- Reinicio seguro de sesión ---
if "reset_done" not in st.session_state:
    st.session_state.clear()
    gc.collect()
    K.clear_session()
    st.session_state["reset_done"] = True

# --- Botón manual para reiniciar ---
if st.sidebar.button("🔄 Reiniciar sesión"):
    st.session_state.clear()
    st.rerun()

# --- Funciones auxiliares ---

def descomprimir_zip(archivo_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

def crear_generadores(directorio, tamano=(224, 224), batch=32):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(directorio, target_size=tamano,
                                            batch_size=batch, class_mode='categorical',
                                            subset='training', shuffle=True)
    val_gen = datagen.flow_from_directory(directorio, target_size=tamano,
                                          batch_size=batch, class_mode='categorical',
                                          subset='validation', shuffle=False)
    return train_gen, val_gen

def crear_modelo(num_clases):
    K.clear_session()
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_clases, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def clasificar_imagenes(modelo, lista_rutas, diccionario_clases):
    if not hasattr(modelo, "predict"):
        st.error("❌ El modelo está dañado o no fue entrenado correctamente.")
        return []
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
                "clase_predicha": rev_map.get(idx, "desconocida"),
                "confianza": round(float(np.max(pred)), 4)
            })
        except Exception as e:
            st.warning(f"⚠️ No se pudo procesar {ruta}: {e}")
    return resultados

# --- Interfaz principal ---

st.set_page_config(page_title="Clasificador de Imágenes", layout="centered")
st.title("🧠 Clasificador de Imágenes Multiclase")
st.write("Entrena un modelo con imágenes organizadas en carpetas y luego clasifica nuevas imágenes.")

# Paso 1: Entrenamiento
st.header("Paso 1: Entrenamiento")
zip_ent = st.file_uploader("📦 Sube ZIP con carpetas por clase", type=["zip"])
if zip_ent:
    with st.spinner("🔍 Procesando ZIP..."):
        dir_ent = descomprimir_zip(zip_ent)
        train_gen, val_gen = crear_generadores(dir_ent)

        if train_gen.samples == 0:
            st.error("❌ No se encontraron imágenes válidas.")
        else:
            st.success(f"✅ {train_gen.samples} imágenes, {len(train_gen.class_indices)} clases.")
            epocas = st.slider("Número de épocas", 1, 20, 3)

            if st.button("🚀 Entrenar modelo"):
                with st.spinner("🧠 Entrenando..."):
                    # Eliminar versiones anteriores
                    for k in ["modelo", "clases"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    K.clear_session()

                    modelo = crear_modelo(len(train_gen.class_indices))
                    modelo.fit(train_gen, validation_data=val_gen, epochs=epocas)
                    val_loss, val_acc = modelo.evaluate(val_gen, verbose=0)

                    st.session_state["modelo"] = modelo
                    st.session_state["clases"] = train_gen.class_indices

                    st.success(f"📈 Precisión en validación: {round(val_acc * 100, 2)}%")

        shutil.rmtree(dir_ent)

# Paso 2: Clasificación
st.header("Paso 2: Clasificación")
if "modelo" not in st.session_state:
    st.info("ℹ️ Primero entrena un modelo en el Paso 1.")
else:
    zip_imgs = st.file_uploader("📷 Sube ZIP con imágenes a clasificar", type=["zip"])
    if zip_imgs:
        with st.spinner("🔍 Clasificando..."):
            dir_test = descomprimir_zip(zip_imgs)
            rutas = [
                os.path.join(dp, f)
                for dp, _, files in os.walk(dir_test)
                for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if len(rutas) == 0:
                st.warning("⚠️ No se encontraron imágenes válidas.")
            else:
                resultados = clasificar_imagenes(st.session_state["modelo"], rutas, st.session_state["clases"])
                df_res = pd.DataFrame(resultados)
                st.write("### 📋 Resultados (primeras 50 filas)")
                st.dataframe(df_res.head(50))
                conteo = Counter(df_res["clase_predicha"])
                st.write("Distribución:", dict(conteo))
                tsv = df_res.to_csv(sep="\t", index=False).encode("utf-8")
                st.download_button("📥 Descargar resultados (.tsv)", tsv, "resultados.tsv", "text/tab-separated-values")
            shutil.rmtree(dir_test)

# Instrucciones
st.sidebar.markdown("""
### 🧾 Instrucciones
1. ZIP con carpetas (una por clase) ➤ Paso 1 ➤ Entrenar.
2. ZIP con imágenes ➤ Paso 2 ➤ Clasificar.
3. Revisa y descarga el `.tsv`.
""")
