# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 14:00:00 2025

@author: SciData
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, zipfile, tempfile, shutil, io
from collections import Counter
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------- utilidades ----------
def descomprimir_zip(archivo_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

def crear_generadores(directorio, tamano=(224, 224), batch=32):
    datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        directorio, target_size=tamano, batch_size=batch,
        class_mode='categorical', subset='training', shuffle=True)
    val_gen = datagen.flow_from_directory(
        directorio, target_size=tamano, batch_size=batch,
        class_mode='categorical', subset='validation', shuffle=False)
    return train_gen, val_gen

@st.cache_resource
def crear_modelo(num_clases):
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
    for layer in base.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(num_clases, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def clasificar_imagenes(modelo, rutas, mapa_clases):
    rev = {v: k for k, v in mapa_clases.items()}
    res = []
    for ruta in rutas:
        try:
            img = Image.open(ruta).convert("RGB").resize((224, 224))
            arr = np.expand_dims(np.array(img)/255.0, 0)
            pred = modelo.predict(arr, verbose=0)[0]
            idx = int(np.argmax(pred))
            res.append({
                "imagen": os.path.basename(ruta),
                "clase_predicha": rev[idx],
                "confianza": round(float(pred[idx]), 4)
            })
        except Exception as e:
            st.warning(f"No pudo procesarse {ruta}: {e}")
    return res

# ---------- interfaz ----------
st.title("Clasificador de Imágenes Multiclase")
st.write("Entrena un modelo con tus carpetas de imágenes y clasifica nuevos archivos.")

# ---- Paso 1: entrenamiento ----
st.header("Paso 1 · Entrenamiento del modelo")
zip_entrenamiento = st.file_uploader("Sube un ZIP con carpetas (una por clase)", type=["zip"])

if zip_entrenamiento:
    dir_ent = descomprimir_zip(zip_entrenamiento)
    train_gen, val_gen = crear_generadores(dir_ent)
    st.success(f"✅ Dataset cargado: {train_gen.samples} imágenes — {len(train_gen.class_indices)} clases")

    if st.button("Entrenar modelo"):
        with st.spinner("Entrenando..."):
            modelo = crear_modelo(len(train_gen.class_indices))
            modelo.fit(train_gen, validation_data=val_gen, epochs=5)

            # --- efectividad en validación ---
            v_loss, v_acc = modelo.evaluate(val_gen, verbose=0)
            st.success(f"📈 Efectividad (accuracy de validación): **{v_acc:.2%}**")

            # guardamos en sesión
            st.session_state["modelo"] = modelo
            st.session_state["clases"] = train_gen.class_indices

        shutil.rmtree(dir_ent)

# ---- Paso 2: predicción ----
st.header("Paso 2 · Clasificación de nuevas imágenes")

if "modelo" not in st.session_state:
    st.info("Entrena primero un modelo en el Paso 1.")
else:
    zip_pred = st.file_uploader("Sube un ZIP con imágenes a clasificar", type=["zip"])
    if zip_pred:
        dir_pred = descomprimir_zip(zip_pred)
        rutas = [os.path.join(dp, f)
                 for dp, _, fs in os.walk(dir_pred)
                 for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        with st.spinner("Clasificando..."):
            resultados = clasificar_imagenes(
                st.session_state["modelo"], rutas, st.session_state["clases"])
            df_res = pd.DataFrame(resultados)

        st.write("### Resultados (primeras 50 filas)")
        st.dataframe(df_res.head(50))

        # distribución rápida
        st.write("Distribución por clase:", dict(Counter(df_res["clase_predicha"])))

        # --- botón de descarga TSV ---
        buffer = io.StringIO()
        df_res.to_csv(buffer, sep='\t', index=False)
        st.download_button("Descargar predicciones (.tsv)",
                           data=buffer.getvalue().encode('utf-8'),
                           file_name="predicciones.tsv",
                           mime="text/tab-separated-values")

        shutil.rmtree(dir_pred)

# ---- instrucciones ----
st.sidebar.markdown("""
### Instrucciones
1. **ZIP de entrenamiento** → Paso 1 → Entrenar  
2. Se muestra la **efectividad** del modelo  
3. **ZIP de imágenes** → Paso 2 → Clasificar  
4. Consulta los resultados y descarga el `.tsv`
""")
