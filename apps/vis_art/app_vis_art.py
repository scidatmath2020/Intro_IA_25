"""
Created on Mon Jul 21 13:30:00 2025
@author: SciData
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory

# -----------------------------------------------------------------------------#
# 1 ─ Configuración general de la página
# -----------------------------------------------------------------------------#
st.set_page_config(
    page_title="Clasificador de Imágenes (MobileNetV2)",
    page_icon="📷",
    layout="wide",
)
st.title("📷 Clasificador de Imágenes con MobileNetV2")
st.write(
    "Entrena un modelo a partir de un ZIP con **carpetas por clase** "
    "y luego clasifica nuevas imágenes. Todo en el navegador."
)

# -----------------------------------------------------------------------------#
# 2 ─ Funciones auxiliares
# -----------------------------------------------------------------------------#
@st.cache_resource(show_spinner=False)
def load_base_model() -> Model:
    """Descarga y cachea MobileNetV2 (sin la parte de clasificación)."""
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3)),
    )
    base.trainable = False  # congelamos pesos
    return base


def build_model(num_classes: int) -> Model:
    base = load_base_model()
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def extract_zip(uploaded_file, to_path: Path) -> None:
    """Extrae el Zip subido a un directorio temporal."""
    with ZipFile(uploaded_file) as zf:
        zf.extractall(to_path)


def get_image_paths(root: Path) -> list[Path]:
    """Obtiene las rutas de todas las imágenes recursivamente."""
    return [p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


# -----------------------------------------------------------------------------#
# 3 ─ Subir ZIP de ENTRENAMIENTO y entrenamiento del modelo
# -----------------------------------------------------------------------------#
st.header("🔨 Entrenamiento")

train_zip = st.file_uploader(
    "Sube un ZIP con **carpetas por clase** para entrenar",
    type=["zip"],
    key="train_zip",
)

epochs = st.number_input("Épocas", 1, 20, value=3, step=1)

if train_zip is not None:
    with st.spinner("Descomprimiendo y preparando imágenes …"):
        tmp_dir = Path(tempfile.mkdtemp())
        extract_zip(train_zip, tmp_dir)

        # ---------------------- Generadores de datos ------------------------- #
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            rotation_range=20,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
        )
        train_gen = datagen.flow_from_directory(
            tmp_dir,
            target_size=(224, 224),
            batch_size=16,
            subset="training",
            class_mode="categorical",
        )
        val_gen = datagen.flow_from_directory(
            tmp_dir,
            target_size=(224, 224),
            batch_size=16,
            subset="validation",
            class_mode="categorical",
            shuffle=False,
        )

    # ----------------------- Construir y entrenar --------------------------- #
    st.success(
        f"✅ Dataset cargado: **{train_gen.samples}** imágenes / "
        f"{len(train_gen.class_indices)} clases."
    )

    if st.button("🏋️‍♂️ Entrenar modelo"):
        model = build_model(num_classes=len(train_gen.class_indices))

        prog_bar = st.progress(0.0, text="Entrenando…")
        hist = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: prog_bar.progress(
                        (epoch + 1) / epochs,
                        text=(
                            f"Época {epoch+1}/{epochs} • "
                            f"val_acc={logs['val_accuracy']:.2%}"
                        ),
                    )
                )
            ],
        )
        prog_bar.empty()

        # ----------------------- Métrica final ------------------------------ #
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        st.metric("📈 Exactitud en validación", f"{val_acc:.2%}")

        # Guardamos el modelo en sesión (no en disco) para predecir luego
        st.session_state["model"] = model
        st.session_state["class_map"] = {
            v: k for k, v in train_gen.class_indices.items()
        }

        # Limpiamos directorio temporal
        shutil.rmtree(tmp_dir)


# -----------------------------------------------------------------------------#
# 4 ─ Subir ZIP para PREDICCIÓN
# -----------------------------------------------------------------------------#
st.header("🔎 Predicción")

if "model" not in st.session_state:
    st.info("Entrena el modelo primero para habilitar la predicción.")
    st.stop()

pred_zip = st.file_uploader(
    "Sube un ZIP con imágenes para clasificar (carpetas opcionales)",
    type=["zip"],
    key="pred_zip",
)

if pred_zip is not None:
    with st.spinner("Descomprimiendo imágenes…"):
        tmp_pred = Path(tempfile.mkdtemp())
        extract_zip(pred_zip, tmp_pred)
        img_paths = get_image_paths(tmp_pred)
        if not img_paths:
            st.error("❌ No se encontraron imágenes válidas en el ZIP.")
            st.stop()

    model: Model = st.session_state["model"]
    class_map: dict[int, str] = st.session_state["class_map"]

    resultados = []
    bar = st.progress(0.0, "Clasificando …")

    for i, img_path in enumerate(img_paths, start=1):
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img) / 255.0
        pred = model.predict(arr[np.newaxis, ...], verbose=0)[0]
        idx = int(pred.argmax())
        resultados.append(
            {
                "archivo": img_path.name,
                "clase_predicha": class_map[idx],
                "confianza": round(float(pred[idx]), 4),
            }
        )
        bar.progress(i / len(img_paths))
    bar.empty()

    df_result = pd.DataFrame(resultados).sort_values("archivo")
    st.dataframe(df_result, use_container_width=True)

    # ----------------------- Botón de descarga ------------------------------ #
    tsv = df_result.to_csv(sep="\t", index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar resultados (.tsv)",
        data=tsv,
        file_name="predicciones.tsv",
        mime="text/tab-separated-values",
    )

    # Limpieza
    shutil.rmtree(tmp_pred)


# -----------------------------------------------------------------------------#
# 5 ─ Instrucciones en la barra lateral
# -----------------------------------------------------------------------------#
st.sidebar.markdown(
    """
## Pasos de uso
1. Prepara un **ZIP** de entrenamiento  
   *Cada carpeta = 1 clase.*  
2. Sube el ZIP y pulsa **Entrenar modelo**  
3. Revisa la exactitud alcanzada  
4. Sube otro ZIP con imágenes (o carpetas)  
5. Descarga el `.tsv` con las predicciones
"""
)

st.sidebar.caption("App demostrativa — SciData 2025")
