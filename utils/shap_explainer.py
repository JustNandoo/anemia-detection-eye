import shap
import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def generate_shap(model, image_path, save_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img_input = preprocess_input(img.astype("float32"))
    img_array = np.expand_dims(img_input, axis=0)

    background = np.zeros((1, 224, 224, 3), dtype=np.float32)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(img_array)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_map = np.mean(np.abs(shap_values[0]), axis=-1)
    shap_map /= (np.max(shap_map) + 1e-8)

    shap_map = cv2.resize(shap_map, (224, 224))
    shap_map = np.uint8(255 * shap_map)
    shap_map = cv2.applyColorMap(shap_map, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        0.6,
        shap_map,
        0.4,
        0
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)

