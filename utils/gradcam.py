""" import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam(model, image_path, save_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img_norm = img / 255.0
    img_array = np.expand_dims(img_norm, axis=0)

    # Find last convolution layer automatically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, superimposed) """

import tensorflow as tf
import numpy as np
import cv2
import os


def generate_gradcam_pp(model, image_path, save_path):
    # -----------------------------
    # Load & preprocess image
    # -----------------------------
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img_norm = img / 255.0
    img_array = np.expand_dims(img_norm, axis=0).astype(np.float32)

    # -----------------------------
    # Find last Conv2D layer
    # -----------------------------
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        raise ValueError("No Conv2D layer found in the model")

    # -----------------------------
    # Build gradient model
    # -----------------------------
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # -----------------------------
    # Gradient computation (Grad-CAM++)
    # -----------------------------
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            (img_array)
        )

        if isinstance(predictions, list):
            predictions = predictions[0]

        # Binary classification → take predicted score
        loss = predictions[0]

    # First, second & third order gradients
    grads = tape.gradient(loss, conv_outputs)

    first = grads
    second = grads ** 2
    third = grads ** 3

    # Global sum over spatial dimensions
    global_sum = tf.reduce_sum(conv_outputs, axis=(1, 2))

    # Alpha computation (Grad-CAM++)
    alpha_num = second
    alpha_denom = 2.0 * second + third * global_sum[:, None, None, :]

    alpha_denom = tf.where(
        alpha_denom != 0.0,
        alpha_denom,
        tf.ones_like(alpha_denom)
    )

    alphas = alpha_num / alpha_denom

    # Positive gradients only
    weights = tf.reduce_sum(
        tf.nn.relu(first) * alphas,
        axis=(1, 2)
    )

    # -----------------------------
    # Generate heatmap
    # -----------------------------
    conv_outputs = conv_outputs[0]
    weights = weights[0]

    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)

    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # -----------------------------
    # Overlay heatmap on image
    # -----------------------------
    superimposed = cv2.addWeighted(
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        0.6,
        heatmap,
        0.4,
        0
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, superimposed)

