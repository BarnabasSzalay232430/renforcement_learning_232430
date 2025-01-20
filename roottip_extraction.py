import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.morphology import skeletonize
import logging
import json

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
PLATE_SIZE_MM = 150
PLATE_POSITION_ROBOT = [0.10775, 0.062, 0.17]  # Base position for robot
RECT_POSITIONS = [
    (0.09, 0.135, 0.18, 0.265),
    (0.27, 0.135, 0.36, 0.265),
    (0.45, 0.135, 0.54, 0.285),
    (0.61, 0.135, 0.75, 0.265),
    (0.80, 0.135, 0.95, 0.295),
]

# Model Losses and Metrics
@tf.keras.utils.register_keras_serializable()
def f1_metric(y_true, y_pred, threshold=0.3):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    TP = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    pred_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = TP / (pred_positives + tf.keras.backend.epsilon())
    recall = TP / (positives + tf.keras.backend.epsilon())
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    dice_loss = 1 - (2 * tf.reduce_sum(y_true * y_pred) + 1e-7) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7
    )
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * dice_loss + 0.5 * bce_loss

# Image Preprocessing and Utilities
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image

def extract_petri_dish(image):
    _, thresholded = cv2.threshold(image, 57, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logging.warning("No contours detected.")
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    max_side = max(w, h)
    x_center, y_center = x + w // 2, y + h // 2
    x_start = max(0, x_center - max_side // 2)
    y_start = max(0, y_center - max_side // 2)
    return image[y_start : y_start + max_side, x_start : x_start + max_side]

def predict_root_mask(image, model, patch_size=128, stride=64):
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(np.stack([patch] * 3, axis=-1) / 255.0)
    patches = np.array(patches)
    predictions = model.predict(patches, verbose=0)
    mask = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for idx, pred in enumerate(predictions):
        y = (idx // ((w - patch_size) // stride + 1)) * stride
        x = (idx % ((w - patch_size) // stride + 1)) * stride
        mask[y : y + patch_size, x : x + patch_size] += pred[..., 0]
        count[y : y + patch_size, x : x + patch_size] += 1
    return (mask / np.maximum(count, 1)) > 0.5

def connect_roots(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=15)

def filter_and_skeletonize_roots(mask, rect_positions):
    h, w = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    root_data = []

    for rect_idx, (x_start, y_start, x_end, y_end) in enumerate(rect_positions):
        rect_x_start = int(x_start * w)
        rect_y_start = int(y_start * h)
        rect_x_end = int(x_end * w)
        rect_y_end = int(y_end * h)

        valid_objects = []
        for i in range(1, num_labels):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] > 10:  # Ignore small noise
                highest_point = np.min(np.where(labels == i)[0])
                if rect_x_start <= highest_point <= rect_x_end:
                    valid_objects.append(i)

        if valid_objects:
            largest_object = max(valid_objects, key=lambda obj_id: stats[obj_id, cv2.CC_STAT_AREA])
            filtered_mask[labels == largest_object] = 255
            root_data.append((rect_idx + 1, largest_object))

    if len(root_data) > 5:
        root_data = root_data[:5]  # Limit to 5 roots

    skeletonized = skeletonize(filtered_mask > 0).astype(np.uint8) * 255
    return filtered_mask, skeletonized, root_data, labels

def find_root_tips(mask, root_data, labels, conversion_factor):
    root_tips = []
    for rect_id, obj_id in root_data:
        coords = np.column_stack(np.where(labels == obj_id))
        if len(coords) == 0:
            continue
        lowest_point = coords[np.argmax(coords[:, 0])]  # Max y-coordinate
        y_pixel, x_pixel = lowest_point
        x_mm = x_pixel * conversion_factor
        y_mm = y_pixel * conversion_factor
        x_robot = (x_mm / 1000) + PLATE_POSITION_ROBOT[0]
        y_robot = (y_mm / 1000) + PLATE_POSITION_ROBOT[1]
        z_robot = PLATE_POSITION_ROBOT[2]
        logging.info(f"Pixel: ({x_pixel}, {y_pixel}), MM: ({x_mm:.2f}, {y_mm:.2f}), Robot: ({x_robot:.5f}, {y_robot:.5f}, {z_robot:.5f})")
        root_tips.append([x_robot, y_robot, z_robot])
    return root_tips

def extract_roots(directory, model, output_file):
    results = {}
    for image_name in os.listdir(directory):
        if not image_name.endswith(".png"):
            continue

        try:
            image_path = os.path.join(directory, image_name)
            logging.info(f"Processing image: {image_name}")
            image = preprocess_image(image_path)
            petri_dish = extract_petri_dish(image)
            root_mask = predict_root_mask(petri_dish, model)
            connected_mask = connect_roots(root_mask)
            filtered_mask, skeletonized_mask, root_data, labels = filter_and_skeletonize_roots(connected_mask, RECT_POSITIONS)
            conversion_factor = PLATE_SIZE_MM / petri_dish.shape[1]
            root_tips = find_root_tips(skeletonized_mask, root_data, labels, conversion_factor)
            results[image_name] = root_tips
        except Exception as e:
            logging.error(f"Failed to process {image_name}: {e}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Root tips saved to: {output_file}")

# Main Execution
if __name__ == "__main__":
    directory = r"C:\Users\szala\Documents\GitHub\renforcement_learning_232430\textures\_plates"
    output_file = r"C:\Users\szala\Documents\GitHub\renforcement_learning_232430\roottip_coordinates.json"
    model_path = r"C:\Users\szala\Documents\GitHub\renforcement_learning_232430\232430_unet_model_128px_v9md_checkpoint.keras"
    model = tf.keras.models.load_model(model_path, custom_objects={"combined_loss": combined_loss, "f1_metric": f1_metric})
    extract_roots(directory, model, output_file)
