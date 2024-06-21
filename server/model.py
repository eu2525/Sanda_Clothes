import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.preprocessing import image
from sklearn.cluster import KMeans

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("IrshadG/Clothes_Pattern_Classification_v2")
model = AutoModelForImageClassification.from_pretrained("IrshadG/Clothes_Pattern_Classification_v2")

# 이미지 전처리 함수 정의
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 이미지 정규화
    return img_array

# 예측 함수 정의
def predict_image(model, img, class_labels):
    processed_image = preprocess_image(img)
    prediction = model.predict(processed_image)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = class_labels[predicted_class_idx]
    return predicted_class

def create_feature_extractor(model):
    last_conv_layer = model.get_layer("conv5_block3_out")
    return keras.Model(model.inputs, last_conv_layer.output)

def create_classifier(model, input_shape):
    input_2 = keras.Input(shape=input_shape[1:])
    x_2 = model.get_layer("flatten_1")(input_2)
    x_2 = model.get_layer("dense_1")(x_2)
    return keras.Model(input_2, x_2)

def compute_gradients(feature_extractor, classifier, x):
    with tf.GradientTape() as tape:
        output_1 = feature_extractor(x)
        tape.watch(output_1)
        preds = classifier(output_1)
        class_id = tf.argmax(preds[0])
        output_2 = preds[:, class_id]

    grads = tape.gradient(output_2, output_1)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    return pooled_grads, output_1

def generate_heatmap(pooled_grads, output_1):
    output_1 = output_1.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        output_1[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(output_1, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_heatmap_on_image(heatmap, img):
    img = image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    
    jet = plt.get_cmap("jet")
    color = jet(np.arange(256))[:, :3]
    color_heatmap = color[heatmap]
    
    color_heatmap = keras.preprocessing.image.array_to_img(color_heatmap)
    color_heatmap = color_heatmap.resize((img.shape[1], img.shape[0]))
    color_heatmap = keras.preprocessing.image.img_to_array(color_heatmap)
    
    overlay_img = color_heatmap * 0.4 + img
    overlay_img = keras.preprocessing.image.array_to_img(overlay_img)
    
    return overlay_img

def extract_key_colors(heatmap, img):
    heatmap =  keras.preprocessing.image.array_to_img(heatmap[..., np.newaxis])
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))
    heatmap_arr = keras.preprocessing.image.img_to_array(heatmap)
    heatmap_arr = heatmap_arr.squeeze()
    
    bound = sorted(heatmap_arr.flatten(), reverse=True)[int(len(heatmap_arr.flatten()) * 0.05)]
    heatmap_key_point = np.where(heatmap_arr > bound)
    key_colors = []
    for pixel in zip(heatmap_key_point[0], heatmap_key_point[1]):
        key_colors.append(img[pixel[0], pixel[1]])

    return np.array(key_colors)

def extract_center_color(img):
    center = (112, 112)
    surrounding_coords = [(80, 80), (150, 150), (80,150) , (150, 80)]

    # 중앙 픽셀의 색상 추출
    center_color = img[center[0], center[1]]

    all_colors = [center_color]
    for x, y in surrounding_coords:
        color_info = img[x, y]
        all_colors.append(color_info)

    median_color = np.median(all_colors, axis=0)

    return median_color.astype(int)

def cluster_color(key_colors, k):
    unique_colors = np.unique(key_colors, axis = 0)
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(unique_colors)
    labels = kmeans_model.predict(unique_colors)
    clustered_colors = []
    
    for i in range(k):
        index = np.where(labels == i)
        clustered_colors.append(np.mean(key_colors[index], axis = 0))
    clustered_colors = np.array(clustered_colors)

    return clustered_colors

