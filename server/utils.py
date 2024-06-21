from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageDraw, ImageOps
import requests
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from model import *
from keras.models import load_model
from transformers import ViTFeatureExtractor, ViTModel
import chromadb
import pandas as pd
from io import BytesIO
import base64
import ast

#한글 깨짐 문제 방지를 위한 설정
plt.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

class_labels = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf"
}

def visualize_segmentation(pred_seg, class_labels, image):
    class_images = {}
    num_classes = len(class_labels)

    height, width = pred_seg.shape
    # 세그멘테이션 맵의 각 픽셀을 순회하면서 클래스별 색상을 적용.
    for class_index in range(num_classes):
        if class_index not in [1, 4, 5, 6, 7]:
            continue
        mask = (pred_seg == class_index)

        if np.any(mask):
            # 세그멘테이션 영역의 바운딩 박스를 계산
            ys, xs = np.where(mask)
            min_y, min_x = np.min(ys), np.min(xs)
            max_y, max_x = np.max(ys), np.max(xs)
            min_y = max(0, min_y - 20)
            min_x = max(0, min_x - 20)
            max_y = min(height, max_y + 20)
            max_x = min(width, max_x + 20)
            # 원본 이미지에서 세그멘테이션 영역 추출.
            segment_image = image.crop((min_x, min_y, max_x, max_y))

            # 잘린 이미지를 딕셔너리에 저장합니다.
            class_images[class_labels[class_index]] = segment_image

    return class_images

def make_data(image, model, class_labels, processor):
  try:
    inputs = processor(images=image, return_tensors="pt")

    #CPU로 진행
    model.to('cpu')

    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    #예측한 Class 확률 중 제일 높은 Class 추출
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    #시각화 및 물품 추출
    visualized_images = visualize_segmentation(pred_seg.numpy(), class_labels, image)
    
    return visualized_images
  
  except:
      return 0


def show_image(image, seg_model, seg_processor):
    class_images = make_data(image = image, model = seg_model, class_labels = class_labels , processor = seg_processor)
     
    if class_images == 0:
        upper_clothes = None
        bottom_clothes = None
    elif any(key in class_images for key in ["Upper-clothes", "Dress", "Skirt", "Pants"]):
        if 'Upper-clothes' in class_images:
            upper_clothes = class_images["Upper-clothes"]
        elif 'Dress' in class_images:
            upper_clothes = class_images["Dress"]
        else:
            upper_clothes = None

        if 'Pants' in class_images:
            bottom_clothes = class_images["Pants"]
        elif 'Skirt' in class_images:
            bottom_clothes = class_images["Skirt"]
        else:
            bottom_clothes = None
    else:
        upper_clothes = None
        bottom_clothes = None

    return upper_clothes, bottom_clothes

def image_to_base64(image):
    # 이미지를 base64 문자열로 변환
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def find_closest_color(clu, mid_color):
    # Calculate the Euclidean distance between mid_color and each color in clu
    distances = np.linalg.norm(clu - mid_color, axis=1)
    
    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    
    # Return the color with the minimum distance
    return clu[closest_index]

def hex_to_rgb(hex_code):
    """Convert hex code to RGB values."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def color_distance(c1, c2):
    """Calculate the Euclidean distance between two RGB colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def closest_color_category(rgb):
    """Determine the closest color category for a given RGB value."""
    color_ranges = {
        "블랙": [(0, 0, 0), (40, 40, 40)],
        "화이트": [(200, 200, 200), (255, 255, 255)],
        "블루": [(0, 0, 100), (80, 110, 255)],
        "중청": [(0, 0, 100), (100, 130, 180)],
        "골드": [(218, 165, 32), (255, 215, 0)],
        "연청": [(130, 140, 170), (176, 224, 230)],
        "카키 베이지": [(180, 170, 100), (200, 200, 200)],
        "카키": [(189, 183, 107), (240, 230, 140)],
        "데님": [(30, 144, 255), (70, 130, 180)],
        "카멜": [(193, 154, 107), (210, 180, 140)],
        "페일 핑크": [(250, 218, 221), (255, 192, 203)],
        "올리브 그린": [(85, 107, 47), (107, 142, 35)],
        "라이트 그린": [(144, 238, 144), (152, 251, 152)],
        "브라운": [(139, 69, 19), (165, 42, 42)],
        "퍼플": [(128, 0, 128), (147, 112, 219)],
        "오렌지": [(255, 165, 0), (255, 140, 0)],
        "레드": [(150, 0, 0), (255, 80, 0)],
        "핑크": [(255, 192, 203), (255, 105, 180)],
        "옐로우": [(200, 150, 0), (255, 255, 100)],
        #"베이지": [(200, 190, 170), (200, 200, 200)],
        "그린": [(0, 128, 0), (0, 255, 0)],
        "그레이": [(128, 128, 128), (169, 169, 169)],
        "아이보리": [(255, 255, 240), (250, 240, 230)],
        "스카이 블루": [(100, 170, 220), (170, 220, 255)],
        "다크 그레이": [(169, 169, 169), (105, 105, 105)],
        "네이비": [(0, 0, 128), (0, 0, 139)]
    }

    for color_name, (lower, upper) in color_ranges.items():
        if all(lower[i] <= rgb[i] <= upper[i] for i in range(3)):  # RGB 값이 범위 내에 있는지 확인
            return color_name
        
    # 범위에 속하지 않으면 가장 가까운 색상 찾기
    min_distance = float('inf')
    closest_color = None
    for color_name, (lower, upper) in color_ranges.items():
        avg_color = tuple((l + u) // 2 for l, u in zip(lower, upper))
        distance = sum((a - b) ** 2 for a, b in zip(rgb, avg_color))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color

def get_color_group(color_name):
    color_groups = {
        "블랙": "검정색 계열",
        "화이트": "흰색 계열",
        "중청": "파란색 계열",
        "골드": "노란색 및 금색 계열",
        "연청": "파란색 계열",
        "카키 베이지" : "갈색 계열",
        "카키": "녹색 계열",
        "데님": "파란색 계열",
        "카멜": "주황색 계열",
        "페일 핑크": "붉은색 및 분홍색 계열",
        "올리브 그린": "녹색 계열",
        "라이트 그린": "녹색 계열",
        "브라운": "갈색 계열",
        "퍼플": "보라색 계열",
        "오렌지": "주황색 계열",
        "레드": "붉은색 계열",
        "핑크": "붉은색 및 분홍색 계열",
        "옐로우": "노란색 및 금색 계열",
        "베이지": "흰색 및 베이지 계열",
        "그린": "녹색 계열",
        "그레이": "검정색 및 회색 계열",
        "아이보리": "흰색 및 베이지 계열",
        "스카이 블루": "파란색 계열",
        "블루": "파란색",
        "다크 그레이": "검정색 및 회색 계열",
        "네이비": "파란색 계열",
    }

    return color_groups.get(color_name, "Unknown color group")

def extract_color_info(img, classification_model):
    up_img = img.resize((224, 224))
    x = image.img_to_array(up_img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0 

    feature_extractor = create_feature_extractor(classification_model)
    classifier = create_classifier(classification_model, feature_extractor.output.shape)

    pooled_grads, output_1 = compute_gradients(feature_extractor, classifier, x)
    heatmap = generate_heatmap(pooled_grads, output_1)

    up_img = image.img_to_array(up_img)
    key_colors = extract_key_colors(heatmap, up_img)

    # Clustering 하고 난 뒤 그 중 사진의 정중앙 색깔이랑 제일 비슷한 색을 고르는게 맞을듯!!
    clu = cluster_color(key_colors, 200)
    mid_color = extract_center_color(up_img)
    
    closest_color = find_closest_color(clu, mid_color)

    print(closest_color)

    color_name = get_color_group(closest_color_category(closest_color))

    return color_name


def filter_results(query_results, desired_product_category, desired_color, desired_pattern):
    metadata = []
    metadatas = query_results['metadatas'][0]
    for result in metadatas:
        product_category = result['category']
        color = result["color"]
        pattern = result["pattern"]
        if product_category == desired_product_category and color == desired_color and pattern == desired_pattern :
            metadata.append(result)

    return metadata

def process_image_and_query_collection(upper_image, vector_feature_extractor, vector_model, collection, predicted_class, color_info, pattern_class):
    test_img = upper_image.resize((224, 224))

    test_img_tensor = vector_feature_extractor(images=test_img, return_tensors="pt").to("cpu")
    test_outputs = vector_model(**test_img_tensor)
    
    test_embedding = test_outputs.pooler_output.detach().cpu().numpy().squeeze()

    query_result = collection.query(
        query_embeddings=[test_embedding.tolist()],
        n_results=5000,
    )

    matching_metadata = filter_results(query_result, predicted_class, color_info, pattern_class)[:40]
    
    index_list = []
    for i, metadata in enumerate(matching_metadata):
        index_list.append(metadata['index'])

    return index_list

def list_to_coordinate(index_list, coordinate_data):
    # index_list에 해당하는 데이터만 추출하여 coordinate_data 필터링
    sorted_coordinate_data = coordinate_data.loc[coordinate_data['Product_index'].isin(index_list)].dropna(axis = 0)

    sorted_coordinate_data['coordi_products_info'] = sorted_coordinate_data['coordi_products_info'].apply(ast.literal_eval)

    sorted_coordinate_data = sorted_coordinate_data[sorted_coordinate_data.apply(lambda row: any(row['Product URL'] in info for info in row['coordi_products_info']), axis=1)]
    
    sorted_coordinate_data = sorted_coordinate_data[sorted_coordinate_data['coordi_products_info'].apply(len) < 10]
    
    coordi_list = list(sorted_coordinate_data.apply(lambda row: [row['coordi_image'], row['coordi_products_info']], axis=1))

    return coordi_list

def map_class_label(index):
    
    print(index)

    mapping = {
        0: "Argyle",
        1: "Check",
        10: "Sequin",
        11: "Solid",
        12: "Stripe",
        13: "Solid",
        2 : "Colour blocking",
        3: "Denim",
        4: "Dot",
        5: "Patterns",
        6: "Lace",
        7: "Metallic",
        8: "Patterns",
        9: "Placement print"
    }

    if index in mapping:
        return mapping[index]
    else:
        return 'other'

def pattern_classification(pattern_model, pattern_processor, image):
    inputs = pattern_processor(images=image, return_tensors="pt")

    outputs = pattern_model(**inputs)
    
    logits = outputs.logits.cpu()

    predicted_label = map_class_label(logits.argmax(-1).item())

    return predicted_label
