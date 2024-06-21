from flask import Flask, request, render_template, jsonify, Response
import json
from utils import *
from model import *
from transformers import AutoImageProcessor, AutoModelForImageClassification

pattern_processor = AutoImageProcessor.from_pretrained("IrshadG/Clothes_Pattern_Classification_v2")
pattern_model = AutoModelForImageClassification.from_pretrained("IrshadG/Clothes_Pattern_Classification_v2")

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

model_path = "./classification_model.h5"
classification_model = load_model(model_path)

vector_feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
vector_model = ViTModel.from_pretrained('facebook/dino-vits16').to("cpu")

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="clothes")

# DataBase로 변경 예정
# coordinate_df = pd.read_csv('final_001001_001002_cordi.csv', dtype = str)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/choice', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    if file.filename == '':
        return 'No selected file'

    image_data = file.read()

    image_stream = BytesIO(image_data)

    image = Image.open(image_stream)

    picture = ImageOps.exif_transpose(image)
    upper_image, pants_image = show_image(image = picture, seg_model = seg_model, seg_processor = processor)
    
    if upper_image != None:
        upper_img_str = image_to_base64(upper_image)
    else:
        upper_img_str = "None"
    
    if pants_image != None:
        pants_img_str = image_to_base64(pants_image)
    else:
        pants_img_str = "None"

    return render_template('choice.html', processed_upper_image=upper_img_str , processed_pants_image=pants_img_str)

@app.route('/recommend_product', methods=['POST'])
def recommend_product():
    selected_image = request.form.get('selected_image')
    
    image_data = selected_image.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    pil_image = Image.open(BytesIO(image_bytes))
    clothes_labels = ['001001','001002','001003','001004','001005', '001011']

    # 이미지 예측
    predicted_class = predict_image(classification_model, pil_image, clothes_labels)
    color_info = extract_color_info(pil_image, classification_model)
    
    pattern_class = pattern_classification(pattern_model, pattern_processor, pil_image)
    
    print(predicted_class, color_info)
    print("pattern은", pattern_class)
    # 유사 상품들 list 추출
    index_list = process_image_and_query_collection(pil_image, vector_feature_extractor, vector_model, collection, predicted_class, color_info, pattern_class)
    
    # 유사 상품을 이용한 코디 정보들 가져오기
    coordi_info_list = list_to_coordinate(index_list, coordinate_df)

    return render_template('recommend_products.html', coordinate_lists = coordi_info_list)

if __name__ == '__main__':
    app.run(debug=True)
