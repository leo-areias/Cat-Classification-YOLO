import cv2 as cv
import os
import json
import boto3
from ultralytics import YOLO

# Configuração do AWS S3
AWS_BUCKET_NAME = "bucket_name"  # Substitua pelo nome do seu bucket
s3_client = boto3.client("s3")

def upload_to_s3(file_path, s3_key):
    """Envia um arquivo para o AWS S3"""
    try:
        s3_client.upload_file(file_path, AWS_BUCKET_NAME, s3_key)
        print(f"Upload bem-sucedido: {s3_key}")
    except Exception as e:
        print(f"Erro ao enviar {s3_key}: {e}")

def preprocess(img):
    resized = cv.resize(img, (256,256))
    return resized

path = "./imgs"
output_dir = "./processed_imgs"
os.makedirs(output_dir, exist_ok=True)

img_list = os.listdir(path)

# Processamento e Upload das Imagens para S3
for img_name in img_list:
    input_path = os.path.join(path, img_name)
    output_path = os.path.join(output_dir, img_name)
    
    image = cv.imread(input_path)
    img_resized = preprocess(image)
    cv.imwrite(output_path, img_resized)

    # Upload da imagem processada para S3
    s3_key = f"imgs/{img_name}"  # Caminho no S3
    upload_to_s3(output_path, s3_key)

# Load YOLO model
model = YOLO("yolo11n.pt")  # Modelo YOLO treinado

# Inferência nas imagens processadas
results = model([os.path.join(output_dir, img) for img in img_list])

# Preparação do JSON
detection_results = []

for img_name, result in zip(img_list, results):
    img_output_path = os.path.join(output_dir, img_name)

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordenadas do bounding box
        confidence = float(box.conf[0])  # Confiabilidade
        class_id = int(box.cls[0])  # ID da classe

        detections.append({
            "class_id": class_id,
            "bounding_box": [x1, y1, x2, y2],
            "confidence": confidence
        })

    detection_results.append({
        "image_path": img_output_path,
        "detections": detections
    })

# Salvar JSON localmente
json_output_path = "./detection_results.json"
with open(json_output_path, "w") as json_file:
    json.dump(detection_results, json_file, indent=4)

# Upload do JSON para S3
json_s3_key = "inferences/detection_results.json"
upload_to_s3(json_output_path, json_s3_key)

print(f"Resultados salvos em {json_output_path} e enviados para S3.")
