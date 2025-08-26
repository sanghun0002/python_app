import os
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    model = YOLO('best.pt')
    print("AI 모델(best.pt) 로딩 성공!")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    model = None

# (가상 데이터베이스는 생략 - 반납 인증 기능에만 집중)

def check_pyeongsang_cleanliness(image_path):
    if not model:
        return 'MODEL_NOT_LOADED'
    
    print(f"이미지 분석 시작: {image_path}")
    results = model(image_path, verbose=False)
    result = results[0]
    class_names = result.names
    pyeongsang_masks, trash_masks = [], []

    if result.masks is not None:
        for i, cls_id in enumerate(result.boxes.cls):
            class_name = class_names[int(cls_id)]
            if class_name == 'pyeongsang': pyeongsang_masks.append(1)
            elif class_name == 'trash': trash_masks.append(1)
    
    print(f"탐지 결과: 평상 {len(pyeongsang_masks)}개, 쓰레기 {len(trash_masks)}개")

    if not pyeongsang_masks: return 'NO_PYEONGSANG'
    if not trash_masks: return 'CLEAN'
    # 실제로는 마스크 겹침 계산이 필요하지만, 단순 탐지만으로 로직 단순화
    return 'DIRTY'

@app.route('/')
def health_check():
    """서버가 살아있는지 확인하는 경로"""
    return "백엔드 서버가 정상적으로 동작 중입니다."

@app.route('/predict', methods=['POST'])
def predict():
    print("'/predict' 요청 수신")
    if 'file' not in request.files:
        print("요청에 파일이 없음")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        print("파일 이름이 없음")
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"이미지 저장 완료: {filepath}")
        
        status = check_pyeongsang_cleanliness(filepath)
        print(f"분석 완료, 상태: {status}")
        
        os.remove(filepath)
        return jsonify({'status': status})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)