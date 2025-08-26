import os
import uuid
import numpy as np
import cv2
import tempfile # 개선점: tempfile 라이브러리 사용
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# --- AI 모델 로드 ---
try:
    model = YOLO('best.pt')
    print("AI 모델(best.pt) 로딩 성공!")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    model = None

# --- 핵심 분석 로직 (정확도 개선) ---
def check_pyeongsang_cleanliness(image_path):
    if not model:
        return 'MODEL_NOT_LOADED'
    
    results = model(image_path, verbose=False)
    result = results[0]
    class_names = result.names
    pyeongsang_masks, trash_masks = [], []

    if result.masks is None:
        return 'NO_PYEONGSANG'

    # 평상과 쓰레기의 마스크 영역을 각각 수집
    for i, cls_id in enumerate(result.boxes.cls):
        class_name = class_names[int(cls_id)]
        mask_tensor = result.masks.data[i]
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
        if class_name == 'pyeongsang':
            pyeongsang_masks.append(mask_np)
        elif class_name == 'trash':
            trash_masks.append(mask_np)

    if not pyeongsang_masks: return 'NO_PYEONGSANG'
    if not trash_masks: return 'CLEAN'

    # 모든 평상 마스크를 하나로 합침
    combined_pyeongsang_mask = np.zeros_like(pyeongsang_masks[0])
    for mask in pyeongsang_masks:
        combined_pyeongsang_mask = np.maximum(combined_pyeongsang_mask, mask)
    
    # 모든 쓰레기 마스크를 하나로 합침
    combined_trash_mask = np.zeros_like(trash_masks[0])
    for mask in trash_masks:
        combined_trash_mask = np.maximum(combined_trash_mask, mask)
    
    # 평상과 쓰레기 마스크가 실제로 겹치는지 계산
    overlap = cv2.bitwise_and(combined_pyeongsang_mask, combined_trash_mask)
    
    # 겹치는 부분이 있으면 'DIRTY', 없으면 'CLEAN'
    return 'DIRTY' if np.sum(overlap) > 0 else 'CLEAN'

# --- API 엔드포인트 ---
@app.route('/')
def health_check():
    return "백엔드 서버가 정상적으로 동작 중입니다."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'AI 모델이 로딩되지 않았습니다. 잠시 후 다시 시도해주세요.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': '파일이 요청에 포함되지 않았습니다.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if file:
        # 개선점: 안전한 임시 파일을 생성하고 자동으로 삭제되도록 처리
        with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            try:
                status = check_pyeongsang_cleanliness(temp_file.name)
                return jsonify({'status': status})
            except Exception as e:
                print(f"분석 중 오류 발생: {e}")
                return jsonify({'error': '이미지 분석 중 오류가 발생했습니다.'}), 500

# --- 서버 실행 코드 제거 ---
# Render의 Start Command에서 gunicorn을 사용하므로 이 부분은 필요 없습니다.
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 10000))
#     app.run(host='0.0.0.0', port=port)
