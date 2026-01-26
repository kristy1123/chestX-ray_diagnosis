import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.segmentation import morphological_chan_vese
from skimage.util import img_as_float
from datetime import datetime

# =========================================================
# [1] 설정 및 데이터 로드
# =========================================================
st.set_page_config(page_title="Lung-AI 진단 솔루션", layout="wide", page_icon="🩻")

# 경로 설정 (실제 환경에 맞게 수정 필요)
MODEL_PATH = "./best_model.pth"
HISTORY_CSV = "diagnosis_history.csv" 
DATASET_CSV_PATH = r"D:/lung_xray/Data_Entry_processed_Final.csv" 
IMAGE_DIR = r"D:\lung_xray\images" 

TARGET_CLASSES = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis']

@st.cache_data
def load_patient_database():
    if os.path.exists(DATASET_CSV_PATH):
        try:
            df = pd.read_csv(DATASET_CSV_PATH)
            # 필요한 컬럼만 로드 및 정리
            required_cols = ['Patient ID', 'Patient Age', 'Patient Sex', 'Image Index', 'View Position']
            available_cols = [c for c in required_cols if c in df.columns]
            df = df[available_cols]
            df = df.rename(columns={'Patient Sex': 'Sex', 'Patient Age': 'Age'})
            # View Position이 없어도 오류나지 않도록 처리하되, UI에서는 보여주지 않음
            if 'View Position' not in df.columns: df['View Position'] = 'PA'
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame(columns=['Patient ID', 'Age', 'Sex', 'Image Index', 'View Position'])

# =========================================================
# [2] 이미지 처리 및 Active Contour (실시간 분석 로직)
# =========================================================

def process_active_contour(image_input):
    """
    [핵심 로직]
    입력된 이미지를 즉시 처리하여 원본 이미지와 병변 마스크를 생성합니다.
    """
    img = None
    # 1. 이미지 읽기
    if isinstance(image_input, str): # 경로일 경우
        if os.path.exists(image_input):
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    elif image_input is not None: # 업로드 파일일 경우
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        image_input.seek(0)

    if img is None:
        return None, None, None

    # 원본 크기 저장
    original_h, original_w = img.shape

    # 2. 전처리 (CLAHE + Resize) -> AI 분석용
    img_resized_ai = cv2.resize(img, (512, 512))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_resized_ai)
    img_float = img_as_float(img_enhanced)

    # 3. 마스크 생성 (Active Contour Algorithm)
    # 초기화 (Convex Hull)
    blur = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127: binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(binary)
    for cnt in contours:
        if cv2.contourArea(cnt) > (512*512*0.05):
            hull = cv2.convexHull(cnt)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)

    # Chan-Vese Active Contour 적용
    cv_mask = morphological_chan_vese(img_float, num_iter=15, init_level_set=hull_mask, smoothing=2, lambda1=1, lambda2=1)
    
    # 4. 결과 마스크를 원본 크기로 복원
    final_mask_small = (cv_mask.astype(np.uint8) * 255)
    final_mask_original_size = cv2.resize(final_mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # 5. 시각화용 원본 이미지 (RGB 변환)
    img_rgb_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 6. BBox 계산
    contours, _ = cv2.findContours(final_mask_original_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        bbox = cv2.boundingRect(c)

    return img_rgb_raw, final_mask_original_size, bbox

@st.cache_resource
def load_ai_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(TARGET_CLASSES))
    device = torch.device("cpu")
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except: pass
    model.to(device)
    model.eval()
    return model, device

def get_prediction(model, device, img_rgb):
    """EfficientNet 모델 추론"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    return probs

# =========================================================
# [3] 저장 및 팝업 기능
# =========================================================

def get_next_filename(patient_id):
    if not os.path.exists(HISTORY_CSV):
        return f"{patient_id}_001.png"
    try:
        df = pd.read_csv(HISTORY_CSV)
        patient_records = df[df['Patient ID'].astype(str) == str(patient_id)]
        return f"{patient_id}_{len(patient_records) + 1:03d}.png"
    except:
        return f"{patient_id}_{int(datetime.now().timestamp())}.png"

def save_result_to_csv(patient_id, age, sex, diagnosis, bbox, saved_filename):
    new_data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Patient ID": [patient_id], "Age": [age], "Sex": [sex],
        "Diagnosis": [diagnosis], "Saved Image Name": [saved_filename],
        "BBox_X": [bbox[0] if bbox else 0], "BBox_Y": [bbox[1] if bbox else 0],
        "BBox_W": [bbox[2] if bbox else 0], "BBox_H": [bbox[3] if bbox else 0]
    }
    df_new = pd.DataFrame(new_data)
    if os.path.exists(HISTORY_CSV):
        df_new.to_csv(HISTORY_CSV, mode='a', header=False, index=False)
    else:
        df_new.to_csv(HISTORY_CSV, mode='w', header=True, index=False)
    return new_data

@st.dialog("✅ 데이터 저장 완료")
def show_success_modal(info_dict):
    st.write("진단 결과가 성공적으로 저장되었습니다.")
    
    # [수정 3] 팝업에 저장 일시(Timestamp) 추가
    summary = pd.DataFrame({
        "항목": ["저장 일시", "환자 ID", "진단명", "파일명"],
        "내용": [
            info_dict['Timestamp'][0], 
            info_dict['Patient ID'][0], 
            info_dict['Diagnosis'][0], 
            info_dict['Saved Image Name'][0]
        ]
    })
    st.table(summary.set_index("항목"))
    if st.button("닫기"): st.rerun()

# =========================================================
# [4] 메인 UI 로직
# =========================================================

def main():
    df_patients = load_patient_database()
    model, device = load_ai_model()

    if 'diagnosis_result' not in st.session_state:
        st.session_state['diagnosis_result'] = None

    # --- Sidebar ---
    with st.sidebar:
        st.title("🩻 Lung-AI System")
        st.header("1. 환자 및 이미지 선택")
        
        # 환자 선택
        patient_ids = df_patients['Patient ID'].unique() if not df_patients.empty else []
        selected_id = st.selectbox("환자 ID", patient_ids)
        
        # [수정 2] 환자 정보 박스형태로 출력 (View Position 제외)
        if not df_patients.empty:
            p_record = df_patients[df_patients['Patient ID'] == selected_id].iloc[0]
            st.markdown("#### 환자 기본 정보")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric(label="나이", value=f"{p_record['Age']}세")
            with col_p2:
                st.metric(label="성별", value=f"{p_record['Sex']}")
            # View Position은 출력하지 않음
        else:
            p_record = {}

        st.markdown("---")

        # 이미지 선택 탭
        tab1, tab2 = st.tabs(["📂 업로드", "🗄️ 기존 기록"])
        input_source = None
        with tab1:
            f = st.file_uploader("X-ray 파일 업로드", type=['png','jpg','jpeg'])
            if f: input_source = f
        with tab2:
            if not df_patients.empty:
                imgs = df_patients[df_patients['Patient ID'] == selected_id]['Image Index'].tolist()
                s_img = st.selectbox("DB 이미지 선택", imgs)
                p = os.path.join(IMAGE_DIR, s_img)
                if os.path.exists(p): input_source = p

        st.markdown("---")
        run_btn = st.button("🚀 진단 시작", type="primary", use_container_width=True)

    # --- Main Content ---
    st.title("📋 폐 질병 정밀 진단")

    # [분석 실행]
    if run_btn and input_source:
        with st.spinner("이미지 전처리 및 Active Contour 분석 중..."):
            # 1. 실시간 이미지 처리 및 마스크 생성
            img_rgb, final_mask, bbox = process_active_contour(input_source)
            
            # 2. AI 모델 추론
            probs = get_prediction(model, device, img_rgb)
            top_idx = np.argmax(probs)
            
            # 3. 결과 저장 (Session State)
            st.session_state['diagnosis_result'] = {
                'img_rgb': img_rgb,           
                'final_mask': final_mask,     
                'bbox': bbox,
                'probs': probs * 100,
                'diagnosis': TARGET_CLASSES[top_idx],
                'p_record': p_record,
                'new_filename': get_next_filename(selected_id)
            }

    # [결과 화면 표시]
    if st.session_state['diagnosis_result']:
        res = st.session_state['diagnosis_result']
        
        col_left, col_right = st.columns(2)
        
        # 1. 왼쪽: 원본 이미지 출력
        with col_left:
            st.subheader("📷 Input Image (Original)")
            st.image(res['img_rgb'], caption="Raw X-ray Input", use_container_width=True)

        # 2. 오른쪽: 최종 결과 이미지 생성 (실시간 합성)
        with col_right:
            st.subheader("🧠 Analysis Result")
            
            # (A) 베이스 이미지 (원본)
            base_img = res['img_rgb'].copy()
            
            # (B) 마스크 오버레이 생성 (파란색 채우기)
            overlay = base_img.copy()
            overlay[res['final_mask'] > 0] = [0, 0, 255] # Mask 영역을 파란색(Blue)
            
            # (C) 이미지 합성 (투명도 적용: 원본 0.7 + 파란마스크 0.3)
            vis_img = cv2.addWeighted(overlay, 0.3, base_img, 0.7, 0)
            
            # [수정 1] 마스크 경계선(Contour) 빨간색으로 그리기
            # 마스크로부터 컨투어 다시 추출
            contours_vis, _ = cv2.findContours(res['final_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 빨간색(RGB: 255, 0, 0), 두께 2
            cv2.drawContours(vis_img, contours_vis, -1, (255, 0, 0), 2)
            
            # (D) BBox 및 텍스트 추가
            if res['bbox']:
                x, y, w, h = res['bbox']
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 255), 3) # 노란색(Cyan) 박스
                label = f"{res['diagnosis']} ({res['probs'][np.argmax(res['probs'])]:.1f}%)"
                cv2.putText(vis_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            st.image(vis_img, caption="Process: Mask Overlay(Blue) + Contour(Red)", use_container_width=True)

        st.divider()

        # 하단: 상세 차트 및 저장 버튼
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Disease Probability")
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.barh(TARGET_CLASSES, res['probs'], color=['#2ecc71', '#e74c3c', '#f1c40f', '#8e44ad'])
            ax.set_xlim(0, 100)
            st.pyplot(fig)
            
        with c2:
            st.markdown(f"#### Diagnosis: **{res['diagnosis']}**")
            st.info("AI 분석 결과입니다. 확진을 위해서는 전문의의 판독이 필요합니다.")
            
            if st.button("💾 결과 저장 (DB Upload)", type="primary", use_container_width=True):
                info = save_result_to_csv(
                    res['p_record']['Patient ID'], res['p_record']['Age'], res['p_record']['Sex'],
                    res['diagnosis'], res['bbox'], res['new_filename']
                )
                show_success_modal(info)

if __name__ == "__main__":
    main()