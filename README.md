# 🫁 chestX-reay_diagnosis: 흉부 X-ray 다중 질환 진단 및 폐 영역 분할 시스템
- **프로젝트 개요**
  Lung-AI는 흉부 X-ray(CXR) 이미지를 분석하여 병변 및 폐 영역(ROI)을 정밀하게 분할(Segmentation)하고 주요 폐 질환을 진단하는 AI 기반 의료 보조 솔루션입니다. 
  <오라클 아카데미 2기 미니 프로젝트 주제 : 흉부 X-ray 내 폐 영역 분할>으로 진행한 사이드 프로젝트 

- **데이터 소개**
  <NIH Chest X-ray Dataset>을 기반으로 진행.
  데이터셋에는 No Finding을 포함한 총 14가지의 질환이 존재

- **개발환경 소개**
  RAM 32GB, GPU 4GB, 1.5TB HHD
  -> 고사양 CPU, 저사양 GPU
  
- **개발 기간**
  25.12.29 ~ 26.1.17 (약 3주)
  
  
## 🌟 주요 기능 (Key Features)

1. **다중 질환 진단 (Multi-class Classification)**
   * 4가지 클래스 분류: `정상(No Finding)`, `침윤(Infiltration)`, `흉수(Effusion)`, `무기폐(Atelectasis)`
     * <Efficient Net-B0 모델 학습 결과>
       * 
       * <img width="1000" height="800" alt="image" src="https://github.com/user-attachments/assets/67646674-06b8-472f-9aa9-7f914c01031c" />
       * <img width="1200" height="1000" alt="image" src="https://github.com/user-attachments/assets/4dc61380-a5ad-4d54-8e38-6ff0a4f57dc1" />

2. **폐 및 병변 영역 정밀 분할 (ROI Segmentation)**
   * Bounding Box 및 정밀 픽셀 마스크(Binary Mask) 생성
   * 원본 X-ray 이미지 위에 시각화 오버레이 제공
   * TVAC 모델
3. **진단 이력 관리 시스템 (Database Management)**
   * 환자 정보, 원본 영상, 진단 결과 및 마스크 데이터를 RDBMS 기반으로 체계적 관리

---

## 🏗️ 기술 스택 및 알고리즘 (Tech Stack & Algorithms)

### 1. Classification (질환 진단)
* **Model:** `EfficientNet-B0` + `DenseNet121` Feature Fusion Ensemble
* **Description:** EfficientNet의 전역적 특징(Global Feature) 추출 능력과 DenseNet의 미세 질감(Texture) 보존 능력을 앙상블하여 높은 민감도와 정확도를 달성했습니다.
* **Metrics:** Macro AUC, ROC Curve (데이터 불균형 문제를 고려하여 Macro AUC를 주요 지표로 채택)

### 2. Segmentation (영역 분할)
* **Deep Learning 기반:** `ResUNet-34`
  * ImageNet Pre-trained 가중치를 활용한 전이 학습(Transfer Learning) 적용.
* **수학적 알고리즘 기반:** `TVAC (Total Variation Active Contour)` & `ISODATA`
  * 데이터가 적거나 노이즈가 심한 환경에서 에너지 함수 최적화를 통해 매끄러운(Smooth) 폐 경계선을 추출합니다.
* **Metrics:** Dice Score (정답 마스크와 예측 마스크의 영역 겹침 지수 평가)

### 3. Database & Backend
* **RDBMS:** MySQL / PostgreSQL (CSV 파일 기반에서 RDBMS로 마이그레이션 적용)
* **Schema Design:** `Patients`(환자) - `Images`(영상) - `Diagnosis_History`(진단 기록) 간의 관계형 데이터베이스 구축.

---

## 📊 데이터베이스 스키마 (ERD)

의료 데이터의 무결성과 효율적인 이력 관리를 위해 설계된 데이터베이스 구조입니다.

```mermaid
erDiagram
    Patients ||--o{ Images : uploads
    Images ||--o{ Diagnosis_History : analyzed_as
    Patients ||--o{ Diagnosis_History : has_records

    Patients {
        VARCHAR(20) patient_id PK
        INT age
        CHAR(1) sex
        DATETIME created_at
    }
    Images {
        INT image_id PK
        VARCHAR(20) patient_id FK
        VARCHAR(255) file_path
        VARCHAR(10) view_position
        DATETIME uploaded_at
    }
    Diagnosis_History {
        INT diagnosis_id PK
        INT image_id FK
        VARCHAR(50) diagnosis_result
        FLOAT confidence_score
        VARCHAR(255) mask_file_path
        INT bbox_x
        INT bbox_y
        DATETIME diagnosis_date
    }
