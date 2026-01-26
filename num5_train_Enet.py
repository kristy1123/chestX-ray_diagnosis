import os
import multiprocessing

# [중요] CPU 사용률 100%를 위한 환경 변수 설정 (PyTorch 임포트 전에 설정 권장)
# WSL에서 물리 코어를 모두 사용하도록 강제합니다.
num_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["TORCH_NUM_THREADS"] = str(num_cores)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 시각화 관련
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics  # [수정] 충돌 방지를 위해 모듈 전체 임포트



# PyTorch 내부 스레드 설정
torch.set_num_threads(num_cores)


# =========================================================
# [1] 데이터 전처리 및 분할 함수 (로직 수정됨)
# =========================================================
def prepare_datasets(csv_path, target_classes):
    print(f"📂 데이터셋 로드 중: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. 전체 데이터에서 타겟 질병과 관련된 데이터만 일단 추립니다.
    # (No Finding 포함, 타겟 질병 중 하나라도 포함된 모든 행)
    condition = df['Finding Labels'].str.contains(target_classes[0])
    for label in target_classes[1:]:
        condition |= df['Finding Labels'].str.contains(label)
    
    filtered_df = df[condition].copy()
    
    # 2. [핵심 변경] "이미지 ID" 기준으로 Train/Test를 먼저 나눕니다 (8:2).
    # 이렇게 해야 희귀 질환도 비율대로 테스트셋에 들어갑니다.
    all_indices = filtered_df['Image Index'].unique()
    train_ids, test_ids = train_test_split(all_indices, test_size=0.2, random_state=42)
    
    # ID 기반으로 데이터프레임 분리
    raw_train_df = filtered_df[filtered_df['Image Index'].isin(train_ids)]
    test_df = filtered_df[filtered_df['Image Index'].isin(test_ids)].reset_index(drop=True)
    
    print(f"📌 전체 데이터 분할 완료: Train 후보 {len(raw_train_df)}장 / Test 확정 {len(test_df)}장")

    # ---------------------------------------------------------
    # 3. Train 데이터셋 내부 밸런싱 (2:1:1:1) 수행
    # ---------------------------------------------------------
    dfs = {}
    for label in target_classes:
        dfs[label] = raw_train_df[raw_train_df['Finding Labels'].str.contains(label)]
    
    # No Finding을 제외한 질병들의 데이터 수 확인
    disease_counts = [len(dfs[c]) for c in target_classes if c != 'No Finding']
    
    if not disease_counts:
        raise ValueError("학습 데이터셋 후보군에 질병 데이터가 없습니다.")
        
    # 가장 적은 질병 데이터 수 기준으로 맞춤
    min_count = min(disease_counts)
    
    # 데이터가 너무 적을 경우 안전장치 (최소 1장은 보장)
    if min_count == 0:
        print("⚠️ 경고: 학습용 데이터 분할 후 특정 질병 데이터가 0개입니다. 재분할이 필요할 수 있습니다.")
        min_count = 1 

    n_disease = min_count
    n_no_finding = min_count * 2 # 정상 데이터는 2배수로
    
    print(f"📊 학습 데이터 밸런싱 기준: 질병 {n_disease}장 / 정상 {n_no_finding}장")
    
    train_fragments = []
    for label in target_classes:
        n_sample = n_no_finding if label == 'No Finding' else n_disease
        
        # 실제 데이터가 목표치보다 적으면 있는 거 다 씀
        actual_count = len(dfs[label])
        if actual_count < n_sample:
            n_sample = actual_count
            
        sampled = dfs[label].sample(n=n_sample, random_state=42)
        train_fragments.append(sampled)
        
    train_df = pd.concat(train_fragments)
    train_df = shuffle(train_df, random_state=42).reset_index(drop=True)
    
    print(f"✅ Train Set Completed (Balanced): {len(train_df)} images")
    print(f"✅ Test Set Completed (Imbalanced): {len(test_df)} images")
    
    # Test 데이터셋에 모든 클래스가 최소 1개 이상 존재하는지 확인
    print("\n[Test Set 클래스 분포 확인]")
    for label in target_classes:
        count = len(test_df[test_df['Finding Labels'].str.contains(label)])
        print(f" - {label}: {count}장")
        if count == 0:
            print(f"⚠️ 경고: Test Set에 '{label}' 데이터가 없습니다! AUC 계산 시 NaN이 발생할 수 있습니다.")

    return train_df, test_df

# =========================================================
# [2] Dataset 클래스
#  - 파일명 공백 제거 / 존재하지 않는 파일 필터링 기능
# =========================================================
class LungMaskDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, classes, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.classes = classes
        
        # 1. 파일명 공백 제거
        df['Image Index'] = df['Image Index'].astype(str).str.strip()
        
        # 2. [핵심] 실제 파일이 존재하는 행만 남기기 (유효성 검사)
        print(f"🔍 데이터 무결성 검사 시작... (총 {len(df)}개)")
        
        valid_indices = []
        missing_count = 0
        
        for idx in tqdm(range(len(df)), desc="Checking Files"):
            fname = df.iloc[idx]['Image Index']
            file_path = os.path.join(img_dir, fname)
            
            # 파일이 실제로 있으면 리스트에 추가
            if os.path.exists(file_path):
                valid_indices.append(idx)
            else:
                missing_count += 1
                # 처음 5개만 예시로 출력
                if missing_count <= 5:
                    print(f"   [Skip] Missing: {fname}")

        if missing_count > 0:
            print(f"⚠️ 총 {missing_count}개의 이미지가 없어 제외했습니다.")
        
        # 존재하는 데이터만 가지고 DataFrame 재생성
        self.df = df.iloc[valid_indices].reset_index(drop=True)
        self.df['labels_list'] = self.df['Finding Labels'].apply(lambda x: x.split('|'))
        
        print(f"✅ 최종 유효 데이터셋 크기: {len(self.df)}개")

    def __len__(self):
        return len(self.df)
    
    def apply_mask_strategy(self, img_path, mask_path):
        try:
            # numpy로 읽어서 디코딩 (한글/특수문자/WSL 경로 호환성)
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            # 마스크가 없으면 원본만 리턴 (유연한 처리)
            if os.path.exists(mask_path):
                mask_array = np.fromfile(mask_path, np.uint8)
                mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            else:
                mask = None
            
        except Exception:
            return None
        
        if img is None: return None
        if mask is None: return Image.fromarray(img).convert('RGB')
        
        # 크기 불일치 보정
        if img.shape != mask.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        
        points = cv2.findNonZero(mask)
        if points is not None:
            x, y, w, h = cv2.boundingRect(points)
            crop_img = masked_img[y:y+h, x:x+w]
        else:
            crop_img = masked_img
            
        return Image.fromarray(crop_img).convert('RGB')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['Image Index']
        
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)
        
        image = self.apply_mask_strategy(img_path, mask_path)
        
        # 만약 읽는 도중 파일이 깨져있다면 검은 화면 반환 (학습 중단 방지)
        if image is None:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        label_vec = torch.zeros(len(self.classes), dtype=torch.float32)
        for i, cls_name in enumerate(self.classes):
            if cls_name in row['labels_list']:
                label_vec[i] = 1.0
                
        return image, label_vec
# =========================================================
# [3] 모델 및 학습 함수
# =========================================================
def get_efficientnet_model(num_classes, device):
    # weights 매개변수를 사용하여 경고 메시지 방지 및 최신 방식 적용
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    # tqdm 옵션 수정: Linux 터미널에서 깨짐 방지 (ascii=True, dynamic_ncols=True)
    for inputs, targets in tqdm(loader, desc="Training", leave=False, dynamic_ncols=True):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, device, target_names):
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        # tqdm 옵션 유지
        for inputs, targets in tqdm(loader, desc="Evaluating", leave=False, dynamic_ncols=True):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    # 데이터가 없으면 빈 값 반환
    if not all_targets:
        return 0.0, np.array([]), np.array([])
        
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    
    # [수정] 전체 Macro AUC 계산 (NaN 방지)
    try:
        # y_true에 클래스가 하나밖에 없는 경우(모두 0 또는 모두 1) 에러가 나므로 예외처리
        auc = roc_auc_score(all_targets, all_preds, average='macro')
    except ValueError:
        auc = 0.0  # 계산 불가능할 경우 0 처리
        
    return auc, all_targets, all_preds


# =========================================================
# [4] 성능평가
# =========================================================

def visualize_performance(y_true, y_pred, target_classes, output_dir="./results"):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, confusion_matrix, classification_report
    
    # 저장 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ---------------------------------------------------------
    # 1. ROC Curve 시각화
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(target_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            
            # [수정 핵심] auc 변수명 충돌 방지를 위해 metrics.auc로 명시적 호출
            roc_auc_val = metrics.auc(fpr, tpr) 
            
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc_val:.4f})')
        except Exception as e:
            print(f"⚠️ {label} ROC Curve 생성 실패: {e}")

    plt.plot([0, 1], [0, 1], 'k--', lw=2) 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Multi-label ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    save_path_roc = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(save_path_roc)
    print(f"📈 ROC Curve 저장 완료: {save_path_roc}")
    # plt.show() # 필요 시 주석 해제

    # ---------------------------------------------------------
    # 2. Confusion Matrix 시각화
    # ---------------------------------------------------------
    threshold = 0.5
    y_pred_binary = (y_pred > threshold).astype(int)

    n_classes = len(target_classes)
    cols = 2
    rows = (n_classes + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()

    for i, label in enumerate(target_classes):
        try:
            cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                        xticklabels=['Negative', 'Positive'], 
                        yticklabels=['Negative', 'Positive'])
            axes[i].set_title(f'Confusion Matrix - {label}')
            axes[i].set_ylabel('Actual')
            axes[i].set_xlabel('Predicted')
        except Exception as e:
            print(f"⚠️ {label} Confusion Matrix 생성 실패: {e}")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_path_cm = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(save_path_cm)
    print(f"📉 Confusion Matrices 저장 완료: {save_path_cm}")
    # plt.show()

    # ---------------------------------------------------------
    # 3. 텍스트 리포트
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📋 Classification Report (Threshold = 0.5)")
    print("="*50)
    print(classification_report(y_true, y_pred_binary, target_names=target_classes, zero_division=0))






# =========================================================
# [Main Execution Flow]
# =========================================================
if __name__ == "__main__":
    # --- [수정] WSL 경로 설정 ---
    # Windows의 D: 드라이브 -> /mnt/d
    # Windows의 C: 드라이브 -> /mnt/c
    RAW_IMG_DIR = "/mnt/d/lung_xray/final_denoised"          
    MASK_DIR = "/mnt/d/lung_xray/final_contour_masks" 
    CSV_PATH = "/mnt/d/lung_xray/Data_Entry_processed_Final.csv" 
    
    TARGET_CLASSES = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis'] 
    
    # [수정] CPU 학습 효율을 위해 배치 사이즈 증가 (RAM 부족 시 16으로 조절)
    BATCH_SIZE = 32 
    EPOCHS = 5
    LR = 1e-4
    
    train_df, test_df = prepare_datasets(CSV_PATH, TARGET_CLASSES)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = LungMaskDataset(train_df, RAW_IMG_DIR, MASK_DIR, TARGET_CLASSES, transform)
    test_ds = LungMaskDataset(test_df, RAW_IMG_DIR, MASK_DIR, TARGET_CLASSES, transform)
    
    # [수정] Linux/WSL 환경에서는 num_workers를 높여야 CPU가 쉬지 않고 일합니다.
    # 안전하게 CPU 전체 코어 수 사용
    num_workers = multiprocessing.cpu_count()
    print(f"🚀 Workers setting: {num_workers} cores")

    # [수정] pin_memory=False (CPU 학습 시에는 False가 오버헤드가 적음)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    device = torch.device("cpu") # 명시적으로 CPU 지정
    print(f"🚀 Device: {device} (Optimized for Multi-core)")
    
    model = get_efficientnet_model(len(TARGET_CLASSES), device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 학습 시작
    print("🚀 Training Start...")
    
    # 모델 저장 경로 설정
    SAVE_PATH = "./best_model.pth"
    best_auc = 0.0

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 테스트셋으로 성능 평가
        auc, _, _ = evaluate(model, test_loader, device, TARGET_CLASSES)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f} | Test AUC: {auc:.4f}")
        
        # [수정 2] 성능이 가장 좋을 때(또는 매 에폭마다) 모델 저장 코드 추가
        # 여기서는 단순히 매 에폭마다 덮어쓰거나, AUC가 갱신될 때 저장합니다.
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  💾 Model Saved! (Best AUC: {best_auc:.4f})")

    print(f"\n✅ 학습 완료! 모델이 저장되었습니다: {SAVE_PATH}")
    
    # ---------------------------------------------------------
    # 저장된 모델을 다시 불러와서 최종 평가 (선택 사항)
    # ---------------------------------------------------------
    model.load_state_dict(torch.load(SAVE_PATH)) 
    
    print("\n====[Final Report]====")
    final_auc, y_true, y_pred = evaluate(model, test_loader, device, TARGET_CLASSES)
    # ... (이하 시각화 코드 동일) ...




    print("\n====[Final Report]====")
    # 1. 평가 수행
    final_auc, y_true, y_pred = evaluate(model, test_loader, device, TARGET_CLASSES)
    
    # 2. 결과 텍스트 출력
    print(f"Overall Macro AUC: {final_auc:.4f}")
    
    for i, cls in enumerate(TARGET_CLASSES):
        try:
            # 안전한 AUC 계산
            if len(np.unique(y_true[:, i])) > 1:
                cls_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                print(f" - {cls:<15} AUC: {cls_auc:.4f}")
            else:
                print(f" - {cls:<15} AUC: N/A (데이터 부족)")
        except:
            print(f" - {cls:<15} AUC: Error")
            
    # =========================================================
    # [추가] 3. 시각화 함수 호출
    # =========================================================
    # y_true, y_pred는 evaluate 함수에서 이미 numpy array로 변환되어 나옴
    visualize_performance(y_true, y_pred, TARGET_CLASSES, output_dir="./results")



