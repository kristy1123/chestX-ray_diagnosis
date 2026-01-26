import cv2
import numpy as np
import os
import glob
import pandas as pd

def calculate_metrics(pred_mask, gt_mask):
    """
    Dice Score와 IoU를 계산하는 함수
    """
    # 0과 1로 이진화 (Thresholding)
    pred_mask = (pred_mask > 127).astype(np.uint8)
    gt_mask = (gt_mask > 127).astype(np.uint8)
    
    # 교집합 (Intersection)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    # 합집합 (Union)
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # 각각의 면적
    pred_area = pred_mask.sum()
    gt_area = gt_mask.sum()
    
    # 1. Dice Score (2 * 교집합 / (A + B))
    if (pred_area + gt_area) == 0:
        dice = 1.0 # 둘 다 공백이면 정답으로 처리
    else:
        dice = (2. * intersection) / (pred_area + gt_area)
        
    # 2. IoU (교집합 / 합집합)
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union
        
    return dice, iou

def evaluate_jsrt_performance(pred_dir, gt_dir, output_csv="jsrt_evaluation_results.csv"):
    """
    pred_dir: 내 모델이 생성한 마스크들이 있는 폴더
    gt_dir: JSRT(SCR) 정답 마스크들이 있는 폴더
    """
    pred_files = glob.glob(os.path.join(pred_dir, "*.png"))
    
    if len(pred_files) == 0:
        print("❌ 예측 마스크 파일이 없습니다. 경로를 확인하세요.")
        return

    results = []
    print(f"🚀 JSRT 성능 평가 시작 (총 {len(pred_files)}장)...")

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, filename)
        
        # 정답 파일이 존재하는지 확인
        if not os.path.exists(gt_path):
            print(f"⚠️ 정답 파일 없음 (Skip): {filename}")
            continue
            
        # 이미지 로드 (Grayscale)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if pred_img is None or gt_img is None:
            continue
            
        # [중요] 크기 맞추기: GT를 예측 크기에 맞춤 (또는 그 반대)
        if pred_img.shape != gt_img.shape:
            gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # 메트릭 계산
        dice, iou = calculate_metrics(pred_img, gt_img)
        
        results.append({
            "Image Index": filename,
            "Dice_Score": dice,
            "IoU_Score": iou
        })

    # 결과 저장
    df = pd.DataFrame(results)
    if not df.empty:
        mean_dice = df["Dice_Score"].mean()
        mean_iou = df["IoU_Score"].mean()
        
        print("\n" + "="*40)
        print(f"📊 최종 평가 결과 (N={len(df)})")
        print(f"✅ 평균 Dice Score : {mean_dice:.4f}")
        print(f"✅ 평균 IoU Score  : {mean_iou:.4f}")
        print("="*40)
        
        df.to_csv(output_csv, index=False)
        print(f"💾 상세 결과 저장됨: {output_csv}")
    else:
        print("❌ 평가할 데이터가 없습니다.")

# =========================================================
# [사용 방법]
# 1. 내 모델로 생성한 마스크 폴더 경로
my_pred_folder = "mnt/d/nodules-in-chest-xrays-jsrt/final_contour_masks"

# 2. 다운로드 받은 JSRT(SCR) 정답 마스크 폴더 경로
jsrt_gt_folder = "mnt/d/nodules-in-chest-xrays-jsrt/masks_answer"

# 실행
evaluate_jsrt_performance(my_pred_folder, jsrt_gt_folder)