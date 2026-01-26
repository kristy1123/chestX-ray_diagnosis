import os
import cv2
import numpy as np
import pandas as pd
from skimage.exposure import match_histograms  # [추가] 히스토그램 매칭
from skimage.restoration import denoise_tv_chambolle
from skimage import img_as_float, img_as_ubyte
from skimage.segmentation import morphological_chan_vese
from tqdm import tqdm

# [중요] CPU 병렬 처리 충돌 방지 설정
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# =========================================================
# [Part 1] 유틸리티 및 알고리즘 함수 모음
# =========================================================

def clear_image_borders(mask, margin=10):
    h, w = mask.shape
    mask[:margin, :] = 0
    mask[h-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, w-margin:] = 0
    return mask

def get_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0: return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def merge_nearby_contours(main_cnt, all_contours, mask_shape, dist_limit=30):
    h, w = mask_shape
    temp_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(temp_mask, [main_cnt], -1, 255, -1)
    
    for cnt in all_contours:
        if np.array_equal(cnt, main_cnt): continue
        center = get_centroid(cnt)
        if center is None: continue
        dist = cv2.pointPolygonTest(main_cnt, center, True)
        if dist > -dist_limit: 
            cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
            
    merged_contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if merged_contours:
        return max(merged_contours, key=cv2.contourArea)
    return main_cnt

def cut_background_leakage(mask, separation_strength=0.25):
    if cv2.countNonZero(mask) == 0: return mask
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
    thresh_val = max_val * separation_strength
    _, core_mask = cv2.threshold(dist_transform, thresh_val, 255, cv2.THRESH_BINARY)
    return core_mask.astype(np.uint8)

def find_lungs_strategy(mask):
    """BBox 추출 로직"""
    h, w = mask.shape
    center_x = w // 2
    mask = clear_image_borders(mask, margin=15)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    left_candidates = []  
    right_candidates = [] 
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w * h * 0.02): continue
        center = get_centroid(cnt)
        if center is None: continue
        if center[0] < center_x: left_candidates.append(cnt)
        else: right_candidates.append(cnt)
            
    final_r_lung = None
    final_l_lung = None
    
    if left_candidates:
        main_cnt = max(left_candidates, key=cv2.contourArea)
        final_r_lung = merge_nearby_contours(main_cnt, left_candidates, (h, w))
    if right_candidates:
        main_cnt = max(right_candidates, key=cv2.contourArea)
        final_l_lung = merge_nearby_contours(main_cnt, right_candidates, (h, w))
        
    found_r = final_r_lung is not None
    found_l = final_l_lung is not None
    
    r_bbox = cv2.boundingRect(final_r_lung) if found_r else None
    l_bbox = cv2.boundingRect(final_l_lung) if found_l else None
    
    status = "Success" if (found_r and found_l) else "Fail"
    
    refined_mask = np.zeros((h, w), dtype=np.uint8)
    if found_r: cv2.drawContours(refined_mask, [final_r_lung], -1, 255, -1)
    if found_l: cv2.drawContours(refined_mask, [final_l_lung], -1, 255, -1)
        
    return r_bbox, l_bbox, status, refined_mask

def is_point_in_bbox(x, y, bbox):
    if bbox is None: return False
    x_min, y_min, w, h = bbox
    x_max, y_max = x_min + w, y_min + h
    return x_min <= x <= x_max and y_min <= y <= y_max

def create_convex_hull_with_bbox(binary_mask, r_bbox, l_bbox):
    """Convex Hull 생성 로직"""
    h, w = binary_mask.shape
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_hull_mask = np.zeros_like(binary_mask)
    
    valid_cnts = []
    
    margin_w = int(w * 0.20)
    margin_h_top = int(h * 0.20)
    margin_h_bottom = int(h * 0.10)

    use_bbox_logic = (r_bbox is not None) or (l_bbox is not None)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w * h * 0.01): continue
        
        center = get_centroid(cnt)
        if center is None: continue
        cx, cy = center
        
        is_valid = False
        if use_bbox_logic:
            in_right = is_point_in_bbox(cx, cy, r_bbox)
            in_left = is_point_in_bbox(cx, cy, l_bbox)
            if in_right or in_left: is_valid = True
        else:
            if (margin_w < cx < w - margin_w) and (margin_h_top < cy < h - margin_h_bottom):
                is_valid = True
                
        if is_valid: valid_cnts.append(cnt)
            
    for cnt in valid_cnts:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(final_hull_mask, [hull], -1, 255, -1)
        
    return final_hull_mask

def fast_active_contour_logic(raw_img, hull_mask):
    """ACM 적용"""
    TARGET_SIZE = (512, 512)
    SMALL_SIZE = (128, 128)
    
    img_512 = cv2.resize(raw_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    mask_512 = cv2.resize(hull_mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    
    # [참고] 여기서도 CLAHE를 한번 더 쓰긴 하지만, 
    # 이미 앞에서 전처리된 이미지가 들어오므로 효과가 중첩되어 더 강한 엣지를 얻음.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_512)
    img_float_512 = img_as_float(img_enhanced)
    
    # Stage 1: Coarse (128px)
    img_small = cv2.resize(img_float_512, SMALL_SIZE, interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask_512, SMALL_SIZE, interpolation=cv2.INTER_NEAREST)
    
    cv_mask_small = morphological_chan_vese(
        img_small, num_iter=70, init_level_set=mask_small, 
        smoothing=2, lambda1=1, lambda2=1
    )
    
    # Stage 2: Fine (512px)
    mask_upscaled = cv2.resize(cv_mask_small.astype(np.float32), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    final_cv_mask = morphological_chan_vese(
        img_float_512, num_iter=7, init_level_set=mask_upscaled, 
        smoothing=3, lambda1=1, lambda2=1
    )
    
    return (final_cv_mask.astype(np.uint8)) * 255

# =========================================================
# [Part 2] 통합 파이프라인 함수 (전처리 포함)
# =========================================================

def run_pipeline_single_image(raw_path, tuned_mask_path, orig_mask_path, ref_img, output_dir):
    """
    Step 0 (선명도 개선) -> Step 1 (BBox) -> Step 2 (Hull) -> Step 3 (ACM)
    """
    filename = os.path.basename(raw_path)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 1. 원본 이미지 로드
    raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    tuned_mask = cv2.imread(tuned_mask_path, cv2.IMREAD_GRAYSCALE)
    orig_mask = cv2.imread(orig_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if raw_img is None or tuned_mask is None:
        print(f"❌ Error loading: {filename}")
        return

    # ---------------------------------------------------------
    # [NEW] Step 0-1: 이미지 선명도 개선 (Histogram Match + CLAHE)
    # ---------------------------------------------------------
    # 이것이 이제 파이프라인의 "새로운 원본(enhanced_img)"이 됩니다.
    try:
        # 
        # 1. 히스토그램 매칭 (Ref 이미지의 톤을 따름)
        matched = match_histograms(raw_img, ref_img, channel_axis=None)
        matched = matched.astype('uint8') # 필수 변환
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        clahe_pre = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe_pre.apply(matched)
        
    except Exception as e:
        print(f"⚠️ 전처리 에러 ({filename}): {e}")
        enhanced_img = raw_img # 에러 발생 시 원본 사용

    # ---------------------------------------------------------
    # [Mod] Step 0-2: 노이즈 제거 (TV Chambolle)
    # ---------------------------------------------------------
    # 이제 enhanced_img를 입력으로 사용합니다.
    img_float = img_as_float(enhanced_img)
    denoised_float = denoise_tv_chambolle(img_float, weight=0.1) 
    denoised_img = img_as_ubyte(denoised_float)

    # ---------------------------------------------------------
    # Step 1: 마스크 BBox 추출 및 최적 마스크 선정
    # ---------------------------------------------------------
    r_bbox, l_bbox, status, best_mask = find_lungs_strategy(tuned_mask.copy())
    
    if status == "Fail" and orig_mask is not None:
        if orig_mask.shape != tuned_mask.shape:
            orig_mask = cv2.resize(orig_mask, (tuned_mask.shape[1], tuned_mask.shape[0]))
            
        merged_mask = cv2.bitwise_or(tuned_mask, orig_mask)
        cut_mask = cut_background_leakage(merged_mask)
        r_bbox_2, l_bbox_2, status_2, best_mask_2 = find_lungs_strategy(cut_mask)
        
        if status_2 == "Success" or (r_bbox_2 is not None or l_bbox_2 is not None):
            r_bbox, l_bbox = r_bbox_2, l_bbox_2
            best_mask = best_mask_2

    # ---------------------------------------------------------
    # Step 2: Convex Hull 생성
    # ---------------------------------------------------------
    hull_mask = create_convex_hull_with_bbox(best_mask, r_bbox, l_bbox)

    # ---------------------------------------------------------
    # Step 3: ACM (Active Contour Model) 적용
    # ---------------------------------------------------------
    # 선명도가 개선되고 노이즈가 제거된 denoised_img를 사용
    final_mask_512 = fast_active_contour_logic(denoised_img, hull_mask)
    
    h, w = raw_img.shape
    final_mask = cv2.resize(final_mask_512, (w, h), interpolation=cv2.INTER_NEAREST)

    # ---------------------------------------------------------
    # 결과 저장 및 시각화
    # ---------------------------------------------------------
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, final_mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    
    vis_dir = os.path.join(output_dir, "visual_check")
    if not os.path.exists(vis_dir): os.makedirs(vis_dir)
    
    # 시각화: [ 원본 | 전처리된 이미지 | 최종 마스크 ]
    vis_raw = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    vis_enhanced = cv2.cvtColor(denoised_img, cv2.COLOR_GRAY2BGR)
    vis_final = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    
    if r_bbox: cv2.rectangle(vis_enhanced, (r_bbox[0], r_bbox[1]), (r_bbox[0]+r_bbox[2], r_bbox[1]+r_bbox[3]), (0, 0, 255), 2)
    if l_bbox: cv2.rectangle(vis_enhanced, (l_bbox[0], l_bbox[1]), (l_bbox[0]+l_bbox[2], l_bbox[1]+l_bbox[3]), (0, 0, 255), 2)

    combined_vis = np.hstack([vis_raw, vis_enhanced, vis_final])
    combined_vis = cv2.resize(combined_vis, (0, 0), fx=0.5, fy=0.5) 
    cv2.imwrite(os.path.join(vis_dir, f"vis_{filename}"), combined_vis)
    
    return True

# =========================================================
# [Main] 폴더 일괄 처리 실행
# =========================================================

def main():
    # --- 경로 설정 ---
    INPUT_RAW_DIR = "D:/lung_xray/final_denoised"
    INPUT_TUNED_MASK_DIR = "D:/lung_xray/final_denoised/lung_mask_tuned"
    INPUT_ORIG_MASK_DIR = "D:/lung_xray/final_denoised/lung_mask"
    
    # [설정] 기준 이미지 경로 (Histogram Matching용) - 경로 수정 필요
    # 가장 선명하고 이상적인 엑스레이 샘플 이미지 1장을 지정하세요.
    REFERENCE_IMG_PATH = "D:/lung_xray/reference_sample.png" 
    
    # 최종 결과 저장 폴더
    OUTPUT_DIR = "D:/lung_xray/final_denoised/final_pipeline_result_enhanced"

    # 1. 기준 이미지 로드 (한 번만 로드하여 재사용)
    if not os.path.exists(REFERENCE_IMG_PATH):
        print(f"❌ 기준 이미지가 없습니다: {REFERENCE_IMG_PATH}")
        print("경로를 확인하거나, 샘플 이미지를 해당 위치에 넣어주세요.")
        return

    ref_img = cv2.imread(REFERENCE_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        print("❌ 기준 이미지 로드 실패. 파일 손상 여부를 확인하세요.")
        return
    print("✅ 기준 이미지 로드 완료.")

    # 2. 파일 리스트 가져오기
    mask_files = glob.glob(os.path.join(INPUT_TUNED_MASK_DIR, '*.png'))
    print(f"🚀 파이프라인 시작 (선명도 개선 모드): 총 {len(mask_files)}장 처리 예정")

    for mask_path in tqdm(mask_files):
        filename = os.path.basename(mask_path)
        
        raw_path = os.path.join(INPUT_RAW_DIR, filename)
        orig_mask_path = os.path.join(INPUT_ORIG_MASK_DIR, filename)
        
        if not os.path.exists(raw_path):
            continue
            
        # 단일 이미지 파이프라인 실행 (ref_img 인자 전달)
        run_pipeline_single_image(raw_path, mask_path, orig_mask_path, ref_img, OUTPUT_DIR)

    print(f"✅ 모든 처리 완료. 결과 경로: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()