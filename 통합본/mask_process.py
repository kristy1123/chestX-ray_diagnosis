import cv2
import numpy as np
import os
import glob
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
from skimage import img_as_ubyte
from skimage import io, color, exposure, img_as_float, img_as_ubyte
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from scipy import ndimage
from tqdm import tqdm 


# =================================================
#             1. Image Denoising
# =================================================
        # Total Variation (TV) Denoising을 사용해 데이터 스머징
        # 목적 : 이미지를 뭉개서 최대한 뭉뚝한 마스크를 생성하기 위함.
def apply_tv_denoising(input_folder, output_folder, weight=0.1):
    """
    폴더 내 모든 이미지에 Total Variation Denoising을 적용합니다.
    
    :param weight: Denoising 강도 (클수록 더 뭉개짐/부드러워짐).
                   X-ray의 경우 보통 0.05 ~ 0.2 사이가 적당합니다.
    """
    # 1. 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 2. 이미지 파일 리스트 가져오기
    image_files = glob.glob(os.path.join(input_folder, '*.png'))
    # jpg도 있다면 아래 주석 해제
    # image_files.extend(glob.glob(os.path.join(input_folder, '*.jpg')))
    
    print(f"🚀 총 {len(image_files)}장 TV Denoising 시작 (Weight={weight})...")
    
    for file_path in tqdm(image_files):
        filename = os.path.basename(file_path)
        
        # 3. 이미지 로드 (GrayScale)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # 4. TV Denoising 적용 (Chambolle 알고리즘)
        # weight: 노이즈 제거 강도. 값이 클수록 이미지가 더 매끄러워(cartoon-like)집니다.
        # 보여주신 예시는 꽤 강하게 들어간 편이라 0.1~0.2 정도가 적당해 보입니다.
        denoised_float = denoise_tv_chambolle(img, weight=weight, channel_axis=None)
        
        # 5. 결과 변환 (Float 0~1 -> Uint8 0~255)
        # skimage 결과는 float 형태이므로 다시 이미지 포맷으로 변환해야 합니다.
        denoised_img = img_as_ubyte(denoised_float)
        
        # 6. 저장
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, denoised_img)
        
    print("-" * 50)
    print("✅ 모든 이미지 변환 완료!")
    print(f"📂 저장 경로: {output_folder}")
    print("-" * 50)




# =================================================
#             실행
# =================================================
input_dir = "D:/lung_xray/final_imgsets" # 원본 이미지 폴더
output_dir = "D:/lung_xray/final_denoised"    # 결과 저장될 폴더

# weight 값을 조절하여 원하는 뭉개짐 정도를 찾으세요.
# 0.05 (약함) -> 0.1 (보통/추천) -> 0.2 (강함/예시 사진과 비슷)
apply_tv_denoising(input_dir, output_dir, weight=0.15)