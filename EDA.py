import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from umap import UMAP

# 1. 설정 및 데이터 로드 (Path: tmp/SNI)
BASE_PATH = './tmp'
# 하위 디렉토리(클래스) 목록 가져오기
CLASSES = sorted([d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))])[:200]

def load_and_preprocess(samples_per_class=20):
    data = []
    labels = []
    img_shape = None
    
    print(f"데이터 로드 중... (클래스당 {samples_per_class}장 샘플링)")
    for cls in CLASSES:
        cls_path = os.path.join(BASE_PATH, cls)
        img_files = [f for f in os.listdir(cls_path) if f.endswith('.png')][:samples_per_class]
        
        for f in img_files:
            img = Image.open(os.path.join(cls_path, f)).convert('L')
            if img_shape is None: img_shape = np.array(img).shape
            # [개선] 0~255 바이트 데이터를 log1p 변환하여 스케일 압축
            data.append(np.log1p(np.array(img).flatten()))
            labels.append(cls)
            
    return np.array(data), np.array(labels), img_shape

X, y, SHAPE = load_and_preprocess()

# 2. 피처 필터링 (분산이 너무 낮은 픽셀 제거)
# 모든 클래스에서 똑같은 값을 가지는 바이트(헤더 등)는 UMAP의 거리를 왜곡시킵니다.
pixel_vars = X.var(axis=0)
threshold = 0.5  # 분산이 이보다 낮은 픽셀은 버림
mask = pixel_vars > threshold
X_filtered = X[:, mask]

print(f"원본 피처 수: {X.shape[1]} -> 필터링 후(유의미한 바이트): {X_filtered.shape[1]}")

# 3. 시각화 함수 정의
def run_enhanced_eda():
    fig = plt.figure(figsize=(18, 12))
    
    # --- [시각화 1] Variance Map (어디가 변별력 포인트인가?) ---
    plt.subplot(2, 2, 1)
    plt.imshow(pixel_vars.reshape(SHAPE), cmap='hot')
    plt.colorbar(label='Variance')
    plt.title("1. Variance Map (Yellow = Discriminative Bytes)")

    # --- [시각화 2] PCA 결과 (선형적 군집 확인) ---
    plt.subplot(2, 2, 2)
    X_scaled = RobustScaler().fit_transform(X_filtered)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_int, cmap='tab20', s=10, alpha=0.6)
    plt.title("2. PCA Projection (Linear Separability)")

    # --- [시각화 3] UMAP (비선형적 군집 확인 - Deep Dive) ---
    plt.subplot(2, 2, 3)
    # PCA로 노이즈를 먼저 잡고 UMAP 투영 (속도와 정확도 향상)
    pca_50 = PCA(n_components=min(50, X_filtered.shape[1]))
    X_pca_50 = pca_50.fit_transform(X_scaled)
    
    reducer = UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_pca_50)
    
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_int, cmap='tab20', s=10, alpha=0.6)
    plt.title("3. Improved UMAP (After Filtering & PCA)")

    # --- [시각화 4] 상위 변별력 픽셀 값 분포 ---
    plt.subplot(2, 2, 4)
    top_idx = np.argsort(pixel_vars)[-1] # 분산이 가장 높은 1개 픽셀
    for i in range(min(5, len(CLASSES))):
        cls_data = X[y == CLASSES[i]][:, top_idx]
        plt.hist(cls_data, alpha=0.5, label=f"Class {i}", bins=20)
    plt.title(f"4. Byte Distribution at Offset {top_idx}")
    plt.legend()

    plt.tight_layout()
    plt.show()

# 실행
run_enhanced_eda()
