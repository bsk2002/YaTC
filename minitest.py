import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 로드 및 전처리 (기존 설정 유지)
BASE_PATH = './tmp'
CLASSES = sorted([d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))])[:200]

def load_diagnostic_data(samples_per_class=50): # 통계를 위해 샘플 수를 조금 늘림
    data, labels = [], []
    for cls in CLASSES:
        cls_path = os.path.join(BASE_PATH, cls)
        img_files = [f for f in os.listdir(cls_path) if f.endswith('.png')][:samples_per_class]
        for f in img_files:
            img = Image.open(os.path.join(cls_path, f)).convert('L')
            # 로그 변환으로 스케일 압축
            data.append(np.log1p(np.array(img).flatten()))
            labels.append(cls)
    return np.array(data), np.array(labels)

X, y = load_diagnostic_data()

# 2. 피처 필터링 (분산이 낮은 바이트 제거 - 모델 효율화)
pixel_vars = X.var(axis=0)
mask = pixel_vars > 0.5 
X_filtered = X[:, mask]

# 3. 학습/테스트 데이터 분할 (8:2)
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Random Forest 모델 학습
print(f"모델 학습 시작 (클래스: {len(CLASSES)}, 피처: {X_filtered.shape[1]})")
rf = RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# 5. 성능 측정 (Top-1 & Top-5)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

# Top-1 Accuracy
top1_acc = accuracy_score(y_test, y_pred)

# Top-5 Accuracy 계산
top5_idx = np.argsort(y_prob, axis=1)[:, -5:]
top5_correct = [y_test[i] in rf.classes_[top5_idx[i]] for i in range(len(y_test))]
top5_acc = np.mean(top5_correct)

print("-" * 30)
print(f"🎯 [결과] Top-1 Accuracy: {top1_acc*100:.2f}%")
print(f"🎯 [결과] Top-5 Accuracy: {top5_acc*100:.2f}%")
print("-" * 30)

# 6. 혼동 행렬(Confusion Matrix) 분석 - 가장 많이 틀리는 클래스 찾기
cm = confusion_matrix(y_test, y_pred)
# 자기 자신(정답)은 제외하고 틀린 것만 추출
np.fill_diagonal(cm, 0)

# 상위 10개의 혼동 쌍 추출
confused_indices = np.unravel_index(np.argsort(cm, axis=None)[::-1][:10], cm.shape)
confused_pairs = []

print("\n🔥 [가장 많이 혼동되는 클래스 Top 10]")
for i, j in zip(*confused_indices):
    if cm[i, j] > 0:
        print(f"Count: {cm[i, j]} | {rf.classes_[i]}  <--->  {rf.classes_[j]}")
