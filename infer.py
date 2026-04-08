import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import models_YaTC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def get_args_parser():
    parser = argparse.ArgumentParser('YaTC inference', add_help=False)
    parser.add_argument('--model_path', default='./output_dir/best_f1_model.pth', type=str, 
                        help='Path to the trained model weights')
    parser.add_argument('--data_path', required=True, type=str, 
                        help='Path to the input image file or directory')
    parser.add_argument('--nb_classes', required=True, type=int, 
                        help='Number of the classification types')
    parser.add_argument('--device', default='cuda', type=str, 
                        help='Device to use for inference (cuda or cpu)')
    return parser

def main(args):
    device = torch.device(args.device)

    # 모델 초기화 (fine-tune.py와 동일한 TraFormer_YaTC 사용)
    model = models_YaTC.TraFormer_YaTC(
        num_classes=args.nb_classes,
        drop_path_rate=0.0
    )
    
    # 모델 가중치 로드
    if not os.path.exists(args.model_path):
        print(f"Model weight file not found: {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    # 데이터 전처리 파이프라인 (fine-tune.py의 build_dataset과 동일)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 단일 이미지 추론 내부 함수
    def infer_single_image(image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device) # 배치 차원 추가
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(img_tensor)
                _, pred = output.topk(1, 1, True, True)
                
        return pred.item()

    # 입력 경로가 파일인지 디렉토리인지 판별하여 처리
    if os.path.isfile(args.data_path):
        pred_class = infer_single_image(args.data_path)
        print(f"File: {args.data_path} | Predicted Class Index: {pred_class}")
        
    elif os.path.isdir(args.data_path):
        print(f"Running inference on directory: {args.data_path}")
        
        # 1. PyTorch ImageFolder와 동일하게 알파벳 순으로 클래스 폴더 정렬 및 인덱스 매핑
        classes = sorted([d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        y_true = []
        y_pred = []

        for dirname in classes:
            dir_path = os.path.join(args.data_path, dirname)
            true_label = class_to_idx[dirname]
            
            for filename in os.listdir(dir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(dir_path, filename)
                    
                    # 단일 이미지 추론
                    pred_class = infer_single_image(file_path)
                    
                    # 결과 누적
                    y_true.append(true_label)
                    y_pred.append(pred_class)
                    if pred_class is not true_label:
                        print(f"Class: {dirname} | File: {filename} | True: {true_label} | Pred: {pred_class}")

        # 2. 전체 데이터세트에 대한 평가 지표 계산
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            macro = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

            cm = confusion_matrix(y_true, y_pred)
            
            # --- Confusion Matrix 저장 코드 추가 시작 ---
            save_path = "confusion_matrix.png"
            
            # 클래스 수가 많으므로 이미지 크기를 충분히 크게 잡습니다 (예: 20x20)
            plt.figure(figsize=(25, 20)) 
            
            # Seaborn Heatmap 생성
            # annot=False: 클래스가 너무 많으면 숫자가 겹치므로 끕니다. 
            # 필요하다면 annot=True로 변경하세요.
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            
            plt.title('Inference Confusion Matrix', fontsize=20)
            plt.xlabel('Predicted Label', fontsize=15)
            plt.ylabel('True Label', fontsize=15)
            plt.xticks(rotation=90, fontsize=8) # x축 글자 90도 회전
            plt.yticks(fontsize=8)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300) # 고해상도 저장
            print(f"\n[Info] Confusion Matrix saved to: {save_path}")
            plt.close() # 메모리 해제

            print("\n" + "="*30)
            print("Inference Evaluation Results")
            print("="*30)
            print(f"Total Samples : {len(y_true)}")
            print(f"Accuracy      : {acc:.4f}")
            print(f"Precision     : {macro[0]:.4f}")
            print(f"Recall        : {macro[1]:.4f}")
            print(f"F1 Score      : {macro[2]:.4f}")
            print("="*30)

            print("\n[Classification Report]")
            print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

            # 오답 분석: 각 클래스별로 틀린 개수 계산
            print("\n[Error Analysis - Top Missed Classes]")
            # 대각선 요소(맞춘 개수)를 0으로 설정하여 틀린 개수만 추출
            cm_errors = cm.copy()
            np.fill_diagonal(cm_errors, 0)
            
            # 행(True label) 기준 오답 합계 계산
            row_errors = np.sum(cm_errors, axis=1)
            
            # 오답이 많은 순서대로 정렬하여 출력
            error_indices = np.argsort(row_errors)[::-1]
            for idx in error_indices:
                if row_errors[idx] > 0:
                    print(f"Class '{classes[idx]}' missed {row_errors[idx]} times.")
                    
        else:
            print("No image files found in the specified directory.")
    else:
        print("Invalid data_path. Please provide a valid file or directory path.")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)