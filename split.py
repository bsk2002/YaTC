import os
import shutil
import glob
import re
from tqdm import tqdm

def split_dataset_dynamic(src_base_path, dst_base_path):
    # 1. 대상 SNI 폴더 목록 가져오기
    sni_folders = [f for f in os.listdir(src_base_path) 
                   if os.path.isdir(os.path.join(src_base_path, f)) 
                   and f not in ['train', 'test', 'valid']]

    for sni in tqdm(sni_folders, desc="Processing SNIs"):
        sni_src_path = os.path.join(src_base_path, sni)
        all_files = glob.glob(os.path.join(sni_src_path, "*.png"))
        
        if not all_files:
            continue

        # 2. 그룹화 규칙: '숫자-connection' 앞부분까지 추출
        # 예: "abc_12345-connection..." -> "abc"
        group_dict = {}
        for f_path in all_files:
            file_name = os.path.basename(f_path)
            # 정규표현식: 숫자들(\d+) 뒤에 -connection이 오는 패턴을 찾아 그 앞까지 슬라이싱
            match = re.search(r'_\d+-connection', file_name)
            if match:
                prefix = file_name[:match.start()]
            else:
                # 패턴이 없을 경우 기존처럼 __ 활용 혹은 파일명 전체 사용
                prefix = file_name.split("__")[0] if "__" in file_name else "others"
                
            if prefix not in group_dict:
                group_dict[prefix] = []
            group_dict[prefix].append(f_path)

        # 3. 동적 개수 설정 (각 그룹의 최소 개수 파악)
        # 5개 그룹이 안 될 수도 있으므로 실제 존재하는 그룹 전체 혹은 상위 5개 대상
        target_prefixes = sorted(group_dict.keys())[:5]
        
        if len(target_prefixes) < 1:
            continue

        # 각 그룹별 파일 개수 중 가장 작은 값을 기준으로 삼음 (동적 할당)
        min_count = min([len(group_dict[p]) for p in target_prefixes])
        
        # 8:1:1로 나누기 위해 최소 10개는 있어야 함 (10개 미만이면 해당 SNI 건너뜀)
        if min_count < 10:
            print(f"\n[Skip] {sni}: 그룹당 파일 개수가 너무 적습니다. (최소: {min_count})")
            continue

        # 8:1:1 정수 배분을 위한 계산
        n_train = int(min_count * 0.8)
        n_test = int(min_count * 0.1)
        n_valid = min_count - n_train - n_test # 나머지 전부 (약 10%)

        # 4. 파일 배분 및 복사
        for prefix in target_prefixes:
            files_in_group = sorted(group_dict[prefix])[:min_count]
            
            # 배분 지점 설정
            splits = {
                'train': files_in_group[:n_train],
                'test': files_in_group[n_train:n_train+n_test],
                'valid': files_in_group[n_train+n_test:]
            }

            for split_name, subset in splits.items():
                target_dir = os.path.join(dst_base_path, split_name, sni)
                os.makedirs(target_dir, exist_ok=True)
                
                for src_file in subset:
                    shutil.copy2(src_file, os.path.join(target_dir, os.path.basename(src_file)))

if __name__ == '__main__':
    source_path = "./tmp"
    destination_path = "./data/new_captured"
    
    print("동적 데이터 분할을 시작합니다...")
    split_dataset_dynamic(source_path, destination_path)
    print(f"\n작업 완료! '{destination_path}' 폴더를 확인하세요.")