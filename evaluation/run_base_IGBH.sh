# ============================================================================
# [한국어] Baseline(DGL 순정 mmap/CPU-load) IGBH 학습 실행 스크립트
#          (run_base_IGBH.sh)
#
# === 파일의 역할 ===
# BaM/GIDS 경로를 전혀 쓰지 않는 **순수 DGL + CPU mmap feature 로더** baseline.
# heterogeneous_train_baseline.py로 IGBH full을 학습해, GIDS/BaM 대비 속도·병목
# 차이를 정량화하는 레퍼런스 성능 수치를 얻는다. "SSD → host memory mmap →
# CPU→GPU 전송"이라는 전통 DGL 경로다.
#
# === 전체 아키텍처에서의 위치 ===
# heterogeneous_train_baseline.py → DGL 표준 DataLoader / NeighborSampler →
# IGBHeteroDGLDatasetMassive (dataloader.py). feature 텐서는 numpy memmap 또는
# in_memory 로딩. gids_module/bam 경로 미사용.
#
# === 타 모듈과의 연결 ===
# - dataloader.py의 IGBHeteroDGLDatasetMassive / IGB260MDGLDataset 사용.
# - GIDS 플래그(--GIDS, --cache_size, --num_ele, --page_size 등) 전혀 없음.
# - run_BaM_IGBH.sh, run_GIDS_IGBH.sh와 같은 학습 하이퍼파라미터를 유지해
#   storage-path만 다르게 한 apples-to-apples 비교 기준이 된다.
#
# === 주요 함수/구조체 요약 ===
# 셸 한 줄. 플래그:
#   --model_type rsage / --num_layers 3 / --fan_out '10,5,5' : 동일 모델 구성.
#   --emb_size 1024                                          : paper feat dim.
# ============================================================================
# [한국어] CUDA_VISIBLE_DEVICES=0: 단일 GPU.
# [한국어] heterogeneous_train_baseline.py: GIDS 미사용 버전 학습 스크립트.
# [한국어] --dataset_size full 중복 지정: 두 번 적혀 있어도 argparse가 뒤엣값으로 덮는다(동일 값이라 무해).
# [한국어] --path /mnt/raid0/: IGBH 원본 경로(mmap/읽기 대상).
# [한국어] --epochs 1, --batch_size 1024: 성능 비교용 단일 epoch / 동일 배치.
# [한국어] --data IGB / --model_type rsage / --num_layers 3 / --fan_out '10,5,5'
#          : 다른 run 스크립트와 동일해야 공정 비교가 된다.
# [한국어] --emb_size 1024: paper node feature 차원.
CUDA_VISIBLE_DEVICES=0  python heterogeneous_train_baseline.py --dataset_size full --path /mnt/raid0/   --dataset_size full --epochs 1  --batch_size 1024  --data IGB --model_type rsage --num_layers 3 --fan_out '10,5,5' --emb_size 1024

