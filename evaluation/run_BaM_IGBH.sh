# ============================================================================
# [한국어] BaM(순정) 모드로 IGBH(이종 그래프) 학습 실행 스크립트 (run_BaM_IGBH.sh)
#
# === 파일의 역할 ===
# GIDS 저장소에서 "BaM baseline" 실험 구성을 1커맨드로 재현한다.
# GIDS 고유 최적화(Constant CPU Buffer, Window Buffering, Accumulator)를 끈 채
# 순수 BaM GPU-direct NVMe 읽기만으로 IGBH(IGB Heterogeneous) full 데이터셋을
# heterogeneous_train.py로 학습한다. 논문의 BaM vs GIDS 비교 기준선이다.
#
# === 전체 아키텍처에서의 위치 ===
# evaluation/heterogeneous_train.py → GIDS_Setup/GIDS/GIDS.py (CPU 버퍼/WB 플래그
# off 상태) → gids_module/gids_nvme.cu (BAM_Feature_Store) → bam (page_cache_t,
# array_t) → /dev/libnvm0. GIDS 옵션을 제외한 공통 경로 검증용.
#
# === 타 모듈과의 연결 ===
# - heterogeneous_train.py가 GIDS_DGLDataLoader를 씀 (Python 측 래퍼).
# - --cache_size, --num_ssd, --num_ele, --page_size는 BAM page_cache_t /
#   Controller / range_t 생성 시 전달된다.
# - run_GIDS_IGBH.sh 와 동일 파라미터지만 --window_buffer, --cpu_buffer,
#   --accumulator 가 빠져 있어 BaM 단독 경로(read_feature_kernel) 로만 동작.
#
# === 주요 함수/구조체 요약 ===
# 셸 한 줄 실행. 주요 플래그:
#   --uva_graph 1 : 그래프 구조(edge_index)는 UVA(pinned host)로 매핑.
#                   feature fetch만 BaM 경유 SSD.
#   --GIDS        : GIDS_DGLDataLoader 활성화 (단, 최적화 플래그 없음=BaM only).
#   --fan_out     : DGL NeighborSampler의 layer별 샘플 수 (root→... 순).
#                   '10,5,5' = 3 layer, root에서 10개, 다음 5, 다음 5 이웃.
#   --cache_size  : BaM GPU page cache 크기 (MB). 4*1024 = 4GB.
#   --page_size   : NVMe 읽기 단위 (bytes). 4096.
#   --emb_size    : feature dimension (IGBH paper feat = 1024).
# ============================================================================
# [한국어] CUDA_VISIBLE_DEVICES=0: 단일 GPU 사용.
# [한국어] heterogeneous_train.py 실행. 아래 모든 --플래그가 argparse로 파싱된다.
# [한국어] --dataset_size full: IGB 전체(paper 269M개) 데이터셋.
# [한국어] --path /mnt/raid0/ : IGBH 원본 파일이 올려진 루트 경로.
# [한국어] --epochs 1        : 1 epoch만 돌려 throughput 측정.
# [한국어] --log_every 1000  : 1000 iteration마다 로깅.
# [한국어] --uva_graph 1     : 그래프 구조를 UVA(pinned)로 GPU에 노출.
# [한국어] --GIDS            : GIDS loader 경로 활성화 (CPU buf/WB는 꺼둠).
# [한국어] --batch_size 1024 : 배치당 seed node 수.
# [한국어] --data IGB        : IGB 계열 dataset class 선택.
# [한국어] --model_type rsage: RSAGE (Relational GraphSAGE) heterograph 모델.
# [한국어] --num_layers 3    : GNN 3층 (fan_out 길이와 일치해야 함).
# [한국어] --fan_out '10,5,5': NeighborSampler layer별 이웃 개수.
# [한국어] $((4*1024))       : bash 산술 확장. cache_size = 4096 MB (4GB).
# [한국어] --num_ssd 1       : BaM Controller 하나 (스트라이핑 없음).
# [한국어] $((550*1000*1000*1024)): 총 feature element 개수.
# [한국어] --page_size 4096  : NVMe LBA/페이지 읽기 단위.
# [한국어] --emb_size 1024   : paper feature 차원.
CUDA_VISIBLE_DEVICES=0  python heterogeneous_train.py --dataset_size full --path /mnt/raid0/   --dataset_size full --epochs 1 --log_every 1000 --uva_graph 1 --GIDS --batch_size 1024  --data IGB --model_type rsage --num_layers 3 --fan_out '10,5,5' --cache_size $((4*1024)) --num_ssd 1   --num_ele $((550*1000*1000*1024)) --page_size 4096 --emb_size 1024

