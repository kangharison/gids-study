# ============================================================================
# [한국어] GIDS 전체 최적화 활성화로 IGBH 학습 실행 (run_GIDS_IGBH.sh)
#
# === 파일의 역할 ===
# VLDB'24 GIDS 논문의 **완전체 설정**(BaM + Window Buffering + Constant CPU
# Buffer + Storage Access Accumulator)을 한 번에 켜고 IGBH full을 학습하는
# 스크립트. run_BaM_IGBH.sh와 같은 하이퍼파라미터에 GIDS 고유 최적화를 추가해
# 최적화의 개별 기여도를 A/B로 비교할 수 있게 한다.
#
# === 전체 아키텍처에서의 위치 ===
# heterogeneous_train.py → GIDS.GIDS(cpu_buffer=True, window_buffer=True, …)
#   → gids_module/gids_nvme.cu: read_feature_with_cpu_backing / set_window_buffering
#   → gids_module/gids_kernel.cu: read_feature_kernel_with_cpu_backing /
#     set_window_buffering_kernel → bam page_cache/array.
#
# === 타 모듈과의 연결 ===
# - --cpu_buffer → GIDS_CPU_buffer<float>::cpu_buffer_init (핫 노드 pin 영역 할당).
# - --pin_file(기본값) → page_rank_node_list_gen.py가 만든 상위 PageRank 노드 ID
#   텐서를 읽어 range_t::set_cpu_buffer 로 매핑.
# - --window_buffer → set_window_buffering_kernel, 다음 wb_size 배치를 프리패치.
# - --accumulator → Storage Access Accumulator (IO batching / reordering).
#
# === 주요 함수/구조체 요약 ===
# 셸 한 줄. 추가 플래그 의미:
#   --window_buffer --wb_size 8       : 8개 배치 lookahead 프리패치.
#   --cpu_buffer --cpu_buffer_percent 0.2: 전체 노드의 20%를 CPU 버퍼에 pin.
#   --accumulator                      : storage access accumulator on.
# ============================================================================
# [한국어] CUDA_VISIBLE_DEVICES=0: 단일 GPU.
# [한국어] heterogeneous_train.py: GIDS 지원 학습 스크립트.
# [한국어] --dataset_size full --path /mnt/raid0/: IGBH full 데이터셋 루트.
# [한국어] --epochs 1 / --log_every 1000 / --uva_graph 1 / --GIDS / --batch_size 1024
#          : run_BaM_IGBH.sh 와 동일 공통 플래그.
# [한국어] --data IGB / --model_type rsage / --num_layers 3 / --fan_out '10,5,5'
#          : 동일 모델·샘플러 구성.
# [한국어] $((4*1024)) --num_ssd 1 $((550*1000*1000*1024)) --page_size 4096 --emb_size 1024
#          : BaM page cache 4GB / 단일 SSD / 전체 element 수 / 4KB 페이지 / 1024 feat dim.
# [한국어] --window_buffer --wb_size 8: Window Buffering 활성, 향후 8 배치 프리패치.
# [한국어] --cpu_buffer --cpu_buffer_percent 0.2: Constant CPU Buffer 활성, 상위 20%
#          PageRank 노드 feature를 pinned host memory에 고정.
# [한국어] --accumulator: Storage Access Accumulator 활성 (SSD bw/latency 추정 기반 batching).
CUDA_VISIBLE_DEVICES=0  python heterogeneous_train.py --dataset_size full --path /mnt/raid0/   --dataset_size full --epochs 1 --log_every 1000 --uva_graph 1 --GIDS --batch_size 1024  --data IGB --model_type rsage --num_layers 3 --fan_out '10,5,5' --cache_size $((4*1024)) --num_ssd 1   --num_ele $((550*1000*1000*1024)) --page_size 4096 --emb_size 1024 --window_buffer --wb_size 8   --cpu_buffer --cpu_buffer_percent 0.2   --accumulator


