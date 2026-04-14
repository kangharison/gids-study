# ============================================================================
# [한국어] 단일-SSD feature 데이터 SSD 기록 스크립트 (write_data.sh)
#
# === 파일의 역할 ===
# IGB/IGBH 각 node type의 node_feat.npy (numpy float32) 파일을 BaM 벤치마크
# 바이너리(nvm-readwrite_stripe-bench)로 **단일 NVMe SSD**에 로우-레벨 기록한다.
# GIDS 학습은 SSD 블록 어드레싱으로 feature를 직접 읽으므로, 학습 전에 이 단계
# 로 각 feature tensor를 지정 LBA 오프셋에 stripe 없이 써둬야 한다.
#
# === 전체 아키텍처에서의 위치 ===
# (전처리 단계) nvme-readwrite_stripe-bench  → libnvm kernel module → /dev/libnvm0
# → NVMe SSD. 이후 evaluation 학습 스크립트가 같은 LBA 레이아웃을 전제로 읽는다.
# write_data_full.sh는 멀티-SSD striping (--n_ctrls 2) 버전.
#
# === 타 모듈과의 연결 ===
# - 사용 바이너리: bam/build/bin/nvm-readwrite_stripe-bench (BaM 서브모듈).
# - heterograph_map (GIDS.py) 의 node type별 logical offset과 --loffset 값이
#   1:1 대응해야 학습 시 feature가 올바르게 매핑된다.
#   예) author_offset = 269346174 × 4096(byte page) = paper 269M 노드 이후 주소.
#
# === 주요 함수/구조체 요약 ===
# 셸. 공통 플래그:
#   --access_type 1   : write 모드.
#   --queue_depth 1024: SQ 깊이.
#   --num_queues 128  : I/O queue 개수.
#   --threads 102400  : 동시 실행 GPU 스레드 수.
#   --n_ctrls 1       : Controller 1개 (striping 없음).
#   --loffset N       : 시작 LBA byte offset (=노드 누적수 × page_size).
# ============================================================================

# [한국어] paper node_feat를 /mnt/nvme17의 파일에서 읽어 SSD 0번지부터 기록.
#          paper는 node-type 순서상 가장 처음이므로 --loffset 지정 없음.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/nvme17/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1

# [한국어] author feature 기록. --loffset = paper 노드 수(269,346,174) × page_size(4096).
#          즉 paper feature 영역 바로 다음 LBA에 author feature를 이어서 쓴다.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/author/node_feat.npy  --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --loffset $((269346174*4096))

# [한국어] fos(field of study) feature. loffset = (paper+author) 누적 노드수 × 4096.
#          heterograph_map에서 fos 시작 오프셋과 같아야 한다.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/fos/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --loffset $((546567057*4096))


# [한국어] institute feature. loffset = (paper+author+fos) 누적 × 4096.
#          마지막 node type이라 이후 loffset 지정 없음.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/institute/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --n_ctrls 1 --loffset $((547280017 * 4096))


