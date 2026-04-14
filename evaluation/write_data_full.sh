# ============================================================================
# [한국어] 멀티 SSD 스트라이핑으로 IGBH full feature SSD 기록 스크립트
#          (write_data_full.sh)
#
# === 파일의 역할 ===
# IGBH full 전체(paper/author/fos/institute/journal/conference) node_feat.npy를
# **2개 NVMe Controller 스트라이핑**(--n_ctrls 2)으로 기록한다. GIDS 학습에서
# range_t를 STRIPE 모드로 만들었을 때와 일관된 레이아웃을 만든다.
#
# === 전체 아키텍처에서의 위치 ===
# 학습 전 1회성 준비. 이후 heterogeneous_train.py가 heterograph_map(ntype→offset)
# 으로 동일 logical offset을 읽어 feature fetch한다. write_data.sh와 달리
# --ioffset 으로 Controller 간 페이지 스트라이핑 시작점을 조정.
#
# === 타 모듈과의 연결 ===
# - bam/build/bin/nvm-readwrite_stripe-bench (bam 서브모듈).
# - GIDS_Setup/GIDS/GIDS.py 의 heterograph_map offsets.
#
# === 주요 함수/구조체 요약 ===
# 추가 플래그:
#   --n_ctrls 2 : 2개 Controller로 페이지 단위 스트라이핑.
#   --pages N   : 한 번에 수행할 페이지 수 (1M pages = 4GB batch).
#   --ioffset   : 첫 Controller 내부 페이지 오프셋(스트라이핑 조정).
#   --loffset   : logical byte offset(= 누적 노드수 × 4096).
# ============================================================================

# [한국어] paper feature. 최초 node type이므로 loffset=0 (--loffset 미지정).
#          --ioffset 0: Controller0 페이지 0부터 기록 시작.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/paper/node_feat.npy  --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2  --ioffset 0
# [한국어] author feature. loffset = paper 노드 수(269,346,174) × page_size(4096).
#          paper 영역 직후 LBA부터 이어서 기록.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/author/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --ioffset 0 --loffset $((269346174*4096))


# [한국어] fos feature. --ioffset 128: 스트라이프 주기 내에서 다른 Controller로
#          진입하도록 오프셋 조정. 이하 소형 node type 들은 같은 패턴.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/fos/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --loffset $((546567057*4096))  --ioffset 128
# [한국어] institute feature. loffset = paper+author+fos 누적 × 4096.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/institute/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --loffset $((547280017 * 4096))  --ioffset 128

# [한국어] journal feature. 누적 오프셋 계산 결과 바이트를 loffset으로.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/journal/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --loffset $((546593975*4096))  --ioffset 128
# [한국어] conference feature. 마지막 node type.
../bam/build/bin/nvm-readwrite_stripe-bench --input /mnt/raid0/full/processed/conference/node_feat.npy --queue_depth 1024 --access_type 1 --num_queues 128 --threads 102400 --pages $((1024*1024)) --n_ctrls 2 --loffset $((546643027 * 4096))  --ioffset 128


