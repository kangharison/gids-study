# ============================================================================
# [한국어] GIDS 단위 테스트 실행 스크립트 (gids_unit_test.sh)
#
# === 파일의 역할 ===
# GIDS 파이썬 파사드(GIDS_Setup/GIDS/GIDS.py)가 pybind11을 통해 C++/CUDA 쪽
# BAM_Feature_Store 모듈과 정상적으로 연결되고, 기본 fetch 경로가 동작하는지를
# 최소 파라미터 조합으로 검증한다. evaluation/GIDS_unit_test.py를 단일 GPU에서
# 실행하는 래퍼 스크립트다.
#
# === 전체 아키텍처에서의 위치 ===
# 실제 학습(run_GIDS_IGBH.sh 등)을 돌리기 전, 파이썬 ↔ gids_module(C++/CUDA) ↔
# bam 서브모듈 ↔ /dev/libnvm* 경로가 살아있는지 smoke test 하는 단계다.
# fetch 실패 시 상위 학습 스크립트가 아니라 이 경로부터 디버그한다.
#
# === 타 모듈과의 연결 ===
# - 호출 대상: evaluation/GIDS_unit_test.py (→ GIDS.GIDS → BAM_Feature_Store_*).
# - 의존: bam 커널 모듈 적재(/dev/libnvm0 존재), gids_module 빌드 완료.
# - 환경 변수 CUDA_VISIBLE_DEVICES로 사용 GPU를 고정한다.
#
# === 주요 함수/구조체 요약 ===
# 셸 한 줄 실행이라 함수는 없다. 파라미터 의미:
#   --cache_size N  : BaM page_cache_t의 MB 단위 크기 (여기선 8*1024=8GB).
#   --num_ssd 1     : 사용할 NVMe 디바이스 수 (striping 없음).
#   --num_ele N     : 데이터셋 원소 총 개수 = Total Size / sizeof(float).
#                     550*1000*1000*1024 ≈ 550B elems → 약 2.2TB feature 공간.
# ============================================================================
# [한국어] CUDA_VISIBLE_DEVICES=0  : 호스트에서 보이는 첫 번째 GPU만 사용해
#          테스트를 단순화. 멀티 GPU 환경에서 실수 방지용.
# [한국어] python GIDS_unit_test.py: evaluation/GIDS_unit_test.py 실행.
#          내부에서 GIDS.GIDS(...)를 생성하고 fetch_test/fetch_hetero_test로
#          BAM_Feature_Store_{float,long}의 pybind 경로를 건드린다.
# [한국어] $((8*1024))            : bash 산술 확장. cache_size MB 단위.
# [한국어] $((550*1000*1000*1024)): 총 element 수 계산 (float 기준 약 2.2TB).
CUDA_VISIBLE_DEVICES=0  python GIDS_unit_test.py  --cache_size $((8*1024)) --num_ssd 1   --num_ele $((550*1000*1000*1024))
