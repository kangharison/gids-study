"""
[한국어 설명] GIDS 패키지 개발용 스모크 테스트 스크립트 (test.py)

=== 파일의 역할 ===
이 파일은 GIDS/BAM 파이썬 확장이 정상적으로 import되고 간단한 GPU 텐서를
인자로 넘겼을 때 `fetch_feature` 경로가 충돌 없이 동작하는지 확인하는
개발자용 스모크 테스트이다. 정식 단위 테스트(evaluation/GIDS_unit_test.py)나
학습 스크립트와 달리 매우 짧고, 의존성(BAM_Util, BAM_Feature_Store, torch, CUDA)이
모두 빌드·설치되어 있는지 1분 안에 확인하기 위한 최소 실행 예시다.
또한 `BAM_Util` 이라는 별도 pybind 유틸 모듈의 존재 여부도 검증하는 역할을 겸한다.
주의: 이 스크립트는 실제 SSD에 올바른 feature가 사전 기록되어 있지 않으면
무의미한 값을 반환할 수 있으며, 정식 회귀 테스트가 아니다.

=== 전체 아키텍처에서의 위치 ===
GIDS 패키지 설치 직후 호출되는 체크용 엔트리 포인트이다.
호출 체인: (개발자 셸) `python test.py` → BAM_Util 클래스 인스턴스화 →
BAM_Util.fetch_feature() → pybind11 → BAM_Feature_Store (또는 BAM_Util) C++ 백엔드 →
BaM page cache → NVMe SSD. GIDS 클래스(GIDS.py)와는 독립적으로 BAM_Util 경로를 탄다.

=== 타 모듈과의 연결 ===
- `BAM_Util`: gids_module 쪽에서 별도로 빌드되는 pybind 확장. 모듈 자체와 동명의
  클래스 `BAM_Util`이 함께 존재한다(아래 두 import 라인이 각각을 가져옴).
- `BAM_Feature_Store`: GIDS 메인 경로가 사용하는 feature store pybind 확장. 이 스크립트는
  실제로는 사용하지 않지만, import 성공 여부를 확인해 두는 의미가 있다.
- `torch`: 테스트 입력을 CUDA 텐서로 만들기 위해 필요. CUDA 디바이스 'cuda:0'을 가정.

=== 주요 함수/구조체 요약 ===
스크립트 전체가 탑레벨 실행문이며 함수 정의는 없다.
- BAM_Util(100, 100): 100/100이라는 임의 초기화 파라미터(추정: num_ele, some_dim 등)
  로 BAM_Util 객체를 생성.
- test.fetch_feature(index, 100): 인덱스 텐서 [0,1,2]의 각 노드에 대해 dim=100 크기
  feature를 GPU 메모리로 읽어와 반환.
"""

# [한국어] pybind11 확장 모듈 `BAM_Util` 을 네임스페이스 레벨에서 import.
#  - 이 import가 실패하면 gids_module/BAM_Util 빌드/설치가 누락된 상태임을 즉시 알 수 있다.
import BAM_Util
# [한국어] 모듈 안의 동명 클래스 `BAM_Util` 을 별도 심볼로 꺼내 쓰기 편하게 가져온다.
#  - pybind11 확장에서는 모듈명과 메인 클래스명이 같은 경우가 흔하므로 이 이중 import 패턴을 사용.
from BAM_Util import BAM_Util

# [한국어] GIDS 메인 경로에서 사용하는 feature store 확장도 import하여 로드 가능한지 확인.
#  - 실제로 이 스크립트 로직에서 쓰이지는 않지만, 설치 검증 목적으로 남겨져 있다.
import BAM_Feature_Store
# [한국어] 테스트용 인덱스 텐서를 만들고 GPU로 옮기기 위해 PyTorch를 임포트.
import torch

# [한국어] BAM_Util 인스턴스 생성. 두 정수 인자의 의미는 C++ 바인딩 정의를 따르며,
#  통상 (num_elements, dim) 또는 (cache_size, dim) 형태로 해석된다 — 스모크 테스트이므로 상징적 값.
test = BAM_Util(100,100)

# [한국어] 테스트용 노드 ID 리스트 [0,1,2] 를 int64 텐서로 생성한다.
#  - BAM_Util/BAM_Feature_Store 의 인덱스 타입은 long(int64)이므로 dtype=torch.long 을 반드시 지정해야
#    C++ 측에서 잘못된 해석을 하지 않는다.
c_index = torch.tensor([0,1,2], dtype=torch.long)
# [한국어] 인덱스 텐서를 기본 CUDA 디바이스로 복사. BaM/GIDS 는 GPU-resident 인덱스만 허용하기 때문.
index = c_index.to('cuda:0')
# [한국어] fetch_feature(인덱스 텐서, feature 차원). 내부에서 GPU 커널을 런칭하여 SSD로부터 3개 노드의
#  dim=100 feature 를 읽어 CUDA 텐서로 반환.
test_gpu = test.fetch_feature(index, 100)

# [한국어] 반환된 GPU 텐서를 print 해 값이 올바르게 채워졌는지(또는 shape가 [3,100]인지) 육안 확인.
print(test_gpu)
