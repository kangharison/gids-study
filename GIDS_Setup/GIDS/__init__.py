"""
[한국어 설명] GIDS Python 패키지 엔트리 포인트 (__init__.py)

=== 파일의 역할 ===
이 파일은 `GIDS` Python 패키지의 초기화 모듈로서, 사용자가 `import GIDS` 또는
`from GIDS import GIDS, GIDS_DGLDataLoader`와 같이 패키지를 import할 때 실제로
노출되는 심볼(이름)을 결정한다. 내부 구현 모듈인 `GIDS/GIDS.py`에서 정의된
`GIDS` 클래스(메인 API)와 `GIDS_DGLDataLoader` 클래스(DGL DataLoader 래퍼)를
패키지 최상위 네임스페이스로 끌어올려, 외부 학습 스크립트(evaluation/*.py)가
내부 모듈 경로를 몰라도 두 핵심 클래스를 바로 사용할 수 있게 한다.

=== 전체 아키텍처에서의 위치 ===
GIDS 프로젝트의 호출 체인에서 가장 바깥쪽 Python 진입점이다.
evaluation/homogenous_train.py 등 학습 스크립트 → `from GIDS.GIDS import GIDS,
GIDS_DGLDataLoader` 또는 이 __init__.py를 통한 간접 import → GIDS 클래스 인스턴스 →
pybind11로 래핑된 `BAM_Feature_Store` C++ 모듈(gids_module/gids_nvme.cu) →
BaM 런타임(bam/) → NVMe SSD 순으로 제어가 전파된다.
실행 컨텍스트는 호스트 메인 Python 프로세스이며, import 시 단 1회만 수행된다.

=== 타 모듈과의 연결 ===
이 파일은 같은 디렉토리의 `GIDS.py`만을 상대 import한다. GIDS.py가 간접적으로
`BAM_Feature_Store`(pybind 확장), `dgl`, `torch`, `nvtx`, `ctypes`에 의존하므로
이 파일이 로드되는 순간 해당 의존성들도 전부 초기화된다.
setup.py(프로젝트 루트)가 `find_packages()`로 이 디렉토리를 감지해 배포 패키지에
포함시키며, pip install 이후 site-packages/GIDS/__init__.py 로 설치된다.

=== 주요 함수/구조체 요약 ===
함수/클래스 정의는 없고, 다음 두 심볼을 re-export 한다:
- GIDS: GIDS 메인 API 클래스. Controller/페이지 캐시/CPU 버퍼/윈도우 버퍼링 관리.
- GIDS_DGLDataLoader: torch.utils.data.DataLoader 서브클래스. DGL 샘플러와
  결합하여 미니배치 단위로 GPU에서 직접 SSD feature를 읽어오는 반복자를 제공.
"""

# [한국어] 같은 패키지 내 GIDS 서브모듈(GIDS.py)에서 메인 API 클래스 GIDS 를 임포트한다.
#  - 상대 import(`.GIDS`)를 쓰는 이유: 설치 후 site-packages에 배치되어도 경로가
#    `GIDS.GIDS.GIDS` 로 안정적으로 해석되도록 하기 위함.
#  - 이 import가 트리거되면 GIDS.py 상단의 torch/dgl/BAM_Feature_Store 로딩이 수반된다.
from .GIDS import GIDS
# [한국어] 동일 서브모듈에서 DGL DataLoader 래퍼 클래스 GIDS_DGLDataLoader 를 함께 노출한다.
#  - 학습 스크립트는 보통 `from GIDS import GIDS, GIDS_DGLDataLoader` 한 줄로 두 클래스를
#    동시에 가져오므로, __init__.py 레벨에서 양쪽 모두 재노출해 두는 것이 관례다.
from .GIDS import GIDS_DGLDataLoader

