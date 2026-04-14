# [한국어] BAM_Feature_Store pybind11 패키지용 setup.py (setup.py)
#
# === 파일의 역할 ===
# CMake 빌드가 생성한 공유 객체(BAM_Feature_Store.so) 를 Python 패키지로
# 설치하기 위한 distutils 설정 스크립트. `${PACKAGE_VERSION}`, `${CMAKE_CURRENT_BINARY_DIR}`
# 는 CMake `CONFIGURE_FILE()` 에 의해 빌드 시 치환되어 최종적으로
# `gids_module/build/BAM_Feature_Store/setup.py` 로 배치된다. 사용자는 해당
# 빌드 디렉토리에서 `python setup.py install` 을 실행해 site-packages 에 설치한다.
#
# === 전체 아키텍처에서의 위치 ===
# GIDS 빌드 파이프라인의 마지막 직전 단계로, C++/CUDA 바이너리를 Python 측
# (`GIDS_Setup/GIDS/GIDS.py`)에서 임포트 가능하도록 포장한다. 실행 컨텍스트는
# 호스트 유저스페이스 Python(빌드 시점).
#
# === 타 모듈과의 연결 ===
# 하위로는 CMake(`CMakeLists.txt`)가 배출한 `BAM_Feature_Store/__init__.py`와
# `BAM_Feature_Store.so`에 의존. 상위로는 `GIDS_Setup/setup.py`(루트)가 이 패키지를
# 전제로 사용자 GIDS Python API 를 설치한다.
#
# === 주요 함수/구조체 요약 ===
# - `setup(...)` 호출 하나만 수행. name/version/packages/package_dir/package_data 를 지정.
# - 함수 정의 없음. 파이썬 버전 가드 if 한 개만 포함.

# [한국어] distutils 의 최상위 setup 진입점을 가져옴. pip가 아직 표준이 아니던
# 시절의 레거시 빌드 인터페이스지만 단순 C 확장 패키징에는 충분하다.
from distutils.core import setup

# [한국어] Python 2/3 분기를 차단하기 위한 sys 모듈 임포트.
import sys
# [한국어] Python 3 미만이면 즉시 종료 — pybind11/CUDA 바인딩이 Py3만 지원.
if sys.version_info < (3,0):
  # [한국어] sys.exit(str): 0이 아닌 종료 코드(=1) + 메시지 출력, 설치 중단.
  sys.exit('Sorry, Python < 3.0 is not supported')

# [한국어] distutils.setup 호출 — 아래 키워드로 패키지 메타데이터와 파일 배치를 지정.
setup(
  # [한국어] PyPI 상의 배포 이름(실제로 업로드하지는 않음). pybind11 템플릿 원본명을 유지.
  name        = 'cmake_cpp_pybind11',
  # [한국어] CMake CONFIGURE_FILE 이 `${PACKAGE_VERSION}` 을 CMakeLists의 0.1.1 로 치환.
  version     = '${PACKAGE_VERSION}', # TODO: might want to use commit ID here
  # [한국어] 설치할 Python 패키지 이름 — 이 한 줄이 site-packages/BAM_Feature_Store/ 생성.
  packages    = [ 'BAM_Feature_Store' ],
  # [한국어] package_dir의 '': 루트 패키지 탐색 경로. CMake가 빌드 바이너리 디렉토리로 치환.
  package_dir = {
    '': '${CMAKE_CURRENT_BINARY_DIR}'
  },
  # [한국어] 빈 키 '' 는 "모든 패키지" 를 의미. .so 를 package data로 함께 포함시켜
  # 설치 시 __init__.py 옆에 공유 객체가 복사되도록 강제한다.
  package_data = {
    '': ['BAM_Feature_Store.so']
  }
)
