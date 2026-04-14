# [한국어] example pybind11 데모 패키지용 setup.py (setup.py)
#
# === 파일의 역할 ===
# cmake-cpp-pybind11 템플릿이 제공한 예제용 설치 스크립트. `example.so`
# (include/example.h 의 Example 클래스 바인딩) 를 Python 패키지로 설치한다.
# GIDS 런타임 경로에는 쓰이지 않는 데모 자산이지만, 원본 템플릿 구조 호환성을
# 위해 유지된다.
#
# === 전체 아키텍처에서의 위치 ===
# GIDS 학습/데이터 로딩 흐름과 무관. Python 인터프리터에 샘플 확장 하나를
# 올려보기 위한 보조 스크립트. 실행 컨텍스트는 빌드 시 호스트 유저스페이스 Python.
#
# === 타 모듈과의 연결 ===
# 하위로 `example/__init__.py`, `example.so` (CMake가 생성) 에 의존. 상위
# 의존성 없음.
#
# === 주요 함수/구조체 요약 ===
# - setup() 한 번 호출, 파이썬 버전 가드 if 한 개.

# [한국어] distutils 의 setup 함수 — 최소한의 Python 패키지 메타데이터 선언용.
from distutils.core import setup

# [한국어] 인터프리터 버전 검사를 위해 sys 모듈 로드.
import sys
# [한국어] Python 2 지원 차단(코드 자체는 무해하나 빌드 정책상 Py3 only).
if sys.version_info < (3,0):
  # [한국어] 메시지 출력 후 종료 코드 1 로 설치 중단.
  sys.exit('Sorry, Python < 3.0 is not supported')

# [한국어] distutils.setup — example 패키지 설치 레시피.
setup(
  # [한국어] 템플릿 기본 배포명 (변경 없음).
  name        = 'cmake_cpp_pybind11',
  # [한국어] `${PACKAGE_VERSION}` 은 CMake CONFIGURE_FILE 이 0.1.1 로 치환.
  version     = '${PACKAGE_VERSION}', # TODO: might want to use commit ID here
  # [한국어] 설치할 패키지 명: site-packages/example/ 디렉토리가 생성된다.
  packages    = [ 'example' ],
  # [한국어] 루트 패키지 탐색 경로를 CMake 빌드 디렉토리로 지정(치환됨).
  package_dir = {
    '': '${CMAKE_CURRENT_BINARY_DIR}'
  },
  # [한국어] example.so 공유 객체를 패키지와 함께 설치되도록 데이터 파일로 등록.
  package_data = {
    '': ['example.so']
  }
)
