# [한국어] BAM_Feature_Store 파이썬 패키지 초기화 모듈 (__init__.py)
#
# === 파일의 역할 ===
# CMake 빌드로 생성되는 공유 객체 `BAM_Feature_Store.so`를 동일 이름의 파이썬 패키지
# 네임스페이스로 재수출(re-export)한다. 사용자는 `import BAM_Feature_Store` 만으로
# pybind11로 노출된 `BAM_Feature_Store_float`, `BAM_Feature_Store_long`,
# `GIDS_Controllers` 클래스에 접근할 수 있다. 이 파일이 없으면 공유 객체가 패키지
# 일급 심볼로 올라오지 않아 상위 Python 래퍼(GIDS_Setup/GIDS/GIDS.py)가 임포트에
# 실패한다.
#
# === 전체 아키텍처에서의 위치 ===
# GIDS의 Python ↔ C++/CUDA 경계에서 가장 얇은 어댑터 층이다. 상위 모듈
# (GIDS_Setup/GIDS/GIDS.py)가 `from BAM_Feature_Store import BAM_Feature_Store_float`
# 형태로 바인딩 클래스를 들여올 때, 이 init이 있어야 패키지로 인식된다. 실행
# 컨텍스트는 호스트 유저스페이스(Python 인터프리터)이다.
#
# === 타 모듈과의 연결 ===
# 하위로는 pybind11 빌드 산출물(`BAM_Feature_Store.so`, 즉 `gids_nvme.cu` +
# `gids_kernel.cu`를 컴파일한 shared library)에 의존한다. 상위로는 GIDS Python
# 래퍼가 이 패키지를 통해 BaM Controller/페이지 캐시를 조작한다. 공유 자료구조로는
# C++ 쪽 `GIDS_Controllers`, `BAM_Feature_Store<float>`, `BAM_Feature_Store<int64_t>`가
# Python 객체로 그대로 노출된다.
#
# === 주요 함수/구조체 요약 ===
# - 본 파일은 실행 가능한 함수를 정의하지 않고, 단일 `from .* import *` 재수출만 수행.
# - 노출되는 심볼: `BAM_Feature_Store_float`, `BAM_Feature_Store_long`,
#   `GIDS_Controllers` (모두 pybind11 `py::class_` 로 선언됨, gids_nvme.cu 하단 참조).

# [한국어] 동일 패키지 내부의 확장 모듈(BAM_Feature_Store.so)에서 pybind11가 노출한
# 모든 심볼을 현재 네임스페이스로 끌어올린다.
# - "from .BAM_Feature_Store": 상대 임포트로, 동일 패키지 디렉토리의 공유 객체를 지정.
# - "import *": PYBIND11_MODULE(BAM_Feature_Store, m) 에서 m.def/py::class_로 등록한
#   BAM_Feature_Store_float/_long, GIDS_Controllers 3개 클래스를 패키지 최상위로 승격.
from .BAM_Feature_Store import *

