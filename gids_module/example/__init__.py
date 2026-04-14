# [한국어] example 파이썬 패키지 초기화 모듈 (__init__.py)
#
# === 파일의 역할 ===
# cmake-cpp-pybind11 템플릿이 남긴 예제 패키지의 초기화 파일. 빌드 시스템이
# pybind11 확장(example.so)을 이 디렉토리 안에 놓고, 사용자는 `import example`
# 로 `Example` 클래스를 사용할 수 있도록 심볼을 재수출한다. 실제 GIDS 기능과는
# 무관한 데모/Skeleton 컴포넌트이지만, CMake 템플릿 호환성을 위해 유지된다.
#
# === 전체 아키텍처에서의 위치 ===
# GIDS 런타임 경로에는 포함되지 않는다. 학습/데이터 로딩 흐름은
# BAM_Feature_Store 패키지만 사용한다. 실행 컨텍스트는 호스트 유저스페이스 Python.
#
# === 타 모듈과의 연결 ===
# 하위로는 pybind11 샘플 확장 `example.so` (include/example.h 의 Example 클래스
# 바인딩)에 의존한다. 상위 의존 모듈은 없다.
#
# === 주요 함수/구조체 요약 ===
# - 함수 없음. `from .example import *` 재수출 한 줄로 Example 클래스(및 그
#   operator+=, print 바인딩)를 패키지 루트 네임스페이스에 노출.

# [한국어] 동일 패키지의 example.so(pybind11 확장)에서 선언된 심볼을 모두 끌어옴.
# - 상대 import `.example`: 같은 폴더의 shared object를 뜻함(파이썬은 .so도 모듈로 인식).
# - `import *`: Example 클래스를 네임스페이스 최상위로 승격시켜 `example.Example(1.0)`
#   같은 호출을 가능하게 한다.
from .example import *

