# [한국어] gids_module 빌드 설정 (CMakeLists.txt)
#
# === 파일의 역할 ===
# GIDS의 C++/CUDA 측 핵심 라이브러리 `BAM_Feature_Store.so` 를 빌드하는 CMake 스크립트.
# `gids_nvme.cu` (호스트 로직 + pybind11) 와 내부 include 된 `gids_kernel.cu`
# (GPU 디바이스 코드)를 단일 공유 객체로 엮고, 상위 `bam/` 서브모듈의 `libnvm.so`
# 를 링크한 뒤, pybind11 패키지 스켈레톤(`BAM_Feature_Store/__init__.py`, `setup.py`)
# 을 빌드 디렉토리로 CONFIGURE 해 파이썬 인스톨 단계로 넘긴다.
#
# === 전체 아키텍처에서의 위치 ===
# 빌드 파이프라인의 첫 단계. `bam/` 이 먼저 빌드되어 `libnvm.so` 를 내놓은 뒤,
# 이 CMake가 GIDS 고유 층을 컴파일한다. 출력물은 `GIDS_Setup/setup.py`(루트)의
# Python 설치 단계에서 소비된다. 실행 컨텍스트는 빌드 호스트 (CMake/Make/nvcc).
#
# === 타 모듈과의 연결 ===
# - 의존: pybind11, Python3 개발 헤더, `bam/include`(page_cache.h 등), `bam/build/lib/libnvm.so`.
# - 결과물은: (a) `BAM_Feature_Store.so` (pybind11 확장, GPU 코드 포함),
#   (b) 빌드 디렉토리에 복사되는 `BAM_Feature_Store/__init__.py` + `setup.py`.
# - 공유 자료구조: C++ 코드에서 BaM 의 `Controller`, `page_cache_t`, `range_t`,
#   `array_t` 가 링크/인클루드를 통해 연결된다.
#
# === 주요 함수/구조체 요약 ===
# - FIND_PACKAGE(pybind11/PythonInterp/PythonLibs): 파이썬 바인딩 빌드 전제 조건.
# - ADD_LIBRARY(BAM_Feature_Store SHARED gids_nvme.cu): 메인 타깃 정의.
# - target_compile_options(... CUDA SHELL -gencode ...): sm_70/sm_80 두 아키텍처 동시 빌드.
# - CONFIGURE_FILE(...): __init__.py/setup.py 의 `${PACKAGE_VERSION}` 등을 치환해 복사.

# [한국어] CMake 최소 버전 지정 — 3.3 이상이어야 CUDA를 1급 언어로 인식하고
# COMPILE_LANGUAGE 제너레이터 식을 사용할 수 있다.
CMAKE_MINIMUM_REQUIRED( VERSION 3.3.0 )
# [한국어] 프로젝트 선언 — CXX(C++)와 CUDA 두 언어를 활성화해 nvcc 툴체인이
# 자동 탐색되도록 한다. 프로젝트명은 템플릿 원본(cmake-cpp-pybind11) 그대로.
PROJECT( cmake-cpp-pybind11 CXX CUDA)

# [한국어] 호스트 C++ 표준을 C++11 로 고정 — BaM 헤더(page_cache.h 등) 가 C++11
# 기능(변수 템플릿, constexpr 등)을 사용하므로 반드시 필요.
SET( CMAKE_CXX_STANDARD 11 )
# [한국어] 위 표준이 만족되지 않으면 빌드 실패로 즉시 중단(자동 폴백 금지).
SET( CMAKE_CXX_STANDARD_REQUIRED ON )

# [한국어] CMake 모듈 탐색 경로 확장 — 프로젝트 로컬 cmake/, cmake/Modules/ 하위의
# FindXxx.cmake, 매크로 정의를 모두 INCLUDE() 로 불러올 수 있게 한다.
LIST( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake           # [한국어] 프로젝트 루트의 cmake 스크립트 디렉토리.
  ${CMAKE_SOURCE_DIR}/cmake/Modules   # [한국어] 추가 모듈(매크로/Find 스크립트) 보관소.
)

########################################################################
# Macros and additional flags
########################################################################

# [한국어] CheckCXXCompilerFlag — 특정 -f 옵션이 컴파일러에서 지원되는지 런타임
# 검사하는 표준 모듈. 매크로 스크립트들이 이를 사용해 플래그 조합을 결정한다.
INCLUDE( CheckCXXCompilerFlag )

# [한국어] cmake/Macros/*.cmake 를 모두 모아 차례로 INCLUDE — 공통 매크로
# (예: enable_warnings, set_rpath 등) 를 자동 로드한다.
FILE( GLOB filenames "cmake/Macros/*.cmake" )
# [한국어] 매 파일마다 CMake 스크립트로 평가(매크로/함수 정의만 수행, 부작용 없음 가정).
FOREACH( filename ${filenames} )
  # [한국어] INCLUDE: 해당 .cmake 파일을 현재 스코프에서 실행.
  INCLUDE( ${filename} )
ENDFOREACH()

########################################################################
# Library setup
########################################################################

# [한국어] 패키지 버전 문자열. setup.py의 `${PACKAGE_VERSION}` 자리에 치환된다.
SET( PACKAGE_VERSION "0.1.1" )

# [한국어] pybind11 CMake 헬퍼 탐색 — 없으면 빌드 즉시 실패(REQUIRED). include
# 경로 및 pybind11::module 타깃을 제공한다.
FIND_PACKAGE( pybind11 REQUIRED )

# Searches for the Python libraries and the Python interpreter with the
# desired version.
#
# Note that the current development version has replace these functions
# by `FindPython3`. It is unclear when this feature will be rolled out,
# though. See
#
#   https://cmake.org/cmake/help/git-master/module/FindPython3.html
#
# for more information.



# [한국어] Python 3 인터프리터 탐색 — setup.py 실행에 필요.
FIND_PACKAGE(PythonInterp 3 REQUIRED)
# [한국어] Python 3 개발 라이브러리(libpythonX.Y, Python.h) 탐색 — pybind11 확장이
# PyObject API 를 링크하기 위해 필요.
FIND_PACKAGE(PythonLibs   3 REQUIRED)


#IF( PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND PYBIND11_FOUND )
  # [한국어] 글로벌 include 경로에 Python/pybind11 헤더 주입 — 뒤에서 만드는
  # 모든 타깃이 PyObject, pybind11::class_ 등을 참조할 수 있게 한다.
  INCLUDE_DIRECTORIES(
    ${PYTHON_INCLUDE_DIRS}    # [한국어] Python.h 위치 (ex: /usr/include/python3.10).
    ${PYBIND11_INCLUDE_DIRS}  # [한국어] pybind11 템플릿 헤더 위치.
  )

# [한국어] 진행 확인용 빌드 로그 메시지 — 스크립트가 여기까지 왔는지 추적.
message("Compilation starts")

# [한국어] 메인 타깃: BAM_Feature_Store 공유 라이브러리. gids_nvme.cu 를 컴파일
# 단위로 지정하지만, 내부에서 `#include "gids_kernel.cu"` 로 GPU 커널도 함께 번들.
ADD_LIBRARY(BAM_Feature_Store SHARED gids_nvme.cu )

  # [한국어] include 탐색 경로: (1) 본 프로젝트 include/ (bam_nvme.h, example.h),
  # (2) bam/include (BaM의 Controller/page_cache/range/array 헤더),
  # (3) bam/include/freestanding — GPU device 측에서 쓰는 stdint/stddef 경량 버전.
  target_include_directories(BAM_Feature_Store PRIVATE ./include ../bam/include ../bam/include/freestanding/include/)
  # [한국어] BaM의 호스트 사용자 라이브러리(libnvm.so) 링크. 이 .so 는 /dev/libnvm*
  # 커널 모듈과 ioctl 로 통신하며 NVMe admin queue, GPU BAR 매핑을 담당.
  target_link_libraries(BAM_Feature_Store PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../bam/build/lib/libnvm.so)
  # The library must not have any prefix and should be located in
  # a subfolder that includes the package name. The setup will be
  # more complicated otherwise.
  # [한국어] 파이썬은 확장 모듈 이름이 정확히 "BAM_Feature_Store.so" 여야 import 가능.
  # PREFIX "" 로 기본 "lib" 접두사를 제거하고, 출력 경로를 패키지 폴더로 지정.
  SET_TARGET_PROPERTIES( BAM_Feature_Store
    PROPERTIES
      PREFIX ""                                                                        # [한국어] libBAM_... 금지, BAM_...로 출력.
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/BAM_Feature_Store"        # [한국어] setup.py가 참조할 위치.
  )

# [한국어] CUDA 컴파일 단위(.cu) 에만 적용되는 플래그 집합. 제너레이터 식
# `$<$<COMPILE_LANGUAGE:CUDA>:...>` 는 호스트 .cpp 에는 플래그가 새지 않게 한다.
target_compile_options(BAM_Feature_Store PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
			-std=c++11;                        # [한국어] 디바이스 코드도 C++11 (constexpr/템플릿 기본인자).
					-O3;                       # [한국어] 최대 최적화 (루프 풀기, inline). 디버그 심볼은 생략.

					#				-G;        # [한국어] (주석) 디바이스 디버그 심볼. 성능 저하로 기본 비활성.
					#		-g;                # [한국어] (주석) 호스트 디버그 심볼.
			--default-stream per-thread;       # [한국어] 각 호스트 스레드마다 별도 CUDA 기본 스트림.
			                                  #           read_feature_hetero 가 스트림 동시 발행 시 필수.
			#	--ptxas-options=-v;           # [한국어] (주석) PTX 어셈블러 레지스터/스택 사용량 리포트.
			#  	--compiler-options='-fPIC $(python3 -m pybind11 --includes)'
			                                  # [한국어] (주석) 과거 -fPIC 수동 주입. 현재는 SHARED 라이브러리에서 자동.
                      >)

# [한국어] Volta (sm_70, V100) 용 SASS 생성. arch=compute_70 은 PTX 레벨, code=sm_70 은 바이너리.
target_compile_options(BAM_Feature_Store PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_70,code=sm_70>"
)
# [한국어] Ampere (sm_80, A100) 용 SASS 생성 — 동일 .so 안에 fat binary로 2개 아키텍처 포함.
target_compile_options(BAM_Feature_Store PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_80,code=sm_80>"
)
  # Place the initialization file in the output directory for the Python
  # bindings. This will simplify the installation.
  # [한국어] __init__.py 를 빌드 디렉토리로 복사 — BAM_Feature_Store.so 옆에 두어
  # 파이썬 import 시 패키지로 인식되게 한다. CONFIGURE_FILE 은 `${VAR}` 를 치환한다.
  CONFIGURE_FILE( BAM_Feature_Store/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/BAM_Feature_Store/__init__.py
  )

  # Ditto for the setup file.
  # [한국어] setup.py 의 `${PACKAGE_VERSION}`, `${CMAKE_CURRENT_BINARY_DIR}` 를 치환하여
  # 배포 가능한 형태로 복사한다. 사용자는 이후 `python setup.py install` 실행.
  CONFIGURE_FILE( BAM_Feature_Store/setup.py
    ${CMAKE_CURRENT_BINARY_DIR}/BAM_Feature_Store/setup.py
  )

#ENDIF()
# [한국어] (참고) IF( PYTHONINTERP_FOUND AND ... ) ... ENDIF() 가드가 원본에선
# 주석 처리되어 있으므로 실제 빌드에서는 위 블록이 무조건 실행된다.

