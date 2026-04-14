"""
[한국어 설명] GIDS 파이썬 패키지 설치 스크립트 (setup.py)

=== 파일의 역할 ===
setuptools 기반으로 `GIDS` 파이썬 패키지를 빌드·설치하기 위한 설정 파일이다.
`pip install .` 또는 `python setup.py install` 실행 시 호출되어, GIDS 디렉토리
(즉 GIDS_Setup/GIDS/)를 찾아 site-packages로 배포한다. 또한 gids_module에서 별도로
빌드된 pybind11 확장 `BAM_Feature_Store.so` 를 함께 데이터 파일로 포함시켜,
설치 후 `import BAM_Feature_Store` 가 가능하도록 경로를 지정한다.
주: 현재 package_data 경로는 `/root/BAM_Tensor/...` 로 **하드코딩**되어 있어
일반 사용자 환경에서는 수정이 필요하며, 이는 원본 리포의 개발자 환경이 그대로
남은 흔적이다 (주석 작업 금지이므로 코드는 수정하지 않는다).

=== 전체 아키텍처에서의 위치 ===
호출 체인: 개발자 쉘 → `pip install .` → setuptools → 이 setup.py →
find_packages()로 GIDS/ 서브패키지 탐색 → site-packages/GIDS/ 설치 →
이후 학습 스크립트가 `from GIDS import GIDS, GIDS_DGLDataLoader` 로 로드.
실행 컨텍스트는 호스트 메인 Python 프로세스이며 설치 시점에만 수행된다.

=== 타 모듈과의 연결 ===
- `GIDS/__init__.py`, `GIDS/GIDS.py`: find_packages()에 의해 자동 수집되는 실제 구현.
- `gids_module/build/BAM_Feature_Store/BAM_Feature_Store.so`: CMake + setuptools로 별도
  빌드된 pybind 확장. package_data 에 경로가 하드코딩되어 있다.
- 런타임 의존성(torch, dgl, numpy, nvtx)은 별도 install_requires에 선언되어 있지
  않으므로 사용자가 사전 설치해 두어야 한다.

=== 주요 함수/구조체 요약 ===
- setup(...): setuptools 엔트리. name/version/packages/package_data 네 필드만 설정.
- find_packages(): 현재 디렉토리의 __init__.py 를 기준으로 하위 패키지(=GIDS) 자동 수집.
"""

# [한국어] setuptools에서 하위 패키지(디렉토리 + __init__.py)를 자동으로 수집하는 헬퍼를 임포트.
from setuptools import find_packages
# [한국어] 실제 패키지 메타데이터를 등록하는 setup() 함수를 임포트.
from setuptools import setup


# [한국어] GIDS 파이썬 패키지 배포 메타데이터를 setuptools 에 전달.
setup(
        # [한국어] 배포 시 `import GIDS` 가 가능하도록 하는 최상위 패키지 이름.
        name = "GIDS",
        # [한국어] 패키지 버전 문자열. pip 의존성 해석과 wheel 파일명에 포함된다.
        version     = '0.1.2',
        # [한국어] 현재 디렉토리 하위에서 __init__.py를 가진 모든 서브패키지(=GIDS/) 자동 탐색.
        packages=find_packages(),
        # [한국어] 루트 패키지('')에 추가 포함할 데이터 파일. 여기서는 사전 빌드된 pybind 확장의 .so 를
        #  절대경로로 지정해 설치 시 함께 배포하려고 한다. 경로가 개발자 환경에 고정되어 있으므로,
        #  다른 환경에서는 이 파일을 수정하거나 PYTHONPATH 로 .so 를 노출하는 방식을 병행해야 한다.
        package_data={
            '':['/root/BAM_Tensor/BAM_DataLoader/module/build/BAM_Feature_Store/BAM_Feature_Store.so'] }
)

