"""
[한국어] NumPy feature 텐서를 GIDS 경유로 SSD 에 기록하는 유틸 (tensor_write.py)

=== 파일의 역할 ===
사전 생성된 numpy .npy feature 파일을 PyTorch 텐서로 로드한 뒤, GIDS 파이썬
객체의 `store_tensor()` 를 통해 NVMe SSD 에 직접 기록한다. 학습 전(run_*.sh)에
한 번 실행해 GPU 가 읽을 LBA 레이아웃을 준비하는 **전처리용 스크립트**다.
내부적으로 pybind11 → BAM_Feature_Store::store_tensor → write_feature_kernel
경로가 호출된다.

=== 전체 아키텍처에서의 위치 ===
(전처리) tensor_write.py → GIDS.GIDS.store_tensor() → gids_module/gids_nvme.cu
→ gids_kernel.cu: write_feature_kernel{,2} → bam page_cache/Controller →
/dev/libnvm* → NVMe SSD. write_data*.sh 가 bam 의 벤치 바이너리를 쓴 것과 달리,
이 스크립트는 GIDS API 자체를 통해 기록한다.

=== 타 모듈과의 연결 ===
- GIDS_Setup/GIDS/GIDS.py: GIDS.store_tensor 래퍼.
- gids_module/gids_nvme.cu: BAM_Feature_Store<float>::store_tensor (pybind 노출).
- dataloader.py: 동일 --path 의 numpy feature 와 논리 오프셋이 학습 시 읽기
  경로에 1:1 대응해야 한다.
- --ssd_list: 스트라이핑 순서를 문자열("0,1,2") 로 받아 int list 로 파싱.

=== 주요 함수/구조체 요약 ===
메인 스크립트 entry 만 존재.
1) argparse 로 페이지/SSD/오프셋/캐시 파라미터 파싱.
2) `GIDS.GIDS(...)` 인스턴스 생성 → BAM_Feature_Store_float 초기화.
3) np.load 로 host numpy 배열 → torch tensor → GPU 이동 →
   `GIDS_Loader.store_tensor(emb_tensor, 0)` 로 logical offset 0 부터 쓰기.
"""

import math                                       # [한국어] 현재 미사용이지만 원본 유지 (셸 산술 외 수학 유틸 후보).
import argparse, datetime                         # [한국어] argparse: CLI 파싱. datetime: 로그 timestamp 용.
import dgl                                        # [한국어] DGL import - 본 파일은 직접 사용 X. models/dataloader import 시 연관.
import sklearn.metrics                            # [한국어] 원본 이식성 때문에 함께 import. 본 파일에선 미사용.
import torch, torch.nn as nn, torch.optim as optim  # [한국어] 텐서/Module/Optimizer. store_tensor 로 전달할 GPU 텐서를 만든다.
import time, tqdm, numpy as np                    # [한국어] np.load 로 feature 파일 메모리 맵/로드.
from models import *                              # [한국어] evaluation/models.py 의 GNN 클래스들. 직접 사용 X - 임포트 호환성 목적.
from dataloader import IGB260MDGLDataset, OGBDGLDataset
# [한국어] 원본이 학습 스크립트를 복제 수정했기 때문에 dataset 클래스들도 같이 import 되어 있음. store 경로에서는 미사용.
import csv                                        # [한국어] 로그 CSV 기록용 후보, 현재 미사용.
import warnings                                   # [한국어] DGL/numpy deprecation 경고 억제 후보.

import torch.cuda.nvtx as t_nvtx                  # [한국어] NVTX 프로파일 범위 표시. 본 파일엔 직접 호출 없으나 템플릿 유지.
import threading                                  # [한국어] 멀티 writer 확장 후보. 현재 미사용.


import GIDS                                        # [한국어] GIDS_Setup 에서 설치한 파이썬 패키지 - BAM_Feature_Store 래퍼.
from GIDS import GIDS_DGLDataLoader               # [한국어] 학습용 DataLoader - 본 파일은 쓰기 경로만 써서 미활용이지만 import 호환성.

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
# [한국어] OGB 데이터셋 유틸 - 본 파일은 데이터 로딩을 np.load 로만 처리하므로 실제 사용 X.
#          학습 스크립트와 import 시그니처를 맞추기 위해 남아 있다.




if __name__ == '__main__':
    # [한국어] CLI 파싱. 아래 플래그들은 GIDS.GIDS 생성자와 1:1 대응된다.
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M',
        help='path containing the datasets')
    # [한국어] --path: SSD 에 기록할 numpy feature 파일(.npy) 경로. np.load 로 읽는다.

    #GPU Software Cache Parameters
    parser.add_argument('--page_size', type=int, default=8)
    # [한국어] --page_size: BaM page_cache_t 의 페이지 크기(byte). NVMe LBA 단위와 일치시켜야 성능 이득.
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD in byte(offset should be in page size granularity)')
    # [한국어] --offset: SSD 내 기록 시작 바이트 오프셋. page_size 의 배수여야 함.
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)')
    # [한국어] --num_ele: 데이터셋의 총 원소 개수 (= 총 바이트 / sizeof(float)). range_t n_elems.
    parser.add_argument('--cache_dim', type=int, default=1024)
    # [한국어] --cache_dim: feature 1개의 차원 수. write_feature_kernel 에서 warp-per-node 구조와 함께 사용.
    parser.add_argument('--num_ssd', type=int, default=1)
    # [한국어] --num_ssd: 사용할 Controller(=libnvm 디바이스) 수. 2 이상이면 striping 가능.
    parser.add_argument('--cache_size', type=int, default=8)
    # [한국어] --cache_size: GPU page cache 크기 (MB). 기록 경로에선 스테이징 버퍼로 사용.

    parser.add_argument('--ssd_list', type=str, default=None)
    # [한국어] --ssd_list: "0,1,2" 형식으로 /dev/libnvm 인덱스 나열. None 이면 0..num_ssd-1 기본.


    parser.add_argument('--mmap', type=int, default=0)
    # [한국어] --mmap: host 메모리 로딩 모드 플래그(확장용). 현재 소비 코드 없음.
    parser.add_argument('--device', type=int, default=0)
    # [한국어] --device: 사용할 CUDA 디바이스 인덱스.

    args = parser.parse_args()                                  # [한국어] sys.argv 파싱.
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    # [한국어] GPU 가용 시 cuda:N, 아니면 cpu. store_tensor 는 GPU 텐서 전제.

    gids_ssd_list = None                                        # [한국어] 기본값: Controller auto 선택.
    if (args.ssd_list != None):                                 # [한국어] 사용자가 리스트를 제공한 경우.
        gids_ssd_list =  [int(ssd_list) for ssd_list in args.ssd_list.split(',')]
        # [한국어] 콤마 구분 문자열 → int 리스트. /dev/libnvm{i} 인덱스로 해석됨.

    print("GIDS SSD List: ", gids_ssd_list)                     # [한국어] 파싱 결과 확인용 로그.

    # [한국어] GIDS 파사드 생성. 내부적으로 C++ BAM_Feature_Store_float 을 통해
    #          Controller 벡터 + page_cache_t + range_t + array_t 를 초기화한다.
    GIDS_Loader = GIDS.GIDS(
        page_size = args.page_size,      # [한국어] BaM page 단위.
        off = args.offset,                # [한국어] range_t 시작 byte offset.
        num_ele = args.num_ele,           # [한국어] range_t 원소 수.
        num_ssd = args.num_ssd,           # [한국어] Controller 개수.
        cache_size = args.cache_size,     # [한국어] page cache MB.
        cache_dim = args.cache_dim,       # [한국어] feature dim (write_feature_kernel warp 매핑).
        ssd_list = gids_ssd_list          # [한국어] 사용할 libnvm 인덱스 목록.
    )

    emb = np.load(args.path)                                    # [한국어] .npy 파일 → numpy array (host).
    emb_tensor = torch.tensor(emb).to(device)                   # [한국어] numpy → torch CPU tensor → GPU 이동. device 는 위 cuda:N.
    GIDS_Loader.store_tensor(emb_tensor, 0)                     # [한국어] 두 번째 인자 = 시작 logical index (= 0 부터 기록).
                                                                 #          내부: BAM_Feature_Store::store_tensor → write_feature_kernel 런칭.






