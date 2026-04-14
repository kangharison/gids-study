"""
[한국어] GIDS 단위 테스트 스크립트 (GIDS_unit_test.py)

=== 파일의 역할 ===
학습 전체를 돌리지 않고도 GIDS 파이썬 ↔ C++/CUDA 바인딩 경로가 살아있는지
검증하는 최소 테스트. `GIDS.GIDS(...)` 객체를 만든 뒤 `fetch_test` 와
`fetch_hetero_test` 를 호출해 BAM_Feature_Store::fetch_test(homogeneous)
및 heterograph 경로의 기본 동작을 확인한다. evaluation/gids_unit_test.sh 가
이 파일을 래핑해 실행한다.

=== 전체 아키텍처에서의 위치 ===
gids_unit_test.sh → GIDS_unit_test.py → GIDS.GIDS (GIDS_Setup/GIDS/GIDS.py)
→ pybind11 → gids_module/gids_nvme.cu: BAM_Feature_Store::fetch_test /
fetch_hetero_test → gids_kernel.cu 의 커널 런칭 → bam page_cache → libnvm.
즉 학습 경로와 동일한 스택을 밟지만 DGL sampler/모델은 생략된 smoke test.

=== 타 모듈과의 연결 ===
- 의존: GIDS (파이썬 패키지), models.py (GNN 모델 import 호환성), BaM 모듈.
- 호출 대상: GIDS_Loader.fetch_test / fetch_hetero_test — 각각 homogeneous /
  heterogeneous 설정에서 page cache 접근 경로 smoke 테스트.
- --heterograph=True 로 GIDS 를 만들어 heterograph_map 경로도 초기화되는지
  함께 확인한다.

=== 주요 함수/구조체 요약 ===
- argparse 기반의 CLI 구성 (학습 스크립트와 동일 옵션 집합 재사용).
- main:
    GIDS.GIDS(...) 생성 → fetch_test(1, 1024) → fetch_hetero_test(1, 1024) →
    반환 텐서 print. 에러가 없으면 pybind 바인딩과 libnvm 접근이 정상.
"""

import argparse, datetime
import dgl
import sklearn.metrics                                    # [한국어] import 호환용. 본 파일에서 직접 사용 X.
import torch, torch.nn as nn, torch.optim as optim         # [한국어] PyTorch 기본. fetch_test 가 반환하는 GPU 텐서 확인에 사용.
import time, tqdm, numpy as np                             # [한국어] 학습 스크립트와 공통 import 세트.
from models import *                                        # [한국어] GNN 모델 import (테스트 자체는 미사용이지만 import 호환).

import csv                                                  # [한국어] 로그 CSV 저장 후보. 미사용.
import warnings                                             # [한국어] deprecation 경고 억제 후보.

import torch.cuda.nvtx as t_nvtx                           # [한국어] NVTX 범위 표기 — 프로파일 연결 편의.
import nvtx                                                 # [한국어] python nvtx 래퍼.
import threading                                            # [한국어] 다중 스레드 확장 후보. 미사용.
import gc                                                   # [한국어] garbage collection 수동 호출 후보.

import GIDS                                                 # [한국어] GIDS 파이썬 패키지. 내부에서 BAM_Feature_Store_{float,long} pybind 모듈 사용.


if __name__ == '__main__':
    # [한국어] CLI 파싱. 아래 인자들은 학습 스크립트와 동일 형태여서 그대로 복사되어 있다.
    #          실제 단위 테스트에서는 --num_ele, --num_ssd, --cache_size 정도만 의미.

    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983, 172, 348,349, 350, 153, 152], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)
    
    # Model
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['rgat', 'rsage', 'rgcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters 
    parser.add_argument('--fan_out', type=str, default='10,15')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=2)

    #GIDS parameter
    parser.add_argument('--GIDS', action='store_true', help='Enable GIDS Dataloader')
    parser.add_argument('--num_ssd', type=int, default=1)
    parser.add_argument('--cache_size', type=int, default=8)
    parser.add_argument('--uva', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)
    parser.add_argument('--wb_size', type=int, default=6)

    parser.add_argument('--device', type=int, default=0)

    #GIDS Optimization
    parser.add_argument('--accumulator', action='store_true', help='Enable Storage Access Accmulator')
    parser.add_argument('--bw', type=float, default=5.8, help='SSD peak bandwidth in GB/s')
    parser.add_argument('--l_ssd', type=float, default=11.0, help='SSD latency in microseconds')
    parser.add_argument('--l_system', type=float, default=20.0, help='System latency in microseconds')
    parser.add_argument('--peak_percent', type=float, default=0.95)

    parser.add_argument('--num_iter', type=int, default=1)

    parser.add_argument('--cpu_buffer', action='store_true', help='Enable CPU Feature Buffer')
    parser.add_argument('--cpu_buffer_percent', type=float, default=0.2, help='CPU feature buffer size (0.1 for 10%)')
    parser.add_argument('--pin_file', type=str, default="/mnt/nvme16/pr_full.pt", 
        help='Pytorch Tensor File for the list of nodes that will be pinned in the CPU feature buffer')

    parser.add_argument('--window_buffer', action='store_true', help='Enable Window Buffering')



    #GPU Software Cache Parameters
    parser.add_argument('--page_size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD') 
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)') 
    parser.add_argument('--cache_dim', type=int, default=1024) #CHECK


    args = parser.parse_args()                                # [한국어] CLI 파싱 완료.

    GIDS_Loader = None                                        # [한국어] 명시적 초기화. 생성 실패 시 None 유지되어 디버깅 쉬움.
    # [한국어] GIDS 객체 생성. 내부에서
    #          - GIDS_Controllers.init_GIDS_controllers(ssd_list, num_ssd, queueDepth, ...)
    #          - BAM_Feature_Store_float.init_controllers(page_cache, range_t, array_t)
    #          - heterograph=True 인 경우 heterograph_map 구조 초기화까지 수행.
    GIDS_Loader = GIDS.GIDS(
        page_size = 4096,            # [한국어] 4KB — 일반적인 NVMe LBA 와 일치.
        off = 0,                     # [한국어] range_t start byte = 0 (SSD 앞부터).
        num_ele = args.num_ele,       # [한국어] range_t 총 element 수. CLI 에서 받음.
        num_ssd = args.num_ssd,       # [한국어] Controller 개수.
        cache_size = args.cache_size, # [한국어] page cache MB.
        cache_dim = 1024,             # [한국어] feature 차원 (fetch_test 는 cache_dim 과 맞춰 GPU 텐서 반환).
        heterograph = True            # [한국어] heterograph 경로 초기화. fetch_hetero_test 호출 전제 조건.
    )
    # ret_ten2 = GIDS_Loader.fetch_hetero_test(1, 1024)
    # print("second retern: ", ret_ten2)
    # torch.cuda.synchronize()
    # [한국어] 위 3줄은 원본 저자가 실험 중 사용한 디버그 흔적. 주석 처리되어 비활성.

    # [한국어] fetch_test(batch_size=1, dim=1024) — homogeneous 경로. 내부에서
    #          read_feature_kernel 을 간이 런칭하고 결과 텐서를 반환해
    #          pybind 반환형(torch.Tensor)과 page_cache/Controller 가 정상 작동함을 증명.
    ret_ten = GIDS_Loader.fetch_test(1, 1024)
    # [한국어] heterograph 경로 smoke 테스트. heterograph_map 오프셋 라우팅까지 관여.
    second_ret_ten = GIDS_Loader.fetch_hetero_test(1, 1024)
    print("second retern: ", second_ret_ten)                  # [한국어] 반환 텐서 내용/shape 확인.
    print("first retern: ", ret_ten)                          # [한국어] fetch_test 결과. 기대: float 텐서 shape=(1,1024).




#CUDA_VISIBLE_DEVICES=0  python GIDS_unit_test.py  --cache_size $((8*1024)) --num_ssd 1   --num_ele $((550*1000*1000*1024)) 