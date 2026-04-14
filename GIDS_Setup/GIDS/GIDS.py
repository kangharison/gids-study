"""
[한국어 설명] GIDS(GPU-Initiated Direct Storage) Python 메인 API 구현 (GIDS.py)

=== 파일의 역할 ===
이 파일은 GIDS 프로젝트의 Python 측 최상위 API를 정의하는 가장 중요한 모듈이다.
C++/CUDA 레이어(gids_module/gids_nvme.cu, gids_kernel.cu)를 pybind11로 노출한
`BAM_Feature_Store` 확장을 감싸서, DGL GNN 학습 파이프라인이 아무 변경 없이
"GPU가 직접 NVMe SSD에서 feature를 읽는" 동작을 사용할 수 있게 만든다.
구체적으로는 (1) BaM Controller/페이지 캐시/range 초기화, (2) DGL DataLoader
라이프사이클 래핑, (3) Window Buffering/Storage Access Accumulator 등 GIDS 고유
최적화 경로의 정책 결정, (4) feature 쓰기(store_tensor)와 캐시 통계 출력을 담당한다.
이 파일 하나를 읽으면 GIDS가 학습 한 iteration 안에서 어떤 순서로 BaM을 호출하고,
어떤 시점에 CPU 버퍼/윈도우 버퍼/직접 SSD 읽기 경로를 선택하는지 전부 파악된다.

=== 전체 아키텍처에서의 위치 ===
호출 체인(하향):
  evaluation/homogenous_train*.py, heterogeneous_train*.py
    → GIDS_DGLDataLoader(...)          (본 파일 클래스)
        → torch.utils.data.DataLoader.__iter__() / super().__iter__()
        → _PrefetchingIter.__next__()  (본 파일 클래스)
            → GIDS.fetch_feature()      (본 파일 메소드, 핵심 진입)
                → BAM_Feature_Store_{float,long}.read_feature / read_feature_hetero /
                  read_feature_merged / read_feature_merged_hetero / set_window_buffering
                  → gids_module/gids_nvme.cu (호스트) 의 BAM_Feature_Store<T>::read_feature*
                  → gids_module/gids_kernel.cu 의 __global__ 커널 런칭
                    → bam_ptr.read() → BaM page_cache → miss 시 NVMe SSD 접근
실행 컨텍스트: 모두 호스트 메인 Python 프로세스(단일 학습 프로세스 또는 torch DDP
rank). num_workers>0일 때는 CollateWrapper 부분만 DataLoader worker 서브프로세스에서
실행되지만, BAM_FS 호출은 반드시 메인 프로세스(=GPU 스트림을 가진 프로세스)에서만
수행되어야 한다(BaM Controller 핸들이 프로세스 로컬 리소스이기 때문).

=== 타 모듈과의 연결 ===
- BAM_Feature_Store (pybind11 확장): 클래스 BAM_Feature_Store_float / _long /
  GIDS_Controllers 제공. long_type=True 일 때 int64 feature 용 _long 바인딩을,
  False 일 때 float32 feature 용 _float 바인딩을 선택하며, 이는 C++ 템플릿
  BAM_Feature_Store<int64_t> vs <float> 에 각각 대응된다. GIDS_Controllers는
  gids_module/gids_nvme.cu 의 GIDS_Controllers (BaM Controller 벡터 소유) 래퍼.
- dgl.dataloading: create_tensorized_dataset(배치 단위 인덱스 데이터셋 생성),
  WorkerInitWrapper(DataLoader worker 프로세스 초기화 훅), BlockSampler, 그리고
  remove_parent_storage_columns(배치의 부모 그래프 참조 정리)를 이용한다.
- torch.utils.data.DataLoader: GIDS_DGLDataLoader 가 직접 상속하여 __iter__에서
  Prefetching 반복자로 교체하는 구조.
- nvtx / ctypes: 프로파일링 마커 및 원시 포인터 전달 용도로 사용될 수 있다 (현재 파일에서는
  import만 되어 있음).

공유 자료구조:
  GIDS_controller(=GIDS_Controllers pybind 객체, SSD당 BaM Controller* 보유)
    ↔ BAM_FS(=BAM_Feature_Store_float/_long, page_cache_t + range_t + array_t 소유)
  self.window_buffer[리스트] ↔ self.return_torch_buffer[리스트]: Storage Access
    Accumulator 모드에서 "샘플러가 만들어낸 배치"와 "fetch 결과 텐서"를 큐로 병렬 보유.

=== 주요 함수/구조체 요약 ===
- CollateWrapper: DGL 샘플러 호출 + device 이동 + 부모 스토리지 정리를 하나의
  collate_fn 으로 묶어 torch DataLoader에 넘기는 얇은 콜러블.
- _PrefetchingIter: torch DataLoader 반복자를 감싸 매 __next__ 마다
  GIDS.fetch_feature(...)를 호출해 "샘플링 + feature fetch" 을 한 스텝으로 제공.
- GIDS_DGLDataLoader: torch DataLoader 서브클래스. graph/sampler/인덱스를 받아
  샘플링 전용 DataLoader를 구성하고, __iter__에서 _PrefetchingIter 로 승격.
- GIDS: 메인 API. BAM_Feature_Store 인스턴스와 BaM 컨트롤러 초기화, Constant CPU
  Buffer 설치(set_cpu_buffer), Window Buffering 예약(set_window_buffering),
  Storage Access Accumulator 정책 관리, fetch_feature 실행, 특성 텐서 쓰기
  (store_tensor/store_mmap_tensor), 그래프 샘플링용 보조 GIDS(graph_GIDS) 관리.
"""

import math
# [한국어] math: self.off(페이지 단위로 정렬된 시작 오프셋) 계산 시 ceil 용도로 사용.
import time
# [한국어] time: fetch_feature/window_buffering 내부 타이밍(self.GIDS_time, self.WB_time) 측정용.
import torch
# [한국어] torch: 인덱스 텐서·feature 반환 텐서 생성과 .to(device)로 GPU 전송에 필수.
import numpy as np
# [한국어] numpy: 현재 파일에서 직접 사용은 없지만 evaluation 코드와의 타입 호환성 유지 목적.
import ctypes
# [한국어] ctypes: 일부 경로에서 Python int/ndarray 를 C 포인터로 변환해 pybind에 넘기기 위한 보조.
import nvtx
# [한국어] nvtx: NVIDIA Tools Extension. 학습 단계별 구간 표시(프로파일링)용으로 import되어 있음.

import BAM_Feature_Store
# [한국어] pybind11로 빌드된 gids_module의 확장 모듈. 아래 세 심볼을 핵심으로 제공:
#   - BAM_Feature_Store.BAM_Feature_Store_float  : C++ BAM_Feature_Store<float> 바인딩 (feature = float32)
#   - BAM_Feature_Store.BAM_Feature_Store_long   : C++ BAM_Feature_Store<int64_t> 바인딩 (feature = int64)
#   - BAM_Feature_Store.GIDS_Controllers         : BaM Controller 벡터 관리자
#  타입 디스패치는 GIDS.__init__의 long_type 플래그로 결정된다.

import dgl
# [한국어] Deep Graph Library. 본 파일에서는 backend(F), DGLHeteroGraph, LazyFeature, DistGraph,
#  그리고 dataloading 헬퍼들을 재사용한다. GIDS는 DGL 샘플러 출력 형식을 그대로 수용한다.
from torch.utils.data import DataLoader
# [한국어] GIDS_DGLDataLoader 가 상속할 PyTorch 표준 DataLoader.
from collections.abc import Mapping
# [한국어] heterograph(이종 그래프)에서 indices가 dict(Mapping)로 들어오는지 판별하기 위해 사용.

from dgl.dataloading import create_tensorized_dataset, WorkerInitWrapper, remove_parent_storage_columns
# [한국어] DGL DataLoader 내부 유틸:
#   - create_tensorized_dataset: 노드 ID 텐서를 배치 사이즈로 쪼개는 IterableDataset 생성.
#   - WorkerInitWrapper: DataLoader worker 프로세스 초기화 훅(스레드 수 고정 등)을 감싸는 래퍼.
#   - remove_parent_storage_columns: 배치(block)에서 부모 그래프의 feature 스토리지 참조를 제거해
#     worker→main 프로세스 전달 시 불필요한 큰 객체 복사를 피한다.
from dgl.utils import (
    recursive_apply, ExceptionWrapper, recursive_apply_pair, set_num_threads, get_num_threads,
    get_numa_nodes_cores, context_of, dtype_of)
# [한국어] DGL 유틸. 본 파일에서 실제로 쓰는 것은 recursive_apply(중첩 컨테이너에 fn 적용)뿐이며,
#  나머지는 원본 리포에서의 래핑 호환용 import 로 남아 있다.

from dgl import DGLHeteroGraph
# [한국어] 그래프 타입 판별용(이종 그래프면 create_formats_/pin_memory_ 수행).
from dgl.frame import LazyFeature
# [한국어] DGL의 지연 로딩 feature 객체 타입 (현재 파일 직접 사용 없음, 호환성 유지 목적).
from dgl.storages import wrap_storage
# [한국어] DGL feature storage 래퍼 (현재 파일 직접 사용 없음).
from dgl.dataloading.base import BlockSampler, as_edge_prediction_sampler
# [한국어] BlockSampler: GIDS가 받는 graph_sampler 의 추상 베이스 타입.
from dgl import backend as F
# [한국어] DGL backend 모듈(F.*). 일부 경로 호환성 목적.
from dgl.distributed import DistGraph
# [한국어] 분산 그래프 타입. GIDS는 DistGraph를 거부하고 DistNodeDataLoader 사용을 요구한다.
from dgl.multiprocessing import call_once_and_share
# [한국어] DGL 멀티프로세싱 유틸(현 파일 직접 사용 없음).

def _get_device(device):
    """
    [한국어]
    _get_device - 사용자가 넘긴 device 스펙(문자열/정수/torch.device)을
    완전한 torch.device 객체로 정규화한다.

    @param device: 'cuda', 'cuda:0', 0, torch.device('cuda') 등 다양한 형태.
    @return: 인덱스가 확정된 torch.device. cuda 타입이지만 인덱스가 None 이면
             현재 스레드의 CUDA 현재 디바이스 인덱스를 채워 넣는다.

    호출자: GIDS_DGLDataLoader.__init__ 에서 self.device 결정 시 1회 호출.
    호출 대상: torch.device 생성자, torch.cuda.current_device.
    실행 컨텍스트: 메인 프로세스, DataLoader 생성 시점.
    """
    # [한국어] 문자열/int/torch.device 어느 것이 와도 torch.device 로 일관되게 래핑.
    device = torch.device(device)
    # [한국어] 'cuda' 만 지정되고 인덱스가 없으면, 현재 스레드의 CUDA 현재 디바이스로 인덱스 고정.
    #  - 이후 .to(self.device) 호출에서 매번 현재 디바이스를 조회하는 오버헤드를 없앤다.
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    # [한국어] 확정된 torch.device 반환. 호출자는 이후 이 값을 self.device 로 고정해 사용.
    return device

class CollateWrapper(object):
    """
    [한국어]
    CollateWrapper - DGL BlockSampler 의 sample 호출을 torch DataLoader 의
    collate_fn 시그니처에 맞게 감싸는 얇은 콜러블 객체.

    역할: DataLoader worker/메인 스레드에서 배치(노드 ID 리스트)가 만들어질 때,
      (1) 노드 ID 텐서를 self.device 로 이동,
      (2) graph_sampler.sample(g, items) 호출로 Message Flow Graph(block) 생성,
      (3) 부모 그래프 스토리지 참조 제거로 프로세스 간 전달 크기 최소화.

    인스턴스 수명: GIDS_DGLDataLoader.__init__ 에서 1회 생성되어 super().__init__ 의
      collate_fn 으로 넘겨지고, DataLoader 가 살아있는 동안 매 배치마다 호출된다.
    상위 호출자: torch DataLoader 내부 collate 파이프라인 (worker 또는 메인 스레드).
    하위 의존: self.sample_func (BlockSampler.sample), dgl.utils.recursive_apply,
      dgl.dataloading.remove_parent_storage_columns.
    """
    def __init__(self, sample_func, g,  device):
        """
        [한국어]
        CollateWrapper 초기화. 샘플 함수/그래프/디바이스를 캡처해 둔다.

        @param sample_func: BlockSampler.sample 바운드 메소드.
          배치 node ID → MFG/block 시퀀스 변환.
        @param g: DGL 그래프 객체(DGLHeteroGraph 또는 호환 타입). 핀 메모리 상태여야 한다.
        @param device: 최종 배치가 실려야 할 torch.device (학습 GPU).
        실행 컨텍스트: 메인 프로세스, DataLoader 생성 시점에 단 1회.
        """
        # [한국어] 샘플러의 sample 메소드 참조. 호출 시점에는 worker 프로세스로 pickle 되어 복제될 수 있다.
        self.sample_func = sample_func
        # [한국어] 원본 DGL 그래프. create_formats_() + pin_memory_() 가 사전에 되어 있어야 한다.
        self.g = g
        # [한국어] 배치 텐서가 최종 이동할 타깃 디바이스 (예: cuda:0).
        self.device = device

    def __call__(self, items):
        """
        [한국어]
        CollateWrapper.__call__ - 한 배치 분량의 노드 ID 리스트를 받아 DGL 블록 배치로 변환.

        @param items: torch IterableDataset 가 내놓는 노드 ID 텐서(또는 이종 그래프면 dict).
        @return: (input_nodes, output_nodes, blocks) 형태의 DGL 배치(tuple). 이후 fetch_feature가
          여기에 feature 텐서를 append/튜플 확장해 반환한다.

        호출자: torch DataLoader 내부 collate 파이프라인.
        하위 호출: recursive_apply(.to(device)), self.sample_func, remove_parent_storage_columns.
        실행 컨텍스트: num_workers>0 이면 worker 서브프로세스, 아니면 메인 스레드.
        """
        # [한국어] 그래프가 얹혀 있는 디바이스 정보를 확보. (현 구현에서는 반환값을 직접 쓰지는 않음)
        graph_device = getattr(self.g, 'device', None)
        # [한국어] 배치 items(텐서 또는 dict) 각 엔트리에 .to(self.device) 를 재귀적으로 적용.
        #  - recursive_apply: dict/list/tuple 내부의 모든 텐서에 람다를 동일 적용하는 DGL 유틸.
        items = recursive_apply(items, lambda x: x.to(self.device))
        # [한국어] DGL 샘플러 호출: 전달된 노드 ID 로부터 MFG(블록) 시퀀스 생성.
        batch = self.sample_func(self.g, items)
        # [한국어] 부모 그래프에 대한 feature 스토리지 참조를 배치에서 제거.
        #  - worker→main 프로세스 pickle 크기를 줄이고, fetch_feature 가 GPU-direct 경로로 feature를
        #    채울 것이므로 parent storage 는 불필요하다.
        return recursive_apply(batch, remove_parent_storage_columns, self.g)


class _PrefetchingIter(object):
    """
    [한국어]
    _PrefetchingIter - torch DataLoader 의 반복자(dataloader_it)를 감싸서 매 next 호출 시
    GIDS.fetch_feature 를 자동으로 실행해 "샘플링된 배치 + GPU로 직접 fetch한 feature"를
    한 번에 반환하는 반복자 래퍼.

    인스턴스 수명: GIDS_DGLDataLoader.__iter__ 가 호출될 때마다 새 인스턴스가 생성된다
      (즉 epoch당 1개). 내부의 dataloader_it 가 고갈되면 StopIteration 이 자연 전파된다.
    상위 호출자: 학습 스크립트의 `for batch in gids_dataloader:` 루프.
    하위 의존: GIDS_Loader.fetch_feature (본 파일 GIDS 클래스).
    """
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None):
        """
        [한국어]
        @param dataloader: 감싸는 GIDS_DGLDataLoader 인스턴스. dim/graph_sampler 참조용.
        @param dataloader_it: 내부 torch DataLoader 의 실제 반복자(샘플링된 배치 생성원).
        @param GIDS_Loader: GIDS 메인 API 인스턴스. fetch_feature 호출 대상.
        실행 컨텍스트: 메인 프로세스, epoch 시작 시.
        """
        # [한국어] 내부 반복자 저장. next(self.dataloader_it) 가 "샘플링만 수행된 배치" 를 반환.
        self.dataloader_it = dataloader_it
        # [한국어] 상위 DataLoader 래퍼. feature 차원(dataloader.dim) 조회 등에 사용.
        self.dataloader = dataloader
        # [한국어] 그래프 샘플러 참조. 현재 구현에서는 직접 호출하지 않지만, 디버그/확장용으로 보관.
        self.graph_sampler = self.dataloader.graph_sampler
        # [한국어] GIDS feature fetch 엔진 참조. fetch_feature 호출의 주체.
        self.GIDS_Loader=GIDS_Loader

    def __iter__(self):
        """
        [한국어]
        자기 자신이 이미 iterator 이므로 그대로 반환. torch 표준 반복자 프로토콜 충족.
        """
        return self

    def __next__(self):
        """
        [한국어]
        __next__ - 다음 학습 배치를 얻는다. 내부적으로 GIDS.fetch_feature 를 호출하며,
        그 안에서 next(dataloader_it) 가 호출되어 샘플링 + feature fetch 가 결합된다.

        @return: fetch_feature 가 반환하는 (샘플링 배치 + feature 텐서) 구조.
        호출자: 학습 루프의 for 문.
        하위 호출: GIDS_Loader.fetch_feature(dim, iterator, device).
        실행 컨텍스트: 메인 프로세스의 학습 스레드.
        """
        # [한국어] fetch_feature 에 넘길 iterator 참조 준비.
        cur_it = self.dataloader_it
        # [한국어] GIDS 엔진 호출: feature 차원, 내부 iterator, GIDS가 관리하는 CUDA 디바이스 식별자.
        #  - dataloader.dim: feature 벡터 차원(예: 1024).
        #  - GIDS_Loader.gids_device: 'cuda:<ctrl_idx>' 형태의 타깃 디바이스.
        batch = self.GIDS_Loader.fetch_feature(self.dataloader.dim, cur_it, self.GIDS_Loader.gids_device)
        # [한국어] 완성된 배치 반환. StopIteration 은 fetch_feature 안쪽 next(it) 에서 자연 전파된다.
        return batch



class GIDS_DGLDataLoader(torch.utils.data.DataLoader):
    """
    [한국어]
    GIDS_DGLDataLoader - DGL 의 DataLoader를 대체하는 GIDS 전용 DataLoader.

    역할:
      - 입력: DGL graph, 학습 노드 인덱스(또는 dict), BlockSampler, batch_size, dim, GIDS 엔진.
      - 내부적으로 torch DataLoader 를 상속하여 collate_fn = CollateWrapper 로 세팅 →
        샘플러가 매 배치마다 MFG(block) 시퀀스를 생성.
      - __iter__ 에서 기본 iterator 를 _PrefetchingIter 로 감싸서, 매 iteration 에
        GIDS.fetch_feature 가 자동으로 호출되게 한다.

    인스턴스 수명: 학습 스크립트 시작 시 1회 생성되어 모든 epoch 동안 재사용.
    상위 호출자: evaluation/*_train.py 의 학습 루프.
    하위 의존: DGL 샘플러, CollateWrapper, _PrefetchingIter, GIDS.
    """

    def __init__(self, graph, indices, graph_sampler, batch_size, dim, GIDS, device=None, use_ddp=False,
                 ddp_seed=0, drop_last=False, shuffle=False,
                 use_alternate_streams=None,

                 **kwargs):
        """
        [한국어]
        GIDS_DGLDataLoader 초기화. DGL NodeDataLoader 의 동작을 재현하면서 GIDS 엔진을 결합.

        @param graph: DGLHeteroGraph(또는 호환 그래프). DistGraph는 허용되지 않음.
        @param indices: 학습/검증에 쓸 seed 노드 ID 텐서, 또는 type→ID 텐서 dict(이종 그래프).
        @param graph_sampler: DGL BlockSampler (예: MultiLayerNeighborSampler).
        @param batch_size: 한 iteration 당 seed 노드 개수.
        @param dim: feature 텐서의 차원. fetch_feature 가 return_torch 크기 결정에 사용.
        @param GIDS: 사전에 생성된 GIDS 인스턴스. self.GIDS_Loader 로 보관되어 매 배치 fetch.
        @param device: 타깃 CUDA 디바이스. None이면 현재 디바이스 사용.
        @param use_ddp / ddp_seed: torch DDP 학습 시 분산 셔플 시드.
        @param drop_last / shuffle: torch DataLoader 표준 옵션.
        @param use_alternate_streams: feature pin/transfer 를 별도 CUDA 스트림에서 수행할지.
        @param **kwargs: num_workers, persistent_workers, pin_memory 등 DataLoader 추가 옵션.

        호출자: 학습 스크립트. 호출 대상: super().__init__ (torch DataLoader).
        실행 컨텍스트: 메인 프로세스, epoch 시작 전 1회.
        """

        # [한국어] GIDS는 UVA(Unified Virtual Addressing) 기반 CPU pin 경로 대신 자체 BaM 경로를 쓰므로 항상 False.
        use_uva = False
        # [한국어] GIDS 엔진 참조 저장. 설정자: 여기 1회. 읽는 자: __iter__ 에서 _PrefetchingIter 에 전달.
        #  - 값 범위: GIDS 인스턴스(None 금지). 동기화: 단일 메인 프로세스 내 소유.
        self.GIDS_Loader = GIDS
        # [한국어] feature 차원. 설정자: 여기 1회. 읽는 자: _PrefetchingIter.__next__.
        self.dim = dim


        # [한국어] 특수 경로: 외부에서 이미 CollateWrapper 가 설정된 kwargs 로 재구성하려는 경우
        #  (DGL 내부 코드에서 super().__init__ 을 다시 호출하면서 본 클래스를 재사용하는 시나리오).
        #  이때는 모든 멤버만 복원하고 super().__init__ 을 바로 호출해 반환한다.
        if isinstance(kwargs.get('collate_fn', None), CollateWrapper):
            assert batch_size is None       # must be None
            # restore attributes
            # [한국어] 아래는 모두 "이미 아는 값"을 self 로 복구하는 단순 대입이다.
            #  설정자: 이 블록. 읽는 자: __iter__, fetch 경로. 값 범위/동기화: 생성자 표준 대입 규칙.
            self.graph = graph
            self.indices = indices
            self.graph_sampler = graph_sampler
            self.device = device
            self.use_ddp = use_ddp
            self.ddp_seed = ddp_seed
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.use_alternate_streams = use_alternate_streams
            self.use_uva = use_uva
            # [한국어] tensorized dataset 이 이미 batch 단위로 잘려 들어오므로 DataLoader batch_size 는 None.
            kwargs['batch_size'] = None
            # [한국어] torch DataLoader 원본 생성자로 남은 kwargs 전달.
            super().__init__(**kwargs)
            return


        # [한국어] 분산 그래프(DistGraph)는 GIDS 의 단일 프로세스 Controller 모델과 맞지 않음.
        if isinstance(graph, DistGraph):
            raise TypeError(
                'Please use dgl.dataloading.DistNodeDataLoader or '
                'dgl.datalaoding.DistEdgeDataLoader for DistGraphs.')

        # [한국어] 일반 경로: 그래프/인덱스/샘플러를 self 에 저장.
        #  설정자: 여기. 읽는 자: CollateWrapper.__call__, __iter__, fetch_feature 경로.
        self.graph = graph
        self.indices = indices
        # [한국어] DataLoader worker 프로세스 개수. 0이면 메인 스레드에서 샘플링.
        num_workers = kwargs.get('num_workers', 0)

        # [한국어] indices 가 어느 디바이스에 있는지 자동 감지.
        indices_device = None
        try:
            # [한국어] 이종 그래프: node type 별 인덱스 dict.
            if isinstance(indices, Mapping):
                # [한국어] 각 value가 torch 텐서가 아니면 torch.tensor 로 변환해 dict 재구성.
                indices = {k: (torch.tensor(v) if not torch.is_tensor(v) else v)
                           for k, v in indices.items()}
                # [한국어] 첫 번째 텐서의 device 를 대표로 삼는다 (보통 모두 동일).
                indices_device = next(iter(indices.values())).device
            else:
                # [한국어] 단일 텐서 경로: torch 텐서로 정규화 후 device 추출.
                indices = torch.tensor(indices) if not torch.is_tensor(indices) else indices
                indices_device = indices.device
        except:     # pylint: disable=bare-except
            # ignore when it fails to convert to torch Tensors.
            # [한국어] 커스텀 dataset 등 torch.tensor 변환 실패 시 조용히 무시하고 아래 폴백으로 진행.
            pass

        # [한국어] 변환 폴백: torch 텐서가 아닌 커스텀 indices 객체는 .device 속성을 반드시 노출해야 한다.
        if indices_device is None:
            if not hasattr(indices, 'device'):
                raise AttributeError('Custom indices dataset requires a \"device\" \
                attribute indicating where the indices is.')
            indices_device = indices.device

        # [한국어] device 인자가 None이면 현재 CUDA 디바이스로 기본 설정.
        if device is None:
            device = torch.cuda.current_device()
        # [한국어] self.device 확정. 이후 모든 배치가 여기로 이동된다.
        #  설정자: 여기. 읽는 자: CollateWrapper, __iter__.
        self.device = _get_device(device)

        # Sanity check - we only check for DGLGraphs.
        # [한국어] DGLHeteroGraph 이면 CSC 포맷 생성 + host pin 처리. GIDS 의 샘플러가
        #  GPU로 그래프 토폴로지를 읽을 때 UVA/peer-access 로 핀된 메모리를 활용하기 위함.
        if isinstance(self.graph, DGLHeteroGraph):
            # [한국어] DGL이 샘플링에 필요한 CSC/CSR 포맷을 lazily 준비.
            self.graph.create_formats_()
            # [한국어] 아직 pin 안 되어 있으면 host pinned memory 로 고정.
            if not self.graph._graph.is_pinned():
                self.graph._graph.pin_memory_()


            # Check use_alternate_streams
            # [한국어] alternate stream 기본값: GPU 학습 + CPU 그래프 + non-UVA 일 때만 True.
            if use_alternate_streams is None:
                use_alternate_streams = (
                    self.device.type == 'cuda' and self.graph.device.type == 'cpu' and
                    not use_uva)

        # [한국어] indices 가 순수 torch 텐서(또는 값이 전부 텐서인 dict) 이면 배치 단위 dataset 으로 랩.
        if (torch.is_tensor(indices) or (
                isinstance(indices, Mapping) and
                all(torch.is_tensor(v) for v in indices.values()))):
            # [한국어] create_tensorized_dataset: DGL 내부의 IterableDataset 팩토리.
            #  shuffle/ddp/persistent_workers 인자를 한 번에 처리해 torch DataLoader 에 맞춘다.
            self.dataset = create_tensorized_dataset(
                indices, batch_size, drop_last, use_ddp, ddp_seed, shuffle,
                kwargs.get('persistent_workers', False))
        else:
            # [한국어] 커스텀 dataset 이면 그대로 사용.
            self.dataset = indices

        # [한국어] 이하 self.* 들은 이후 경로(__iter__, shuffle 처리, 디버그 출력)에서 읽는 단순 상태.
        #  설정자: 여기. 읽는 자: 자신. 동기화: 메인 프로세스 단일 스레드이므로 없음.
        self.ddp_seed = ddp_seed
        self.use_ddp = use_ddp
        self.use_uva = use_uva
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.graph_sampler = graph_sampler
        self.use_alternate_streams = use_alternate_streams


        # [한국어] NUMA CPU affinity 기능 enable 여부. 현재 항상 False로 초기화만 되어 있다(미사용).
        self.cpu_affinity_enabled = False

        # [한국어] DGL 이 제공하는 WorkerInitWrapper. 사용자가 worker_init_fn 을 넘겼다면 그 앞에 DGL 고유 초기화를
        #  체이닝하여 래핑한다(스레드 수 제한 등).
        worker_init_fn = WorkerInitWrapper(kwargs.get('worker_init_fn', None))

        # [한국어] DGL DataLoader 호환용: edge subgraph 등의 추가 storage 를 담는 dict. 본 GIDS 경로에선 미사용.
        self.other_storages = {}

        # [한국어] 최종적으로 torch DataLoader 원본 생성자에 위임한다.
        #  - collate_fn: CollateWrapper 로 고정 (샘플링 + device 이동 + 부모 스토리지 제거).
        #  - batch_size=None: dataset 단에서 이미 배치로 쪼갰기 때문.
        #  - pin_memory=False: feature 는 GIDS 경로로 GPU에 직접 들어오므로 불필요.
        super().__init__(
            self.dataset,
            collate_fn=CollateWrapper(
                self.graph_sampler.sample, graph, self.device),
            batch_size=None,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            **kwargs)

    def __iter__(self):
        """
        [한국어]
        __iter__ - epoch 시작마다 호출. 기본 DataLoader iterator 를 _PrefetchingIter 로 승격시켜
        feature fetch 가 자동 삽입되도록 한다.

        호출자: 학습 루프 `for batch in loader:`.
        하위 호출: self.dataset.shuffle(), super().__iter__(), _PrefetchingIter 생성.
        실행 컨텍스트: 메인 프로세스.
        """
        # [한국어] shuffle=True 이면 IterableDataset 에 셔플 신호를 보내 epoch 간 순서를 섞는다.
        if self.shuffle:
            self.dataset.shuffle()
        # When using multiprocessing PyTorch sometimes set the number of PyTorch threads to 1
        # when spawning new Python threads.  This drastically slows down pinning features.
        # [한국어] num_workers>0 이면 메인 스레드의 torch 스레드 수를 기록해 두었다가 worker 에 전파하려는 의도로 보관.
        #  (현 구현은 보관만 하고 직접 사용하진 않음 — 향후 worker_init 에서 set_num_threads 에 활용 가능).
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        # [한국어] torch DataLoader 의 표준 iterator 를 _PrefetchingIter 로 감싸 반환.
        return _PrefetchingIter(
            self, super().__iter__(), GIDS_Loader=self.GIDS_Loader)

    def print_stats(self):
        """
        [한국어] GIDS 엔진 측 누적 통계(페이지 캐시 히트/미스 등) 를 콘솔에 출력.
        호출자: 학습 스크립트의 epoch 종료부. 하위 호출: GIDS.print_stats → BAM_FS.print_stats.
        """
        self.GIDS_Loader.print_stats()

    def print_timer(self):
        """
        [한국어] 타이밍 리셋 유틸. 주석 처리된 디버그 print 블록이 있으며, 현재는 두 타이머를 0으로 초기화만 한다.
         호출자: 학습 스크립트가 epoch 간 경과 시간을 리셋하고 싶을 때.
        """
        #if(self.bam):
        #     print("feature aggregation time test: %f" % self.sample_time)
        #print("graph travel time: %f" % self.graph_travel_time)
        # [한국어] 아래 두 필드는 다른 경로에서 누적되는 것으로 가정된 디버그용 카운터 (본 파일에서 선언은 없음).
        self.sample_time = 0.0
        self.graph_travel_time = 0.0

class GIDS():
    """
    [한국어]
    GIDS - GPU-Initiated Direct Storage 메인 엔진 클래스.

    역할:
      - BaM Controller 벡터와 페이지 캐시/range/array 를 초기화 (BAM_FS.init_controllers 경유).
      - feature dtype 에 따라 BAM_Feature_Store_float (C++ <float>) 또는 _long (C++ <int64_t>)
        pybind 바인딩을 선택.
      - Constant CPU Buffer 설치(cpu_backing_buffer + set_cpu_buffer)로 핫 노드를 호스트 pinned
        메모리에 고정해 SSD 접근을 회피.
      - Window Buffering(set_window_buffering)으로 다음 n배치 feature 를 선행 프리패치.
      - Storage Access Accumulator 경로로 여러 배치를 묶어 read_feature_merged 로 일괄 fetch.
      - 샘플링용 그래프 전용 GIDS(graph_GIDS) 인스턴스 관리(init_graph_GIDS).
      - 학습 외 유틸: store_tensor/store_mmap_tensor (feature → SSD 기록), flush_cache.

    인스턴스 수명: 학습 스크립트 시작 시 1회 생성되어 모든 epoch 에서 재사용.
    상위 호출자: 학습 스크립트와 _PrefetchingIter.__next__.
    하위 의존: BAM_Feature_Store (pybind), torch, time.
    """
    def __init__(self, page_size=4096, off=0, cache_dim = 1024, num_ele = 300*1000*1000*1024,
        num_ssd = 1,  ssd_list = None, cache_size = 10,
        ctrl_idx=0,
        window_buffer=False, wb_size = 8,
        accumulator_flag = False,
        long_type=False,
        heterograph=False,
        heterograph_map=None):
        """
        [한국어]
        GIDS 초기화.

        @param page_size: BaM 페이지 크기(bytes). NVMe LBA 의 정수배여야 하며 일반적으로 4096.
        @param off: 전체 feature 영역이 시작되는 byte offset. 내부적으로 ceil(ceil(off/page_size)/num_ssd)
          로 "SSD당 페이지 인덱스" 로 정규화됨 (striping 시 SSD별 오프셋).
        @param cache_dim: feature 차원. (C++ 쪽에서 read 단위 계산 시 사용)
        @param num_ele: SSD에 기록된 feature 원소의 총 개수. 기본값은 300G 규모.
        @param num_ssd: 사용할 SSD 개수.
        @param ssd_list: /dev/libnvm{i} 인덱스 리스트. None 이면 [0..num_ssd-1] 자동 생성.
        @param cache_size: BaM GPU 페이지 캐시 크기(MB).
        @param ctrl_idx: CUDA 디바이스 인덱스(gids_device = 'cuda:<ctrl_idx>').
        @param window_buffer: Window Buffering 활성화 여부.
        @param wb_size: Window Buffer에 미리 채워둘 배치 수.
        @param accumulator_flag: Storage Access Accumulator(여러 배치 병합 fetch) 모드.
        @param long_type: feature dtype 이 int64 이면 True → _long 바인딩 사용.
        @param heterograph: 이종 그래프 여부. True 면 배치[0] 가 node_type→tensor dict.
        @param heterograph_map: node_type → SSD 내 byte offset(key_off) 매핑 dict.
        실행 컨텍스트: 메인 프로세스, 학습 초기.
        """

        #self.sample_type = "LADIES"

        # [한국어] feature dtype 에 따라 pybind 서브클래스 선택.
        #  - _long  ↔ C++ BAM_Feature_Store<int64_t> (gids_nvme.cu 의 PYBIND11_MODULE 에서 바인딩됨)
        #  - _float ↔ C++ BAM_Feature_Store<float>
        #  여기서 생성된 객체는 이후 모든 read_feature*/store_tensor 호출의 주체이다.
        if(long_type):
            self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store_long()
        else:
            self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store_float()

        # CPU Buffer and Storage Access Accumulator Metadata
        # [한국어] accumulator_flag: True 이면 fetch_feature 가 여러 배치를 모아 한 번에 read_feature_merged 호출.
        #  설정자: 여기 1회. 읽는 자: fetch_feature. 동기화: 단일 스레드.
        self.accumulator_flag = accumulator_flag
        # [한국어] Accumulator 모드에서 한 번에 묶을 총 storage access 수(분석적으로 계산된 임계값).
        #  설정자: set_required_storage_access + fetch_feature 루프. 값 범위: >=0.
        self.required_accesses = 0
        # [한국어] 직전 루프에서 CPU 버퍼로 처리된 접근 수(장치→호스트 카운터 get_cpu_access_count/num_iter).
        #  임계값 계산 시 required_accesses 에 더해져 "CPU 버퍼 히트분만큼 더 허용" 으로 동작.
        self.prev_cpu_access = 0
        # [한국어] 미리 fetch 해둔 feature 텐서들의 큐(FIFO). pop(0) 하여 각 배치에 붙여 반환.
        #  설정자: fetch_feature 내부. 읽는 자: 같음. 동기화: 단일 스레드.
        self.return_torch_buffer = []
        # [한국어] (미사용) 향후 확장을 위한 인덱스 목록 보관용 필드.
        self.index_list = []


        # Window Buffering MetaData
        # [한국어] Window Buffering 활성 플래그. 설정자: 여기. 읽는 자: fetch_feature.
        self.window_buffering_flag = window_buffer
        # [한국어] 다음 n배치의 "샘플링된 배치" 자체를 담는 윈도우 큐. pop(0) 순으로 반환.
        self.window_buffer = []
        # [한국어] Window Buffer 가 초기 wb_size 만큼 채워졌는지 여부(한 번만 True 로 전환됨).
        self.wb_init = False
        # [한국어] 윈도우 크기. 설정자: 여기. 읽는 자: fill_wb.
        self.wb_size = wb_size

        # Cache Parameters
        # [한국어] BaM 페이지 크기. 설정자: 여기. 읽는 자: set_required_storage_access 등.
        self.page_size = page_size
        # [한국어] SSD 시작 offset 을 페이지 단위로 올림 → SSD 수로 나눠 "SSD당 페이지 인덱스" 로 변환.
        #  이후 init_controllers 의 3번째 인자로 전달되어 gids_nvme.cu 의 range_t 의 read_off 로 쓰인다.
        self.off = math.ceil(math.ceil(off / page_size)/num_ssd)
        # [한국어] 전체 feature 원소 개수. init_controllers 에 전달 → range_t 의 n_elems.
        self.num_ele = num_ele
        # [한국어] GPU 페이지 캐시 크기(MB). init_controllers 에 전달 → page_cache_t 가 내부에서 bytes 로 환산.
        self.cache_size = cache_size

        #True if the graph is heterogenous graph
        # [한국어] 이종 그래프 여부. 설정자: 여기. 읽는 자: window_buffering, fetch_feature.
        self.heterograph = heterograph
        # [한국어] node_type → SSD byte offset(key_off) 매핑. fetch_feature 에서 read_feature_hetero 에 전달.
        self.heterograph_map = heterograph_map
        # [한국어] 샘플링 그래프 토폴로지 전용 보조 GIDS (int64 인덱스) 인스턴스 핸들. init_graph_GIDS 에서 생성.
        self.graph_GIDS = None

        # [한국어] feature dim. fetch_feature 에서 read_feature 의 cache_dim 인자로 그대로 전달.
        self.cache_dim = cache_dim
        # [한국어] 인덱스/반환 텐서가 살아야 하는 CUDA 디바이스 문자열. ctrl_idx 로 구성.
        self.gids_device="cuda:" + str(ctrl_idx)


        # [한국어] BaM Controller 벡터 관리자 생성 (C++ 측 GIDS_Controllers pybind 바인딩).
        #  - 초기에는 컨트롤러가 비어 있고, 아래 init_GIDS_controllers 호출로 실제 /dev/libnvm{i} 를 오픈.
        self.GIDS_controller = BAM_Feature_Store.GIDS_Controllers()

        # [한국어] ssd_list 기본값 처리: 미지정 시 0..num_ssd-1 순서로 사용.
        if (ssd_list == None):
            print("SSD are not assigned")
            self.ssd_list = [i for i in range(num_ssd)]
        else:
            self.ssd_list = ssd_list

        # [한국어] 디버그 출력: 어떤 SSD 리스트로 striping 될지 확인.
        print("ssd list: ", ssd_list)
        # [한국어] BaM Controller 초기화. gids_module/gids_nvme.cu 의 GIDS_Controllers::init_GIDS_controllers 로 진입.
        #  - 1024: numQueues(I/O 큐 개수), 128: queueDepth(큐 당 SQE 수). 이 값들은 BaM Controller 생성자 파라미터로
        #    들어가 GPU-resident SQ/CQ 할당 크기를 결정한다.
        #  - self.ssd_list 에 명시된 /dev/libnvm{i} 마다 Controller 객체를 생성해 vector에 push.
        self.GIDS_controller.init_GIDS_controllers(num_ssd, 1024, 128, self.ssd_list)
        # [한국어] feature store 초기화. gids_nvme.cu 의 BAM_Feature_Store<T>::init_controllers 로 진입.
        #  내부 순서: page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0], 64, ctrls)
        #             → range_t<T>(0, num_ele, off, n_pages, 0, page_size, h_pc, dev, STRIPE)
        #             → array_t<T>(...) 를 통해 GPU 커널이 볼 array_d_t 뷰 확보.
        self.BAM_FS.init_controllers(self.GIDS_controller, page_size, self.off, cache_size,num_ele, num_ssd)

        # [한국어] fetch_feature 누적 시간(초). print_stats 에서 출력 후 리셋.
        self.GIDS_time = 0.0
        # [한국어] Window Buffering 누적 시간(초).
        self.WB_time = 0.0




    # For Sampling GIDS operation
    def init_graph_GIDS(self, page_size, off, cache_size, num_ele, num_ssd):
        """
        [한국어]
        init_graph_GIDS - 그래프 토폴로지(CSC/CSR 인덱스) 자체를 SSD에서 읽기 위한 보조 GIDS 초기화.

        feature 와 다르게 그래프 토폴로지는 int64 인덱스 배열이므로 항상 BAM_Feature_Store_long 을 쓴다.
        같은 GIDS_Controllers(=동일 SSD 세트)를 공유하지만 별도의 range_t/array_t 를 갖는다.

        @param page_size/off/cache_size/num_ele/num_ssd: feature 쪽 초기화와 동일 의미.
        호출자: 학습 스크립트 또는 샘플러 래퍼. 하위 호출: graph_GIDS.init_controllers.
        """
        # [한국어] int64 인덱스를 다루므로 long 바인딩을 선택.
        self.graph_GIDS = BAM_Feature_Store.BAM_Feature_Store_long()
        # [한국어] feature 경로와 동일한 Controller 를 재사용 → SSD 오픈을 중복하지 않는다.
        self.graph_GIDS.init_controllers(self.GIDS_controller,page_size, off, cache_size, num_ele, num_ssd)

    def get_offset_array(self):
        """
        [한국어] graph_GIDS 가 관리하는 3-tuple 오프셋 배열(샘플링 시 행/열/값 offset) 포인터 반환.
         호출자: 커스텀 샘플러(예: ladies_sampler). 하위: graph_GIDS.get_offset_array (C++ pybind).
        """
        ret = self.graph_GIDS.get_offset_array()
        return ret

    def get_array_ptr(self):
        """
        [한국어] graph_GIDS 의 array_t<int64_t>* 를 raw 포인터로 반환. CUDA 커널에 직접 넘길 용도.
        """
        return self.graph_GIDS.get_array_ptr()

    # For static CPU feature buffer
    def cpu_backing_buffer(self, dim, length):
        """
        [한국어]
        Constant CPU Buffer 슬롯 할당. gids_nvme.cu 의 BAM_Feature_Store::cpu_backing_buffer 로 진입하여
        cudaHostAlloc(cudaHostAllocMapped) 로 length×dim 크기 pinned 호스트 버퍼를 확보한다.
        이후 set_cpu_buffer 가 이 슬롯에 어느 노드를 매핑할지 결정한다.

        @param dim: feature 차원.
        @param length: 최대 CPU 버퍼 슬롯 수(=핫 노드 개수).
        실행 컨텍스트: 메인 프로세스, 학습 시작 전 1회.
        """
        self.BAM_FS.cpu_backing_buffer(dim, length)

    def set_cpu_buffer(self, ten, N):
        """
        [한국어]
        set_cpu_buffer - 상위 N개 핫 노드 ID 를 CPU 버퍼 슬롯 0..N-1 에 매핑한다.

        @param ten: PageRank 등으로 정렬된 (노드ID) 텐서(CPU 또는 GPU).
        @param N: 상위 몇 개를 CPU 버퍼에 올릴지.

        내부적으로 (1) 앞 N개 슬라이스 → (2) GPU로 복사 → (3) BAM_FS.set_cpu_buffer 로 포인터 전달.
        C++ 측은 range_t::set_cpu_buffer 로 node→slot 매핑을 구축하고, 이후 GPU 커널에서
        range_t::get_cpu_offset(row) 가 slot 인덱스를 1비트 마커와 함께 반환한다.
        """
        # [한국어] 텐서 앞 N개만 사용.
        topk_ten = ten[:N]
        # [한국어] 실제 길이 (N이 ten 의 길이보다 클 수 있으므로 슬라이스 후 재측정).
        topk_len = len(topk_ten)
        # [한국어] GPU로 전송 — C++ 쪽에서 GPU 커널로 매핑 구축 시 디바이스 포인터가 필요.
        d_ten = topk_ten.to(self.gids_device)
        # [한국어] pybind 호출: data_ptr() 은 __int__ 인데 C++ 에서 uintptr_t 로 수신.
        self.BAM_FS.set_cpu_buffer(d_ten.data_ptr(), topk_len)

    # Window Buffering
    def window_buffering(self, batch):
        """
        [한국어]
        window_buffering - 주어진 샘플링 배치의 입력 노드 ID들을 BaM 페이지 캐시에 선행 프리패치.
        내부적으로 gids_nvme.cu 의 BAM_Feature_Store::set_window_buffering 으로 진입하여
        set_window_buffering_kernel(CUDA) 을 런칭한다.

        @param batch: DGL 샘플러가 만든 배치. batch[0] 은 (이종) node ID 텐서(또는 dict).
        실행 컨텍스트: 메인 프로세스. 호출자: fill_wb, fetch_feature.
        """
        # [한국어] 타이밍 시작.
        s_time = time.time()
        # [한국어] 이종 그래프: 각 node type 별로 별도 key_off 적용.
        if(self.heterograph):
             for key, value in batch[0].items():
                if(len(value) == 0):
                    # [한국어] 빈 텐서 타입은 스킵.
                    next
                else:
                    s_time = time.time()
                    # [한국어] 노드 ID를 GPU로 복사.
                    input_tensor = value.to(self.gids_device)
                    # [한국어] node type 별 SSD 내 byte offset. 기본 0.
                    key_off = 0
                    if(self.heterograph_map != None):
                        if (key in self.heterograph_map):
                            key_off = self.heterograph_map[key]
                        else:
                            # [한국어] 매핑 누락 경고. 이 경우 key_off=0 으로 진행 → 잘못된 SSD 영역을 읽을 수 있음.
                            print("Cannot find key: ", key, " in the heterograph map!")

                    # [한국어] 프리패치할 노드 수(=커널 블록 수의 기준).
                    num_pages = len(input_tensor)
                    # [한국어] pybind 호출: GPU에서 set_window_buffering_kernel 런칭. 각 warp가 노드 1개를 담당해
                    #  페이지 캐시에 해당 페이지를 로드한다.
                    self.BAM_FS.set_window_buffering(input_tensor.data_ptr(), num_pages, key_off)
                    e_time = time.time()
                    # [한국어] WB 누적 시간.
                    self.WB_time += e_time - s_time

        else:
            # [한국어] 동종 그래프 경로: key_off=0 으로 한 번만 호출.
            input_tensor = batch[0].to(self.gids_device)
            num_pages = len(input_tensor)
            self.BAM_FS.set_window_buffering(input_tensor.data_ptr(), num_pages, 0)
            e_time = time.time()
            self.WB_time += e_time - s_time


    # Window Buffering Helper Function
    def fill_wb(self, it, num):
        """
        [한국어]
        fill_wb - Window Buffer 초기 채우기. 샘플러 iterator 에서 num 개 배치를 먼저 뽑아
        self.window_buffer 에 쌓고 각 배치에 대해 window_buffering(=프리패치)을 실행.

        @param it: torch DataLoader iterator (_PrefetchingIter 가 넘겨준 dataloader_it).
        @param num: 사전에 채울 배치 수(=self.wb_size).
        호출자: fetch_feature 의 초기 1회.
        """
        for i in range(num):
            # [한국어] 배치 하나 샘플링.
            batch = next(it)
            # [한국어] 윈도우 큐에 저장 → 나중에 fetch_feature 가 pop(0) 으로 꺼낸다.
            self.window_buffer.append(batch)
            #run window buffering for the current batch
            # [한국어] BaM 페이지 캐시 선행 프리패치.
            self.window_buffering(batch)


    # BW in GB/s, latency in micro seconds
    def set_required_storage_access(self, bw, l_ssd, l_system, num_ssd, p):
        """
        [한국어]
        set_required_storage_access - Storage Access Accumulator 임계값을 분석식으로 계산해 저장.

        공식: accesses = (p * bw_per_sec_in_pages * (l_ssd + l_system) * num_ssd) / (1 - p)
          - bw: SSD 초당 대역폭(GB/s). 내부에서 bw*1024/page_size 로 "초당 페이지" 변환.
          - l_ssd, l_system: us 단위 레이턴시. 합해서 한 요청의 end-to-end 지연.
          - num_ssd: SSD 수 (병렬도).
          - p: 캐시 미스 비율(0~1).
        의미: 한 iteration 동안 파이프라인을 가득 채우기 위해 필요한 동시 접속 수.
        """
        accesses = (p * bw * 1024 / self.page_size * (l_ssd + l_system) * num_ssd) / (1-p)
        # [한국어] 결과를 self 에 보관. 이후 fetch_feature 의 누적 루프 조건으로 쓰임.
        self.required_accesses = accesses
        print("Number of required storage accesses: ", accesses)

    #Fetching Data from the SSDs
    def fetch_feature(self, dim, it, device):
        """
        [한국어]
        fetch_feature - 한 학습 iteration 의 (샘플 배치 + feature 텐서) 를 만들어 반환한다.
        GIDS의 모든 최적화 경로(Window Buffering / Storage Access Accumulator / CPU Buffer)가 여기서 분기된다.

        @param dim: feature 차원. return_torch shape = [index_size, dim].
        @param it: 내부 torch DataLoader iterator (샘플러 출력).
        @param device: 타깃 CUDA 디바이스 (self.gids_device 와 동일한 값이 전달됨).
        @return: 원본 배치 + feature 텐서(또는 dict) 가 합쳐진 tuple/list.

        호출자: _PrefetchingIter.__next__. 하위: next(it), BAM_FS.read_feature*, window_buffering.
        실행 컨텍스트: 메인 프로세스의 학습 스레드. BAM_FS 호출은 내부적으로 GPU 커널을 런칭하며 동기화 포함.

        호출 체인:
          _PrefetchingIter.__next__ → [fetch_feature]
            → self.fill_wb / window_buffering → BAM_FS.set_window_buffering
            → next(it) → CollateWrapper → graph_sampler.sample
            → BAM_FS.read_feature / read_feature_hetero / read_feature_merged / _merged_hetero
              → gids_nvme.cu 의 read_feature* → gids_kernel.cu 의 __global__ 커널
                → bam_ptr.read() → BaM page cache (hit) 또는 NVMe SQE 제출 (miss)
        """
        # [한국어] 전체 fetch 경로 타이머 시작.
        GIDS_time_start = time.time()

        if(self.window_buffering_flag):
            #Filling up the window buffer
            # [한국어] 최초 호출에서만 wb_size 개 배치를 선행 fetch/프리패치.
            if(self.wb_init == False):
                self.fill_wb(it, self.wb_size)
                self.wb_init = True

        #print("Sample  start")
        # [한국어] 이번 iteration의 "다음" 배치 하나 샘플링 (윈도우 큐에 추가용).
        next_batch = next(it)
        #print("Sample  done")

        # [한국어] 샘플링된 배치를 윈도우 큐 맨 뒤에 push.
        self.window_buffer.append(next_batch)
        #Update Counters for Windwo Buffering
        # [한국어] 윈도우 버퍼링이 켜져 있으면, 새로 들어온 배치에 대해서도 페이지 캐시 프리패치 실행.
        if(self.window_buffering_flag):
            self.window_buffering(next_batch)

        # When the Storage Access Accumulator is enabled
        if(self.accumulator_flag):
            # [한국어] 여러 배치 요청을 누적 후 한 번에 read_feature_merged 로 SSD 접근량을 합치는 경로.
            index_size_list = []   # [한국어] 각 배치의 노드 수 배열. C++ 에 vector<int> 로 전달.
            index_ptr_list = []    # [한국어] 각 배치의 노드 ID 텐서 data_ptr 배열 (GPU 디바이스 포인터).
            return_torch_list = [] # [한국어] 각 배치에 대응하는 return_torch data_ptr 배열.
            key_list = []          # [한국어] 이종 그래프에서 node type 별 key_off 배열 (동종이면 사용 안 함).

            # [한국어] 이미 merge fetch가 끝나 캐싱된 결과가 있으면 바로 꺼내서 반환 (여러 배치 중 첫 번째가 여기서 반환됨).
            if(len(self.return_torch_buffer) != 0):
                return_ten = self.return_torch_buffer.pop(0)
                return_batch = self.window_buffer.pop(0)
                return_batch.append(return_ten)
                self.GIDS_time += time.time() - GIDS_time_start
                return return_batch

            # [한국어] 현재 윈도우 큐 길이 — 앞으로 얼마나 많은 배치를 누적 할지 계산할 때 상한으로 사용.
            buffer_size = len(self.window_buffer)
            current_access = 0
            num_iter = 0
            # [한국어] 지역 required_accesses 에 prev_cpu_access(CPU 버퍼 히트 분) 를 더하면서 적정 accum 사이즈 계산.
            required_accesses = self.required_accesses


            if(self.heterograph):
                # [한국어] 이종 그래프: batch[0] 이 dict(type→tensor).
                while(1):
                    if(num_iter >= buffer_size):
                        # [한국어] 윈도우 큐를 다 소진하면 더 뽑아 채움.
                        batch = next(it)
                        for k , v in batch[0].items():
                            # [한국어] type 별 노드 수를 합산해 총 접근량 추정.
                            current_access += len(v)

                        self.window_buffer.append(batch)
                        if(self.window_buffering_flag):
                            # [한국어] 새로 받은 배치도 페이지 캐시 프리패치.
                            self.window_buffering(batch)

                    else:
                        batch = self.window_buffer[num_iter]
                        for k , v in batch[0].items():
                            current_access += len(v)

                    num_iter +=1
                    # [한국어] 임계값 누적 — CPU 버퍼 히트를 고려해 required 증가.
                    required_accesses += self.prev_cpu_access
                    if(current_access > (required_accesses )):
                        # [한국어] 임계값 초과 시 누적 종료.
                        break

                num_concurrent_iter = 0
                # [한국어] 확정된 num_iter 개 배치에 대해 read_feature_merged_hetero 호출 인자 구성.
                for i in range(num_iter):
                    batch = self.window_buffer[i]
                    ret_ten = {}
                    for k , v in batch[0].items():
                        if(len(v) == 0):
                            # [한국어] 빈 type: shape [0, dim] 더미 텐서를 만들어 반환 dict 에 삽입.
                            empty_t = torch.empty((0, dim)).to(self.gids_device)
                            ret_ten[k] = empty_t
                        else:
                            key_off = 0
                            if(self.heterograph_map != None):
                                if (k in self.heterograph_map):
                                    key_off = self.heterograph_map[k]
                                else:
                                    print("Cannot find key: ", k, " in the heterograph map!")
                            # [한국어] GPU로 인덱스 전송.
                            v = v.to(self.gids_device)
                            index_size = len(v)
                            index_size_list.append(index_size)
                            # [한국어] 결과 텐서: [index_size, dim] float32. contiguous 보장을 위해 zeros 로 생성.
                            return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device)
                            index_ptr_list.append(v.data_ptr())
                            ret_ten[k] = return_torch
                            return_torch_list.append(return_torch.data_ptr())
                            key_list.append(key_off)
                            num_concurrent_iter += 1
                    # [한국어] 이 배치에 해당하는 반환 dict 를 결과 큐에 저장 (순서 중요).
                    self.return_torch_buffer.append(ret_ten)
                # [한국어] pybind 호출: gids_nvme.cu 의 read_feature_merged_hetero 로 진입 →
                #  여러 배치/타입의 read 요청을 한 커널 런칭으로 병렬 수행. GPU 큐 재사용 이득.
                self.BAM_FS.read_feature_merged_hetero(num_concurrent_iter, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim, key_list)

                # [한국어] 즉시 현재 iteration 분만 꺼내 반환.
                return_ten = self.return_torch_buffer.pop(0)
                return_b = self.window_buffer.pop(0)
                # [한국어] 원본 배치가 tuple 이면 feature dict 를 tuple 에 추가, 아니면 list 로 append.
                if type(return_b) is tuple:
                    return_batch = (*return_b, return_ten)
                else:
                    return_batch = return_b
                    return_batch.append(return_ten)
                self.GIDS_time += time.time() - GIDS_time_start

                # [한국어] 디바이스 측 CPU 버퍼 히트 카운터를 호스트로 읽어와 평균을 기록 → 다음 누적 임계값에 반영.
                cpu_access_count = self.BAM_FS.get_cpu_access_count()
                self.prev_cpu_access = int(cpu_access_count / num_iter)
                # [한국어] 다음 루프를 위해 디바이스 카운터를 0으로 초기화.
                self.BAM_FS.flush_cpu_access_count()

                return return_batch
            else:
                # [한국어] 동종 그래프 accumulator 경로.
                while(1):
                    if(num_iter >= buffer_size):
                        batch = next(it)
                        current_access += len(batch[0])
                        self.window_buffer.append(batch)
                        if(self.window_buffering_flag):
                            self.window_buffering(batch)
                    else:
                        batch = self.window_buffer[num_iter]
                        current_access += len(batch[0])
                    num_iter +=1
                    required_accesses += self.prev_cpu_access
                    if(current_access > (required_accesses )):
                        break

                # [한국어] merge fetch 대상 배치들의 인덱스/반환 포인터 모으기.
                for i in range(num_iter):
                    batch = self.window_buffer[i]
                    index = batch[0].to(self.gids_device)
                    index_size = len(index)
                    index_size_list.append(index_size)
                    return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device)
                    index_ptr_list.append(index.data_ptr())
                    return_torch_list.append(return_torch.data_ptr())
                    self.return_torch_buffer.append(return_torch)

                # [한국어] pybind → gids_nvme.cu::read_feature_merged → gids_kernel.cu 의 read_feature_kernel*
                #  여러 배치가 하나의 launch 에 묶여 BaM 페이지 캐시를 공유한다.
                self.BAM_FS.read_feature_merged(num_iter, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim)
                return_ten = self.return_torch_buffer.pop(0)
                return_b = self.window_buffer.pop(0)
                if type(return_b) is tuple:
                    return_batch = (*return_b, return_ten)
                else:
                    return_batch = return_b
                    return_batch.append(return_ten)


                self.GIDS_time += time.time() - GIDS_time_start

                cpu_access_count = self.BAM_FS.get_cpu_access_count()
                self.prev_cpu_access = int(cpu_access_count / num_iter)
                self.BAM_FS.flush_cpu_access_count()

                return return_batch

        # Storage Access Accumulator is disabled
        else:
            # [한국어] 단일 배치 fetch 경로. 매 iteration 마다 1회 read_feature* 호출.
            if(self.heterograph):
                # [한국어] 이종 그래프: window_buffer 에서 가장 오래된 배치 1개 꺼내 type별로 fetch.
                batch = self.window_buffer.pop(0)
                ret_ten = {}
                index_size_list = []
                index_ptr_list = []
                return_torch_list = []
                key_list = []

                num_keys = 0
                for key , v in batch[0].items():
                    if(len(v) == 0):
                        # [한국어] 빈 type → empty (0,dim) 더미.
                        empty_t = torch.empty((0, dim)).to(self.gids_device).contiguous()
                        ret_ten[key] = empty_t
                    else:
                        key_off = 0
                        if(self.heterograph_map != None):
                            if (key in self.heterograph_map):
                                key_off = self.heterograph_map[key]
                            else:
                                print("Cannot find key: ", key, " in the heterograph map!")

                        # [한국어] GPU 인덱스 / 반환 텐서 준비.
                        g_index = v.to(self.gids_device)
                        index_size = len(g_index)
                        index_ptr = g_index.data_ptr()

                        return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
                        return_torch_list.append(return_torch.data_ptr())
                        ret_ten[key] = return_torch
                        num_keys += 1
                        index_ptr_list.append(index_ptr)
                        index_size_list.append(index_size)
                        key_list.append(key_off)

                # [한국어] pybind → read_feature_hetero (gids_nvme.cu) → read_feature_kernel* 런칭.
                self.BAM_FS.read_feature_hetero(num_keys, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim, key_list)

                self.GIDS_time += time.time() - GIDS_time_start
                if type(batch) is tuple:
                    batch2 = (*batch, ret_ten)
                    return batch2
                else:
                    batch.append(ret_ten)
                    return batch

            else:
                # [한국어] 동종 그래프 단일 fetch.
                batch = self.window_buffer.pop(0)
                #print("batch 0: ", batch.ndata['_ID'])
                # [한국어] 배치의 입력 노드 ID 텐서를 GPU로 이동.
                index = batch[0].to(self.gids_device)
                index_size = len(index)
                #print(batch[0])
                index_ptr = index.data_ptr()
                # [한국어] [index_size, dim] float32 결과 텐서 준비(contiguous).
                return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
                # [한국어] pybind → gids_nvme.cu::read_feature → gids_kernel.cu::read_feature_kernel 또는
                #  read_feature_kernel_with_cpu_backing (CPU 버퍼 hybrid 경로).
                #  마지막 인자 0 = key_off (동종 그래프이므로 0).
                self.BAM_FS.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
                self.GIDS_time += time.time() - GIDS_time_start

                if type(batch) is tuple:
                    batch2 = (*batch, return_torch)
                    return batch2
                else:
                    batch.append(return_torch)
                    return batch



    def print_stats(self):
        """
        [한국어] 학습 단계별 누적 시간과 BAM_FS/graph_GIDS 의 캐시 통계를 출력하고, 타이머를 리셋.
        호출자: 학습 스크립트의 epoch 종료부 또는 GIDS_DGLDataLoader.print_stats.
        하위: BAM_FS.print_stats (컨트롤러 포함), graph_GIDS.print_stats_no_ctrl (컨트롤러 제외).
        """
        print("GIDS time: ", self.GIDS_time)
        wbtime = self.WB_time
        print("WB time: ", wbtime)
        # [한국어] 타이머 리셋 — 다음 epoch 측정이 누적되지 않도록.
        self.WB_time = 0.0
        self.GIDS_time = 0.0
        # [한국어] C++ 측에서 관리되는 캐시 hit/miss/throughput 등 출력.
        self.BAM_FS.print_stats()

        if (self.graph_GIDS != None):
            # [한국어] 같은 Controller 를 공유하므로 컨트롤러 통계 중복 출력을 피하기 위해 _no_ctrl 버전 사용.
            self.graph_GIDS.print_stats_no_ctrl()
        return

    # Utility FUnctions
    def store_tensor(self, in_ten, offset):
        """
        [한국어]
        store_tensor - 주어진 텐서를 SSD에 offset 페이지부터 순차 기록.
        gids_nvme.cu 의 BAM_Feature_Store::store_tensor 로 진입 → GPU 커널에서 write SQE 발행.

        @param in_ten: torch 텐서 (CPU 또는 GPU). data_ptr 이 유효한 선형 메모리여야 한다.
        @param offset: SSD 내 시작 페이지 인덱스.
        """
        num_e = len(in_ten)
        self.BAM_FS.store_tensor(in_ten.data_ptr(),num_e,offset)

    def store_mmap_tensor(self, in_ten, offset):
        """
        [한국어]
        store_mmap_tensor - numpy memmap 어레이를 SSD에 기록. memmap 이 그대로 pybind 로 넘어가면
        lazy 로딩 이슈가 있어서 .copy() 로 연속 메모리를 강제한 뒤 ctypes 로 원시 포인터를 넘긴다.

        @param in_ten: numpy.memmap (혹은 ndarray).
        @param offset: SSD 내 시작 페이지 인덱스.
        """

        #y = in_ten[:200000].copy()
        # [한국어] 전체 복사 — 연속 메모리 블록 확보(디버그 print 포함).
        y = in_ten.copy()
        print(y)
        print(y.flags)
        # for i in range(100):
        #     print("Tensor val: ", y[i])

        num_e = len(y)
        # [한국어] ctypes.data = void* 포인터(uintptr_t). pybind 서명이 uint64_t 포인터이므로 그대로 전달 가능.
        print("num ele: ", num_e, " ptr: ", y.ctypes.data)
        self.BAM_FS.store_tensor(y.ctypes.data,num_e,offset)

    def read_tensor(self, num, offset):
        """
        [한국어] offset 페이지부터 num 개 원소를 SSD에서 읽어 C++ 내부 버퍼로 로드(디버그용).
        """
        self.BAM_FS.read_tensor(num, offset)

    def flush_cache(self):
        """
        [한국어] BaM 페이지 캐시를 무효화한다. epoch 경계에서 "워밍업 효과 없이" 측정하고 싶을 때 사용.
        """
        self.BAM_FS.flush_cache()


