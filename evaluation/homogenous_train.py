"""
[한국어 설명] 동종(homogeneous) 그래프용 GIDS+BaM 학습 스크립트 (homogenous_train.py)

=== 파일의 역할 ===
이 파일은 IGB260M/OGB와 같이 노드 타입이 하나뿐인 "동종 그래프"에 대한 GNN
(Graph Neural Network: 그래프 신경망) 학습 엔트리포인트이다. DGL
(Deep Graph Library)의 MultiLayerNeighborSampler로 mini-batch를 만든 뒤,
`GIDS_DGLDataLoader` 래퍼를 통해 GPU가 직접 NVMe SSD에서 feature를 읽어오도록
한다(= GPU-Initiated Direct Storage, GIDS + BaM 경로). 기존 CPU → GPU
memcpy 병목을 제거하여 테라바이트 급 그래프에서도 학습이 가능하도록 하는 것이
이 스크립트의 핵심 목적이다. 학습 100 iter 성능 측정 후 return 하도록 되어 있어,
end-to-end 벤치마크 용도로도 쓰인다.

=== 전체 아키텍처에서의 위치 ===
프로젝트 전체 흐름도에서 "최상단 Python 학습 스크립트" 단계에 해당한다. 호출
체인: `python homogenous_train.py` → `__main__` → `track_acc_GIDS(g, args, device)`
→ `GIDS.GIDS(...)` (페이지 캐시/컨트롤러 초기화) → `GIDS_DGLDataLoader(...)` 반복
→ 내부에서 DGL sampler가 node id 배치 생성 → `fetch_feature`를 통해
`gids_module/gids_nvme.cu`의 `BAM_Feature_Store<T>::read_feature*()` 호출 →
`gids_kernel.cu`의 CUDA 커널이 bam_ptr.read() 로 SSD 접근. 실행 컨텍스트는 호스트
유저스페이스 Python 스레드이며, GPU 측은 CUDA 커널에서 NVMe SQE를 직접 제출한다.

=== 타 모듈과의 연결 ===
- `models.py`: GCN/SAGE/GAT 모델 클래스 제공.
- `dataloader.py`: IGB260MDGLDataset/OGBDGLDataset 그래프 로더 제공. 그래프 topology만
  들고 있고 feature는 SSD에 남아 있음.
- `GIDS_Setup/GIDS/GIDS.py`: `GIDS` 클래스(C++ `BAM_Feature_Store_{float,long}`
  pybind 래퍼), `GIDS_DGLDataLoader`(DGL DataLoader 서브클래스 + prefetch) 제공.
- 데이터 흐름: SSD(=feature 저장소) → BaM page_cache(GPU) → `ret` 텐서(device
  메모리) → model forward. 라벨(`labels`)은 그래프 객체 내부(호스트)에 있고 필요
  시 `.to(device)`로 복사된다.
- 공유 상태: `args`가 GIDS 하이퍼파라미터(page_size, num_ele, cache_size, wb_size,
  accumulator 등)를 C++ 측으로 전달하는 유일한 창구.

=== 주요 함수/구조체 요약 ===
- `fetch_data_chunk(test, out_t, page_size, stream_id)`: 디버그용 — 백킹 메모리
  청크 복사를 NVTX 구간으로 감싸 프로파일링에서 식별 가능하게 한다.
- `print_times(transfer_time, train_time, e2e_time)`: 단계별 시간 출력.
- `track_acc_GIDS(g, args, device, label_array)`: 본 스크립트의 메인 학습 루프.
  GIDS Loader 초기화 → CPU backing buffer 설정 → DataLoader 생성 → 모델 학습 →
  Evaluation.
- `__main__`: argparse 파싱 → 데이터셋 로드 → csc 포맷 변환 → `track_acc_GIDS` 호출.
"""

# [한국어] argparse: CLI 파라미터 파싱. datetime: 타임스탬프 로깅용(현재는 미사용이나 원본 유지).
import argparse, datetime
# [한국어] DGL(Deep Graph Library): NeighborSampler/DataLoader/graph format 변환을 제공하는 GNN 프레임워크.
import dgl
# [한국어] sklearn.metrics: test accuracy(accuracy_score) 계산에만 사용.
import sklearn.metrics
# [한국어] torch: 텐서/자동미분. nn: 모델 계층. optim: Adam 등 옵티마이저.
import torch, torch.nn as nn, torch.optim as optim
# [한국어] time: wall-clock 측정. tqdm: 진행바. numpy: test/label 배열 연결(concatenate)에 사용.
import time, tqdm, numpy as np
# [한국어] evaluation/models.py 의 GCN/SAGE/GAT 클래스를 와일드카드로 임포트.
from models import *
# [한국어] 동종 그래프 로더 — 둘 다 feature는 SSD에 두고 topology만 반환.
from dataloader import IGB260MDGLDataset, OGBDGLDataset
# [한국어] csv: 결과 로깅 유틸(현재 본문에서 직접 사용되진 않음).
import csv
# [한국어] warnings: 실험 시 노이즈 경고 억제를 위해 filterwarnings("ignore") 수행.
import warnings

# [한국어] torch.cuda.nvtx: PyTorch 측 NVTX 마커 주입용(현재는 미사용, 추후 프로파일링에서 활용).
import torch.cuda.nvtx as t_nvtx
# [한국어] nvtx: 함수 단위 NVTX 구간 데코레이터(@nvtx.annotate)를 제공, Nsight에서 가시화.
import nvtx
# [한국어] threading: CPU 프리페처 스레드 실험에 쓰이며 기본 경로에서는 비활성.
import threading
# [한국어] gc: 대형 그래프 로드 후 명시적 수거가 필요할 때를 대비해 임포트.
import gc

# [한국어] GIDS 패키지(`GIDS_Setup/GIDS/`): GIDS.GIDS(메타 설정 + BAM_Feature_Store pybind 객체)
# 와 GIDS_DGLDataLoader(DGL DataLoader 래퍼)를 제공. GIDS.GIDS.__init__ 내부에서
# gids_module/BAM_Feature_Store/bam_feature_store.*.so 를 dlopen하여 C++ 템플릿
# 인스턴스(BAM_Feature_Store_float / _long)를 바인딩한다.
import GIDS
# [한국어] GIDS_DGLDataLoader: DGL DataLoader를 상속하여, 각 iter마다 sampler가
# 만든 input_nodes를 받아 fetch_feature(BAM_Feature_Store::read_feature_float)로
# GPU에 feature를 직접 채워 ret 텐서로 반환한다.
from GIDS import GIDS_DGLDataLoader

# [한국어] OGB: Open Graph Benchmark 데이터셋 접근 API(grpah 레벨/노드 레벨).
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

# [한국어] 재현성을 위해 torch/dgl 시드 고정. sampler 내 random choice 포함.
torch.manual_seed(0)
dgl.seed(0)
# [한국어] DGL/OGB/DeprecationWarning 등 실험 로그를 가리기 위해 모든 경고 무시.
warnings.filterwarnings("ignore")


# [한국어]
# fetch_data_chunk - BAM_Feature_Store의 backing memory 청크를 out_t 텐서로 복사
#
# @param test: C++ BAM_Feature_Store pybind 객체(fetch_from_backing_memory_chunk 메서드 제공).
# @param out_t: 결과를 받을 torch 텐서(out_t.data_ptr() 은 device pointer).
# @param page_size: 한 번에 전송할 페이지 크기 — BaM page_cache_t 블록 단위.
# @param stream_id: CUDA stream 인덱스 — 비동기 전송 스케줄링용.
# @return: None(in-place 복사).
#
# 이 함수는 디버그/프로파일 편의용으로 NVTX "fetch_data_chunk()" 블루 마커로 감싸
# Nsight Systems에서 백킹 메모리 fetch 구간을 즉시 식별할 수 있게 한다. 호스트 측
# Python 스레드에서 실행되며 내부적으로 cudaMemcpyAsync 및 BaM 경로를 유발한다.
# 호출 체인: (실험 코드) → fetch_data_chunk → BAM_Feature_Store::fetch_from_backing_memory_chunk
#           → gids_nvme.cu 내부 CUDA stream 작업.
@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    # [한국어] C++ 측에 device pointer(int)와 size/stream을 전달. Python 측은 텐서 수명만 관리.
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)


# [한국어]
# print_times - 학습 루프 단계별 누적 시간 출력
#
# @param transfer_time: blocks/labels를 device로 복사하는 데 걸린 초 단위 누적.
# @param train_time: forward + backward + optimizer.step 누적.
# @param e2e_time: 측정 구간 end-to-end wall-clock.
# @return: None.
#
# warm-up 이후 100 iter를 측정한 뒤 한 번만 호출된다. 단순 print이므로
# 실행 컨텍스트/동기화 고려 사항은 없다.
def print_times(transfer_time, train_time, e2e_time):
    print("transfer time: ", transfer_time)  # [한국어] device로의 H2D 복사 누적
    print("train time: ", train_time)         # [한국어] 모델 학습 계산 누적
    print("e2e time: ", e2e_time)             # [한국어] 전체 구간 wall-clock

# [한국어]
# track_acc_GIDS - GIDS+BaM 경로로 동종 그래프 GNN을 학습하는 메인 함수
#
# @param g: DGL graph 객체(csc 포맷). feature는 SSD에 저장되어 있고 topology만 메모리에 있음.
# @param args: argparse 네임스페이스. page_size/num_ele/cache_size/wb_size/cpu_buffer/...
#              BaM 및 학습 하이퍼파라미터 전체를 담는 단일 전달체.
# @param device: "cuda:N" 문자열. 모델/배치 텐서의 targat device.
# @param label_array: OGB 계열에서 별도 label 텐서를 넘기는 용도(현재 경로는 None).
# @return: None(내부에서 100 iter 측정 후 조기 return).
#
# 단계: (1) GIDS.GIDS 객체 생성 — 내부에서 BAM_Feature_Store_float 인스턴스화 및
# BaM Controller/page_cache/range_t 초기화(gids_nvme.cu::init_controllers).
# (2) 옵션: cpu_backing_buffer + set_cpu_buffer — Page-Rank 등으로 고른 핫 노드를
# host-pinned memory에 고정하여 SSD 접근 없이 읽도록 설정.
# (3) DGL NeighborSampler + GIDS_DGLDataLoader 구성.
# (4) 모델 빌드(GCN/SAGE/GAT) → Adam 옵티마이저 → epoch 루프.
# (5) 각 step에서 ret(= GPU feature 버퍼) 수신 → blocks.to(device) → forward → loss →
#     backward → step. warm_up_iter(=100) 이후 100 iter를 측정하여 early return.
# 실행 컨텍스트: 단일 Python 프로세스, 단일 CUDA device. DataLoader num_workers>0이면
# 샘플링은 fork된 프로세스에서 수행.
#
# 호출 체인:
#   __main__ → track_acc_GIDS → GIDS.GIDS.__init__ → BAM_Feature_Store::init_controllers
#                             → GIDS_DGLDataLoader iter → fetch_feature (C++)
#                             → read_feature_kernel (gids_kernel.cu) → bam_ptr.read()
def track_acc_GIDS(g, args, device, label_array=None):

    # [한국어] === 단계 1: GIDS Loader 초기화 ===
    # GIDS_Loader를 None으로 먼저 선언하는 것은 원본 스타일(이후 재할당). 실제 생성
    # 시 아래 파라미터가 BAM_Feature_Store<float> 템플릿 인스턴스의 init_controllers()
    # 로 넘어가 BaM Controller/page_cache_t/range_t 를 모두 구성한다.
    GIDS_Loader = None
    GIDS_Loader = GIDS.GIDS(
        page_size = args.page_size,          # [한국어] BaM page_cache 의 페이지 크기(바이트 단위 power-of-two).
        off = args.offset,                   # [한국어] SSD 내 feature 데이터 시작 오프셋(바이트). tensor_write.py 로 기록한 위치.
        num_ele = args.num_ele,              # [한국어] 데이터셋 총 element 수 = total_bytes/sizeof(TYPE). range_t n_elems로 전달.
        num_ssd = args.num_ssd,              # [한국어] 스트라이핑에 사용할 NVMe SSD 개수 → Controller 벡터 크기와 일치.
        cache_size = args.cache_size,        # [한국어] GPU 페이지 캐시 크기(MB). n_pages = cache_size*1MB/page_size.
        window_buffer = args.window_buffer,  # [한국어] Window Buffering 프리패치 활성 플래그.
        wb_size = args.wb_size,              # [한국어] window buffering 에서 미리 당겨올 batch 수(예: 6).
        accumulator_flag = args.accumulator, # [한국어] Storage Access Accumulator(접근 스케줄링) 활성 플래그.
        cache_dim = args.cache_dim           # [한국어] feature 벡터 차원(예: 1024) — 커널 launch dim3에 반영.

    )
    dim = args.emb_size  # [한국어] embedding(=feature) 차원. 모델 in_feats 및 GIDS 내부 fetch 에 사용.

    # [한국어] Accumulator 사용 시 실제 필요 storage 대역폭 파라미터 전달.
    # 내부적으로 BAM_Feature_Store::set_required_storage_access 가 peak_percent 기반으로
    # 동시 in-flight request 수를 조정한다.
    if(args.accumulator):
        GIDS_Loader.set_required_storage_access(args.bw, args.l_ssd, args.l_system, args.num_ssd, args.peak_percent)


    # [한국어] === 단계 1-b: Constant CPU Buffer(핫 노드 pin) 설정 ===
    # 선택 사항. 학습 전체 기간 동안 자주 접근되는 노드의 feature를 host-pinned
    # memory에 고정해 두고, GPU 커널의 read_feature_kernel_with_cpu_backing 경로가
    # SSD 접근을 우회하도록 한다. 핀할 노드 개수는 전체 노드의 cpu_buffer_percent 비율.
    if(args.cpu_buffer):
        num_nodes = g.number_of_nodes()  # [한국어] 동종 그래프 전체 노드 수(heterograph 아님).
        num_pinned_nodes = int(num_nodes * args.cpu_buffer_percent)  # [한국어] e.g. 0.2 → 20% 핀.
        # [한국어] cpu_backing_buffer: C++ 측 GIDS_CPU_buffer<float> 를 cudaHostAlloc(Mapped)로 할당.
        GIDS_Loader.cpu_backing_buffer(dim, num_pinned_nodes)
        # [한국어] pin_file: 페이지랭크 등으로 정렬한 "상위 N개 노드 인덱스 텐서". page_rank_node_list_gen.py 생성물.
        pr_ten = torch.load(args.pin_file)
        # [한국어] set_cpu_buffer: range_t::set_cpu_buffer 로 노드→CPU슬롯 매핑을 기록.
        # 이후 bam_ptr.read() 가 get_cpu_offset 으로 hit 여부 판정.
        GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned_nodes)


    # [한국어] === 단계 2: DGL NeighborSampler ===
    # fan_out="10,15" 면 layer별 각 노드에서 10/15개 이웃을 샘플링. 리스트 길이가 GNN layer 수.
    # 결과는 MFG(block) 리스트로 반환되어 input_nodes(모든 src) / seeds(마지막 dst) / blocks 로 언팩.
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
               [int(fanout) for fanout in args.fan_out.split(',')]
               )

    # [한국어] DGL 관례에 맞춰 'feat' → 'features', 'label' → 'labels' 로 alias 부여. 실제 텐서는 동일 view.
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    # [한국어] === 단계 3: split mask 로부터 train/val/test node id 추출 ===
    # nonzero(..., as_tuple=True)[0] 은 마스크가 True인 node index 1D 텐서 반환.
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    # [한국어] in_feats: feature 차원. 모델의 첫 계층 입력 크기로 사용.
    in_feats = g.ndata['features'].shape[1]

    # [한국어] === 단계 4: GIDS DataLoader (학습용) ===
    # GIDS_DGLDataLoader 는 내부 _PrefetchingIter 를 통해 sampler가 만든 input_nodes
    # 집합을 GIDS_Loader.fetch_feature 로 전달 → BAM_Feature_Store::read_feature_float
    # → read_feature_kernel 에서 각 warp가 한 node를 담당하여 bam_ptr.read() 수행.
    # 반환되는 ret 은 이미 GPU 메모리 상의 [batch, dim] 텐서.
    train_dataloader = GIDS_DGLDataLoader(
        g,
        train_nid,
        sampler,
        args.batch_size,   # [한국어] seed node 개수. fetch 대상 input_nodes 는 이보다 훨씬 크다(fanout 확장).
        dim,               # [한국어] feature 차원 — ret 텐서의 두 번째 차원.
        GIDS_Loader,       # [한국어] BaM page_cache/Controller 를 보유한 핵심 객체.
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,       # [한국어] 0 이면 샘플링이 메인 프로세스에서 실행.
        use_alternate_streams=False         # [한국어] DGL 내부 stream 교대 비활성화 — GIDS 자체 stream 관리 사용.
        )

    # [한국어] validation/test DataLoader: 일반 DGL DataLoader. 본 스크립트 초기 100 iter 측정 모드에서는 사용되지 않을 수도 있음.
    val_dataloader = dgl.dataloading.DataLoader(
        g, val_nid, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, test_nid, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)

    # [한국어] === 단계 5: 모델 선택 ===
    # GCN: 기본. SAGE: neighbor aggregation + self. GAT: attention head 기반.
    # in_feats(=feature dim) → hidden_channels → num_classes 를 num_layers 만큼 쌓음.
    if args.model_type == 'gcn':
        model = GCN(in_feats, args.hidden_channels, args.num_classes,
            args.num_layers).to(device)  # [한국어] .to(device): 파라미터를 GPU로 이동.
    if args.model_type == 'sage':
        model = SAGE(in_feats, args.hidden_channels, args.num_classes,
            args.num_layers).to(device)
    if args.model_type == 'gat':
        model = GAT(in_feats, args.hidden_channels, args.num_classes,
            args.num_layers, args.num_heads).to(device)  # [한국어] GAT 만 num_heads 사용.

    # [한국어] 분류 문제 → CrossEntropyLoss. device에 올려 target label도 device에 있도록 맞춤.
    loss_fcn = nn.CrossEntropyLoss().to(device)
    # [한국어] Adam: SGD보다 수렴이 안정적. weight_decay=decay 는 L2 regularization.
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate, weight_decay=args.decay
        )

    # [한국어] warm_up_iter: 초반 100 iter 는 BaM cache warmup / CUDA context / Python import 지연 포함이므로
    # 통계에서 제외하기 위한 경계. warm_up + 100 까지 측정.
    warm_up_iter = 100
    # Setup is Done
    # [한국어] === 단계 6: epoch 루프 ===
    for epoch in tqdm.tqdm(range(args.epochs)):
        epoch_start = time.time()          # [한국어] 각 epoch 시작 시각(참고용).
        epoch_loss = 0                      # [한국어] 배치 loss 누적(학습 진단용).
        train_acc = 0                       # [한국어] 정확도 누적(미사용이나 원본 유지).
        model.train()                        # [한국어] dropout/bn training 모드 ON.

        batch_input_time = 0                # [한국어] feature fetch 시간 — GIDS 내부에서 기록되므로 여기선 placeholder.
        train_time = 0                      # [한국어] forward/backward 누적.
        transfer_time = 0                   # [한국어] blocks/labels → device 복사 누적.
        e2e_time = 0                        # [한국어] 구간 end-to-end 누적.
        e2e_time_start = time.time()        # [한국어] 구간 시작 시각.

        # [한국어] GIDS_DGLDataLoader iter: (input_nodes, seeds, blocks, ret) 4-tuple 반환.
        # ret 은 이미 GPU에 올라간 feature 텐서([len(input_nodes), dim]).
        for step, (input_nodes, seeds, blocks, ret) in enumerate(train_dataloader):
            #print("step: ", step)
            # [한국어] warm-up 경계 도달: 통계 리셋 및 print_stats/print_timer 로 BaM 캐시 hit/miss 출력.
            if(step == warm_up_iter):
                print("warp up done")
                train_dataloader.print_stats()   # [한국어] C++ 측 BAM_Feature_Store::print_stats — cache hit/miss 카운터.
                train_dataloader.print_timer()   # [한국어] fetch 시간 분해(커널 시간, sync 시간).
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                e2e_time_start = time.time()


            # Features are fetched by the baseline GIDS dataloader in ret

            # [한국어] --- 단계 6-a: feature는 이미 GPU 상의 ret 텐서에 있음 ---
            batch_inputs = ret
            transfer_start = time.time()

            # [한국어] 마지막 layer 의 dst 노드에 대한 정답 라벨. blocks[-1].dstdata 는 호스트 메모리.
            batch_labels = blocks[-1].dstdata['labels']


            # [한국어] --- 단계 6-b: blocks/labels 를 device로 이동 ---
            # block.int() 는 id dtype 을 int32 로 변경(모델 내부 scatter/gather 효율).
            blocks = [block.int().to(device) for block in blocks]
            batch_labels = batch_labels.to(device)
            transfer_time = transfer_time +  time.time()  - transfer_start

            # Model Training Stage
            # [한국어] --- 단계 6-c: forward / loss / backward / step ---
            train_start = time.time()
            batch_pred = model(blocks, batch_inputs)   # [한국어] MFG + feature 로 GNN forward.
            loss = loss_fcn(batch_pred, batch_labels)  # [한국어] cross-entropy 계산.
            optimizer.zero_grad()                       # [한국어] 이전 배치 gradient clear.
            loss.backward()                             # [한국어] autograd backward — GPU에서 수행.
            optimizer.step()                            # [한국어] Adam 파라미터 업데이트.
            epoch_loss += loss.detach()                 # [한국어] graph 분리 후 누적(history 미보존).
            train_time = train_time + time.time() - train_start

            # [한국어] --- 단계 6-d: 측정 구간 종료 ---
            if(step == warm_up_iter + 100):
                print("Performance for 100 iteration after 1000 iteration")
                e2e_time += time.time() - e2e_time_start
                train_dataloader.print_stats()          # [한국어] 100 iter 구간 후 최종 캐시 통계.
                train_dataloader.print_timer()
                print_times(transfer_time, train_time, e2e_time)

                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0

                #Just testing 100 iterations remove the next line if you do not want to halt
                # [한국어] 벤치마크 모드: 100 iter 측정 후 바로 종료. 실제 학습 시 이 return 제거.
                return None




    # Evaluation
    # [한국어] === 단계 7: Evaluation (현재 경로에서는 위 return 으로 인해 도달하지 않음) ===
    # 주의: test_dataloader 는 일반 DGL DataLoader 이므로 feature fetch 는 host-side 에서 수행됨.
    # 실제 production 평가에서는 test_dataloader 도 GIDS 경로로 교체하는 것이 일관적.

    model.eval()         # [한국어] dropout/bn eval 모드.
    predictions = []     # [한국어] 배치별 argmax 결과 축적.
    labels = []          # [한국어] 배치별 정답 축적.
    with torch.no_grad():  # [한국어] gradient 비활성 → 메모리 절약.
        for _, _, blocks in test_dataloader:
            blocks = [block.to(device) for block in blocks]   # [한국어] MFG device 이동.
            inputs = blocks[0].srcdata['feat']                 # [한국어] baseline: feat을 graph 에서 직접 사용(GIDS 경로 아님).

            if(args.data == 'IGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())  # [한국어] eval용 label → host numpy.
            elif(args.data == 'OGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())
                # out_label = torch.index_select(label_array, 0, b[1]).flatten()
                # labels.append(out_label.numpy())
            predict = model(blocks, inputs).argmax(1).cpu().numpy()  # [한국어] class argmax.
            predictions.append(predict)

        # [한국어] 전체 accuracy 계산.
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
    print("Test Acc {:.2f}%".format(test_acc))




# [한국어]
# __main__ 진입점 - CLI 인자 파싱, 데이터셋 로드, track_acc_GIDS 호출
# 실행 컨텍스트: 호스트 유저스페이스. GPU device 초기화는 model.to(device) 시점에서 lazy 발생.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    # [한국어] --path: IGB260M 또는 OGB 원본 topology 파일 디렉토리. feature 는 별도 SSD offset 에 있음.
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M',
        help='path containing the datasets')
    # [한국어] --dataset_size: 데이터셋 스케일. large/full 은 heterogeneous 쪽에서 주로 사용.
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    # [한국어] --num_classes: 출력 class 수. IGB 19 / IGBH paper 2983 / OGB-arxiv 172 등.
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983, 172], help='number of classes')
    # [한국어] --in_memory: 0=mmap 모드(기본, GIDS 대체), 1=메모리 전체 로드. GIDS 경로에선 0 권장.
    parser.add_argument('--in_memory', type=int, default=0,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    # [한국어] --synthetic: 1이면 IGB feature 대신 랜덤 텐서 생성(디버그용).
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    # [한국어] --data: 'IGB' 또는 'OGB' 분기 — 데이터셋 클래스 선택.
    parser.add_argument('--data', type=str, default='IGB')
    # [한국어] --emb_size: feature 차원. GIDS.GIDS(dim=…) 및 모델 in_feats.
    parser.add_argument('--emb_size', type=int, default=1024)

    # Model
    # [한국어] --model_type: GCN/SAGE/GAT 중 하나. track_acc_GIDS 내 분기에서 인스턴스화.
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gat', 'sage', 'gcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')  # [한국어] 체크포인트 경로 placeholder.
    parser.add_argument('--model_save', type=int, default=0)                 # [한국어] 1이면 학습 후 저장(현 경로 미사용).

    # Model parameters
    # [한국어] --fan_out: NeighborSampler 의 레이어별 이웃 수. "10,15" 면 2-layer(10, 15) 샘플링.
    parser.add_argument('--fan_out', type=str, default='10,15')
    # [한국어] --batch_size: seed node 배치 크기. fetch 대상 input_nodes 는 fan_out 에 따라 훨씬 커짐.
    parser.add_argument('--batch_size', type=int, default=1024)
    # [한국어] --num_workers: DataLoader 서브프로세스 개수. 0이면 main 프로세스.
    parser.add_argument('--num_workers', type=int, default=0)
    # [한국어] --hidden_channels: GNN 중간 레이어 폭.
    parser.add_argument('--hidden_channels', type=int, default=128)
    # [한국어] --learning_rate / --decay: Adam 학습률/L2 계수.
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.001)
    # [한국어] --epochs: 전체 epoch 수. 100 iter 측정 모드에서는 1 epoch 내 조기 return.
    parser.add_argument('--epochs', type=int, default=3)
    # [한국어] --num_layers: GCN/SAGE 에만 사용(fan_out 길이와 맞춰야 함).
    parser.add_argument('--num_layers', type=int, default=6)
    # [한국어] --num_heads: GAT 전용 attention head 수.
    parser.add_argument('--num_heads', type=int, default=4)
    # [한국어] --log_every: 로깅 주기(본 스크립트 현재 경로에서는 미사용).
    parser.add_argument('--log_every', type=int, default=2)

    #GIDS parameter
    # [한국어] --GIDS: (플래그) GIDS 경로 활성 여부. 실제 분기는 track_acc_GIDS 호출 자체로 결정.
    parser.add_argument('--GIDS', action='store_true', help='Enable GIDS Dataloader')
    # [한국어] --num_ssd: 스트라이핑할 SSD 수 → BaM Controller 개수. gids_nvme.cu::init_GIDS_controllers 로 전달.
    parser.add_argument('--num_ssd', type=int, default=1)
    # [한국어] --cache_size: BaM page_cache 크기(MB). n_pages = cache_size*1MB/page_size 로 변환.
    parser.add_argument('--cache_size', type=int, default=8)
    # [한국어] --uva / --uva_graph: UVA(Unified Virtual Addressing) 경로 플래그(baseline 비교용).
    parser.add_argument('--uva', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)
    # [한국어] --wb_size: window buffering 이 앞쪽 n개 배치를 프리패치할지 결정. set_window_buffering_kernel 에 전달.
    parser.add_argument('--wb_size', type=int, default=6)

    # [한국어] --device: CUDA device 번호 — "cuda:N" 문자열 구성에 사용.
    parser.add_argument('--device', type=int, default=0)

    #GIDS Optimization
    # [한국어] --accumulator: (플래그) Storage Access Accumulator 활성. set_required_storage_access 로 하드웨어 파라미터 전달.
    parser.add_argument('--accumulator', action='store_true', help='Enable Storage Access Accmulator')
    parser.add_argument('--bw', type=float, default=5.8, help='SSD peak bandwidth in GB/s')              # [한국어] SSD 피크 대역폭.
    parser.add_argument('--l_ssd', type=float, default=11.0, help='SSD latency in microseconds')         # [한국어] SSD I/O 레이턴시.
    parser.add_argument('--l_system', type=float, default=20.0, help='System latency in microseconds')   # [한국어] PCIe+host 추가 지연.
    parser.add_argument('--peak_percent', type=float, default=0.95)                                       # [한국어] 대역폭 타겟 비율(0~1).

    # [한국어] --num_iter: 반복 측정 횟수(스위프용, 본 경로 미사용).
    parser.add_argument('--num_iter', type=int, default=1)

    # [한국어] --cpu_buffer: (플래그) Constant CPU Buffer 활성. 핫 노드 feature를 host-pinned 에 고정.
    parser.add_argument('--cpu_buffer', action='store_true', help='Enable CPU Feature Buffer')
    # [한국어] --cpu_buffer_percent: 핀 노드 비율(0.2 → 20% of total nodes).
    parser.add_argument('--cpu_buffer_percent', type=float, default=0.2, help='CPU feature buffer size (0.1 for 10%)')
    # [한국어] --pin_file: 핀할 노드 인덱스 텐서(.pt). page_rank_node_list_gen.py 결과물.
    parser.add_argument('--pin_file', type=str, default="/mnt/nvme17/pr_full.pt",
        help='Pytorch Tensor File for the list of nodes that will be pinned in the CPU feature buffer')

    # [한국어] --window_buffer: Window Buffering 프리패치 활성. set_window_buffering_kernel 트리거.
    parser.add_argument('--window_buffer', action='store_true', help='Enable Window Buffering')



    #GPU Software Cache Parameters
    # [한국어] --page_size: BaM page_cache 페이지 크기(바이트). 보통 8 → 8바이트 단위 아닌 "index unit" 해석은 구현 참조.
    parser.add_argument('--page_size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD')   # [한국어] SSD 내 feature 시작 바이트 오프셋.
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)')  # [한국어] range_t n_elems.
    parser.add_argument('--cache_dim', type=int, default=1024) #CHECK  # [한국어] feature dim — GPU 커널 grid 결정.


    args = parser.parse_args()
    # [한국어] 주요 옵션 요약 출력 — 실험 로그에서 어떤 경로로 돌렸는지 식별하기 위함.
    print("GIDS DataLoader Setting")
    print("GIDS: ", args.GIDS)
    print("CPU Feature Buffer: ", args.cpu_buffer)
    print("Window Buffering: ", args.window_buffer)
    print("Storage Access Accumulator: ", args.accumulator)

    labels = None  # [한국어] OGB 에서만 별도 사용되는 경우를 위해 placeholder.
    # [한국어] device 문자열 구성. CUDA 없으면 CPU 폴백(실전은 GPU 전제).
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)   # [한국어] IGB260M topology 로드(feature는 SSD).
        g = dataset[0]                       # [한국어] 단일 그래프 반환.
        g  = g.formats('csc')                # [한국어] DGL CSC(Compressed Sparse Column) — NeighborSampler 효율 포맷.
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    else:
        g=None
        dataset=None

    # [한국어] 학습 시작. labels 는 OGB 전용 경로에서만 넘겨짐.
    track_acc_GIDS(g, args, device, labels)




