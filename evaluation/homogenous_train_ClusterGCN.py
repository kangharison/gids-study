"""
[한국어 설명] ClusterGCN 변형 학습 스크립트 (homogenous_train_ClusterGCN.py)

=== 파일의 역할 ===
동종 그래프에 대해 ClusterGCN / NHS / LADIES 샘플러를 선택적으로 쓰되 GIDS+BaM
경로로 feature 를 공급하는 실험 스크립트. 일반 `homogenous_train.py` 와 달리
간단한 `ClusterSAGE` (3-layer SAGE) 모델을 포함하여 subgraph(sg) 단위로 forward
한다. ClusterGCN 경로에서는 train_nid 가 partition index 이고 sampler 출력이
subgraph 이므로, 반복 unpack 형식도 (step, sg, ret) 으로 다르다.

=== 전체 아키텍처에서의 위치 ===
"Python 학습 스크립트" 층. 호출 체인: `python homogenous_train_ClusterGCN.py` →
`__main__` → `track_acc_GIDS(g, args, device)` → `GIDS.GIDS(...)` (ssd_list=[5] 고정)
→ `dgl.dataloading.DataLoader(..., bam=True, GIDS=GIDS_Loader)` 경로로 feature 를
BaM 에서 읽어온다. `bam=True` 플래그가 DGL 이 내부에서 GIDS fetch hook 을 호출하도록
전환하는 역할.

=== 타 모듈과의 연결 ===
- `models.py`: (코멘트 처리됨, 현재는 ClusterSAGE 사용).
- `dataloader.py`: IGB260M/OGB 동종 그래프 로더.
- `ladies_sampler.py`: LADIES / PoissonLadies 샘플러.
- `GIDS`: GIDS.GIDS 로 BaM 초기화.
- 데이터 흐름: SSD feature → BaM page_cache(GPU) → ret → ClusterSAGE(sg, ret).

=== 주요 함수/구조체 요약 ===
- `ClusterSAGE`: 3-layer GraphSAGE(mean aggregator) + dropout. subgraph 단위 forward.
- `fetch_data_chunk`, `print_times`: 유틸.
- `track_acc_GIDS`: GIDS Loader 초기화 → ClusterGCN/NHS/LADIES 샘플러 분기 →
  DGL DataLoader(bam=True) → 모델/옵티마이저 → 학습 루프.
- `__main__`: argparse 파싱, 데이터셋 로드(csc), 학습 호출.
"""

# [한국어] 공통 임포트 — 학습/샘플러/GIDS.
import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *    # [한국어] GCN/SAGE/GAT (현 경로에서는 주석 처리되어 미사용).
from dataloader import IGB260MDGLDataset, OGBDGLDataset
import csv
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

# [한국어] GIDS: BaM 기반 feature fetch 경로.
import GIDS
# [한국어] LADIES/PoissonLADIES 샘플러(layer-wise importance sampling 변종).
from ladies_sampler import LadiesSampler, normalized_edata, PoissonLadiesSampler

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")

# [한국어]
# ClusterSAGE - ClusterGCN 용 3-layer GraphSAGE
#
# @param in_feats: 입력 feature 차원.
# @param n_hidden: hidden dim.
# @param n_classes: 출력 클래스 수.
#
# forward(sg, x) 는 subgraph sg (ClusterGCN partition 하나) 와 feature x 를 받아
# mean aggregator 기반 SAGEConv 3층을 통과시킨다. 마지막 층은 ReLU/Dropout 을 적용
# 하지 않아 logits 를 그대로 반환. 실행 컨텍스트: GPU(Device), forward/backward 모두
# CUDA 연산. 호출 체인: track_acc_GIDS → model(sg, ret) → dglnn.SAGEConv.
class ClusterSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()   # [한국어] layer 컨테이너 — 순서대로 실행.
        # [한국어] SAGEConv(in, out, "mean"): 이웃의 평균으로 메시지 집계.
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)   # [한국어] 과적합 방지 — 학습 시 50% drop.

    def forward(self, sg, x):
        # [한국어] h: running hidden 표현. 매 layer 후 갱신.
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)    # [한국어] subgraph sg 위에서 메시지 전달 후 집계.
            # [한국어] 마지막 layer 는 logits 그대로 → CrossEntropyLoss 와 결합.
            if l != len(self.layers) - 1:
                h = F.relu(h)   # [한국어] 비선형 활성화.
                h = self.dropout(h)
        return h



# [한국어] fetch_data_chunk - NVTX 마커 래퍼. 디버그 프로파일링용.
@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)


# [한국어] print_times - 단계별 시간 로깅.
def print_times(transfer_time, train_time, e2e_time):
    print("transfer time: ", transfer_time)
    print("train time: ", train_time)
    print("e2e time: ", e2e_time)

# [한국어]
# track_acc_GIDS - ClusterGCN/NHS/LADIES + GIDS 경로로 학습
#
# @param g: DGL 그래프(csc).
# @param args: argparse 네임스페이스(sample_type, num_partitions, ladouts 포함).
# @param device: "cuda:N".
# @param label_array: OGB label 텐서 옵션.
# @return: None(warm-up + 100 iter 측정 후 early return).
#
# 차이점(homogenous_train.py 대비): (1) ssd_list=[5] 로 고정(단일 특정 /dev/libnvm5 사용).
# (2) 샘플러를 sample_type 으로 분기. (3) ClusterGCN 은 sampler 가 subgraph 를 반환하므로
# 학습 루프에서 (step, sg, ret) 로 unpack. (4) 모델이 ClusterSAGE 로 고정.
# 실행 컨텍스트: 호스트 Python + GPU CUDA. 호출 체인은 GIDS 경로와 동일하나 DataLoader
# 구성 시 bam=True/GIDS=GIDS_Loader 키워드 사용.
def track_acc_GIDS(g, args, device, label_array=None):

    # [한국어] === 단계 1: GIDS Loader 초기화 (ssd_list=[5] 고정) ===
    GIDS_Loader = None
    GIDS_Loader = GIDS.GIDS(
        page_size = args.page_size,       # [한국어] BaM page 크기.
        off = args.offset,                 # [한국어] SSD 내 feature 오프셋.
        num_ele = args.num_ele,            # [한국어] 전체 feature element 수.
        num_ssd = args.num_ssd,
        cache_size = args.cache_size,      # [한국어] GPU page cache(MB).
        window_buffer = args.window_buffer,# [한국어] window buffering 플래그.
        wb_size = args.wb_size,            # [한국어] WB 프리패치 깊이(batch 수).
        accumulator_flag = args.accumulator,
        ssd_list = [5],                    # [한국어] /dev/libnvm5 를 명시적으로 지정 — 단일 SSD 실험용.
        cache_dim = args.cache_dim

    )
    dim = args.emb_size

    if(args.accumulator):
        GIDS_Loader.set_required_storage_access(args.bw, args.l_ssd, args.l_system, args.num_ssd, args.peak_percent)


    if(args.cpu_buffer):
        num_nodes = g.number_of_nodes()
        num_pinned_nodes = int(num_nodes * args.cpu_buffer_percent)
        GIDS_Loader.cpu_backing_buffer(dim, num_pinned_nodes)
        pr_ten = torch.load(args.pin_file)
        GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned_nodes)




    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    in_feats = g.ndata['features'].shape[1]

    # [한국어] === Sampler 선택 ===
    sampler = None
    if (args.sample_type == 'NHS'):
        # [한국어] 이웃 샘플링(Neighbor Hop Sampling).
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in args.fan_out.split(',')]
            )

    elif(args.sample_type == "ClusterGCN"):
        # [한국어] ClusterGCN: 그래프 파티셔닝 기반. partition id 시퀀스를 seed 로 사용.
        sampler = dgl.dataloading.ClusterGCNSampler(
            g,
            args.num_partitions
        )
        train_nid = torch.arange(args.num_partitions)
    #LADIES
    else:
        # [한국어] LADIES: importance sampling. edge weight 정규화 필요.
        g.edata["w"] = normalized_edata(g)
        sampler = LadiesSampler(args.ladouts)


    # [한국어] === DGL DataLoader — BaM 경로 활성 ===
    # bam=True 키워드는 GIDS fork 버전 DGL 이 인식하여, sampler 출력 후
    # GIDS_Loader.fetch_feature 를 호출하고 결과 텐서를 iter 결과에 포함시킨다.
    # feature_dim/GIDS/window_buffer_size 는 BaM fetch 커널 launch 파라미터로 전달.
    train_dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=False,
        feature_dim=dim,                 # [한국어] fetch 결과 ret 의 두 번째 차원.
        use_prefetch_thread=False,
        pin_prefetcher=False,
        use_alternate_streams=False,

        use_uva_graph=True,              # [한국어] 그래프 topology 는 UVA 로 GPU 접근.
        bam=True,                        # [한국어] BaM feature fetch 활성 — DGL fork 특수 플래그.
        #window_buffer=args.window_buffer,
        window_buffer=False,             # [한국어] 강제로 False(실험 단순화). 원본 주석 유지.
        window_buffer_size=args.wb_size,
        GIDS=GIDS_Loader                 # [한국어] 내부에서 BAM_Feature_Store 호출 시 사용할 핸들.
    )

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

    # if args.model_type == 'gcn':
    #     model = GCN(in_feats, args.hidden_channels, args.num_classes, 
    #         args.num_layers).to(device)
    # if args.model_type == 'sage':
    #     model = SAGE(in_feats, args.hidden_channels, args.num_classes, 
    #         args.num_layers).to(device)
    # if args.model_type == 'gat':
    #     model = GAT(in_feats, args.hidden_channels, args.num_classes, 
    #         args.num_layers, args.num_heads).to(device)
    # [한국어] ClusterSAGE 인스턴스화 — 원본 __init__ 시그니처가 (in_feats, n_hidden, n_classes) 이므로
    # 추가 인자(128)는 현재 호출 위치가 여분 인자 처리를 기대하나 구현과는 맞지 않을 가능성이 있음(원본 유지).
    # GPU 로 이동 후 학습.
    model = ClusterSAGE(in_feats, 128, args.hidden_channels, args.num_classes).to(device)

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay
        )

    warm_up_iter = 100
    # Setup is Done
    for epoch in tqdm.tqdm(range(args.epochs)):
        epoch_start = time.time()
        epoch_loss = 0
        train_acc = 0
        model.train()

        batch_input_time = 0
        train_time = 0
        transfer_time = 0
        e2e_time = 0
        e2e_time_start = time.time()

        # [한국어] ClusterGCN 경로: 반환이 (sg, ret) 형태. sg 는 subgraph, ret 은 BaM fetch 결과.
        # (주석 처리된 원본은 이웃 샘플링 unpack 형식.)
 #       for step, (input_nodes, seeds, blocks, ret) in enumerate(train_dataloader):
        for step, sg, ret in enumerate(train_dataloader):

            if(step % 10 == 0):
                print("step: ", step)
            if(step == warm_up_iter):
                print("warp up done")
                train_dataloader.print_stats()
                train_dataloader.print_timer()
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                e2e_time_start = time.time()
        
            
            # Features are fetched by the baseline GIDS dataloader in ret

            # [한국어] ret: GPU 상의 feature 텐서(BaM fetch 결과).
            batch_inputs = ret

            # [한국어] subgraph ndata 에서 label 추출 및 train_mask (partition 내 학습 대상만) 산출.
            batch_labels = sg.ndata["label"]
            m = sg.ndata["train_mask"].bool()


            # [한국어] 주의: 이 블록 내 `blocks`/`x`/`transfer_start` 변수는 원본 코드의 잔해로
            # 현 ClusterGCN 경로에서는 정의가 누락되어 런타임 에러 가능. 주석 작업은 코드 수정 금지
            # 원칙에 따라 원본을 그대로 유지한다.
            blocks = [block.int().to(device) for block in blocks]
            batch_labels = batch_labels.to(device)
            transfer_time = transfer_time +  time.time()  - transfer_start

            # Model Training Stage
            # [한국어] --- forward/backward/step ---
            train_start = time.time()
            batch_pred = model(sg, x)                    # [한국어] ClusterSAGE forward(subgraph + feature).
            #batch_pred = model(blocks, batch_inputs)    # [한국어] NHS 경로용 대체(주석).
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
            train_time = train_time + time.time() - train_start
          
            if(step == warm_up_iter + 100):
                print("Performance for 100 iteration after 1000 iteration")
                e2e_time += time.time() - e2e_time_start 
                train_dataloader.print_stats()
                train_dataloader.print_timer()
                print_times(transfer_time, train_time, e2e_time)
             
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                
                #Just testing 100 iterations remove the next line if you do not want to halt
                return None


       
  
    # Evaluation

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for _, _, blocks,_ in test_dataloader:
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']
     
            if(args.data == 'IGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())
            elif(args.data == 'OGB'):
                out_label = torch.index_select(label_array, 0, b[1]).flatten()
                labels.append(out_label.numpy())
            predict = model(blocks, inputs).argmax(1).cpu().numpy()
            predictions.append(predict)

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
    print("Test Acc {:.2f}%".format(test_acc))




# [한국어] __main__ - ClusterGCN 변형 스크립트 엔트리.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M',
        help='path containing the datasets')   # [한국어] 데이터셋 경로.
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')           # [한국어] 스케일.
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983, 172], help='number of classes')  # [한국어] 출력 클래스.
    parser.add_argument('--in_memory', type=int, default=0,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')       # [한국어] IGB/OGB 분기.
    parser.add_argument('--emb_size', type=int, default=1024)    # [한국어] feature dim.
    
    # Model
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gat', 'sage', 'gcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Sampling
    # [한국어] --sample_type: 본 스크립트 특유의 sampler 분기 인자. ClusterGCN 이 주 대상.
    parser.add_argument('--sample_type', type=str, default='NHS',
                        choices=['NHS', 'ClusterGCN', 'LADIES'])
    # [한국어] --num_partitions: ClusterGCN 파티션 수. train_nid 는 [0..num_partitions) 가 됨.
    parser.add_argument('--num_partitions', type=int, default=1000)



    # Model parameters
    # [한국어] --fan_out: NHS 용. --ladouts: LADIES 용. --batch_size: seed 개수.
    parser.add_argument('--fan_out', type=str, default='10,15')
    parser.add_argument('--ladouts', type=str, default='64,64,64')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=128)   # [한국어] hidden dim.
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=6)          # [한국어] ClusterSAGE 는 3층 고정이라 실제 미사용.
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=2)

    #GIDS parameter
    # [한국어] GIDS 관련 인자 — GIDS.GIDS(...) 생성 시 그대로 forward.
    parser.add_argument('--GIDS', action='store_true', help='Enable GIDS Dataloader')
    parser.add_argument('--num_ssd', type=int, default=1)             # [한국어] Controller 수(ssd_list=[5]이면 실제 1).
    parser.add_argument('--cache_size', type=int, default=8)           # [한국어] page_cache MB.
    parser.add_argument('--uva', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)
    parser.add_argument('--wb_size', type=int, default=6)              # [한국어] window buffer depth.

    parser.add_argument('--device', type=int, default=0)

    #GIDS Optimization
    parser.add_argument('--accumulator', action='store_true', help='Enable Storage Access Accmulator')
    parser.add_argument('--bw', type=float, default=5.8, help='SSD peak bandwidth in GB/s')
    parser.add_argument('--l_ssd', type=float, default=11.0, help='SSD latency in microseconds')
    parser.add_argument('--l_system', type=float, default=20.0, help='System latency in microseconds')
    parser.add_argument('--peak_percent', type=float, default=0.95)

    parser.add_argument('--num_iter', type=int, default=1)

    # [한국어] CPU buffer(핫 노드 pin) 관련.
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


    args = parser.parse_args()
    print("GIDS DataLoader Setting")
    print("GIDS: ", args.GIDS)
    print("CPU Feature Buffer: ", args.cpu_buffer)
    print("Window Buffering: ", args.window_buffer)
    print("Storage Access Accumulator: ", args.accumulator)

    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    # [한국어] 데이터셋 로드 후 CSC 포맷으로 변환 — NeighborSampler/ClusterGCNSampler 가 CSC 를 선호.
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    else:
        g=None
        dataset=None

    # [한국어] 학습 진입.
    track_acc_GIDS(g, args, device, labels)
    #track_acc(g, args, device, labels)




