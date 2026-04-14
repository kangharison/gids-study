"""
[한국어 설명] 동종 그래프 mmap 베이스라인 학습 스크립트 (homogenous_train_baseline.py)

=== 파일의 역할 ===
GIDS+BaM 경로를 사용하지 않고, feature를 디스크에서 OS mmap 으로 읽는 기존
(baseline) 방식으로 학습하여 성능을 비교하기 위한 스크립트이다. 동일한 DGL
sampler(NHS/ClusterGCN/LADIES 택 1), 동일 모델(GCN/SAGE/GAT)을 쓰되, feature
접근 경로만 다르다(blocks[0].srcdata['feat'] 로 host → device 복사). GIDS 논문
Table/Fig 재현 시 대조군 역할을 한다.

=== 전체 아키텍처에서의 위치 ===
"Python 학습 스크립트" 단계이되 BaM 호출 체인은 타지 않는다. 호출 체인:
`python homogenous_train_baseline.py` → `__main__` → `track_acc_Baseline(g, args, device)`
→ DGL `DataLoader` iter → (mmap 된) feature 를 `blocks[0].srcdata['feat']` 로 얻음
→ `.to(device)` 로 PCIe 전송 → 모델 forward. 실행 컨텍스트는 호스트 유저스페이스
CPU 스레드이며, feature 로드는 Linux page cache/mmap 경로.

=== 타 모듈과의 연결 ===
- `models.py`: GCN/SAGE/GAT.
- `dataloader.py`: IGB260M / OGB 로더. mmap 기반 feature 를 제공(GIDS 버전에서는 feature가 SSD).
- `ladies_sampler.py`: LADIES 샘플러 + normalized_edata 헬퍼.
- `GIDS` 모듈은 임포트만 되어 있지만 본 스크립트에서는 실제 API 를 호출하지 않음 —
  argparse 인자 호환 유지 목적.
- 데이터 흐름: 디스크 파일 → mmap → host feature 텐서 → `.to(device)` → model.

=== 주요 함수/구조체 요약 ===
- `fetch_data_chunk`: NVTX 마커 래퍼(실제 baseline 경로에선 미호출).
- `print_times`: 단계별 시간 출력.
- `track_acc_Baseline(g, args, device, label_array)`: 샘플러 분기 → DataLoader 생성 →
  모델 학습. warm-up 1000 iter 이후 100 iter 성능 측정 후 early return.
- `__main__`: argparse 파싱 → 데이터셋 로드(LADIES 면 coo, 그 외 csc) → 사용자에게
  "Lock CPU memory" 입력을 요구한 뒤 학습 시작.
"""

# [한국어] argparse/datetime: CLI 및 로깅(본 경로에서 datetime 은 미사용).
import argparse, datetime
# [한국어] DGL: sampler/DataLoader/graph format.
import dgl
# [한국어] sklearn.metrics: 배치 단위 accuracy_score 계산.
import sklearn.metrics
# [한국어] torch 계열.
import torch, torch.nn as nn, torch.optim as optim
# [한국어] 시간 측정 및 진행바.
import time, tqdm, numpy as np
from models import *
# [한국어] baseline 경로에서는 feature 가 dataset 내부 mmap 텐서로 직접 들어감.
from dataloader import IGB260MDGLDataset, OGBDGLDataset
import csv
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

# [한국어] GIDS 는 argparse 호환을 위해 임포트만 — 실제 호출은 하지 않음.
import GIDS

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

# [한국어] LADIES 샘플러: layer-wise importance sampling. normalized_edata 는 edge weight 정규화 헬퍼.
from ladies_sampler import LadiesSampler, normalized_edata


# [한국어] 재현성 시드 고정.
torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")


# [한국어]
# fetch_data_chunk - (현 baseline 경로 미사용) NVTX 구간으로 감싼 backing memory 청크 복사
# @param test: BAM_Feature_Store pybind 객체.
# @param out_t: 결과 텐서.
# @param page_size: 전송 단위.
# @param stream_id: CUDA stream.
# baseline 스크립트에서는 GIDS 를 호출하지 않으므로 dead code 이나 원본 호환을 위해 유지.
@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)


# [한국어]
# print_times - 학습 루프 단계별 누적 시간 출력(GIDS 버전과 동일 시그니처).
# @param transfer_time: H2D 복사 누적.
# @param train_time: forward+backward 누적.
# @param e2e_time: end-to-end wall-clock.
def print_times(transfer_time, train_time, e2e_time):
    print("transfer time: ", transfer_time)
    print("train time: ", train_time)
    print("e2e time: ", e2e_time)



# [한국어]
# track_acc_Baseline - mmap 기반 baseline 경로로 GNN 학습
#
# @param g: DGL 그래프(csc 또는 coo). feature 는 dataset 내 host mmap 텐서.
# @param args: argparse 네임스페이스. sample_type/num_partitions/ladouts 등 포함.
# @param device: "cuda:N".
# @param label_array: OGB 계열 외부 label 텐서(옵션).
# @return: None(1000 iter warmup 후 100 iter 측정 시 조기 return).
#
# 단계: (1) sample_type 에 따라 sampler 선택(NHS/ClusterGCN/LADIES).
# (2) DGL DataLoader(use_uva=False 즉 host path) 구성.
# (3) 모델/옵티마이저 빌드.
# (4) warm_up_iter=1000, eval_iter=100 기준 측정.
# (5) 각 step 에서 blocks[0].srcdata['feat'] 로 host feature 를 읽고 .to(device) 로 복사.
# 실행 컨텍스트: CPU 메인 프로세스(DataLoader num_workers=0 기본). feature 접근은
# OS page cache / mmap 경로이므로 초반에는 cold cache → page fault 발생.
#
# 호출 체인: __main__ → track_acc_Baseline → DGL DataLoader → sampler → 호스트 feature 로딩 → model.
def track_acc_Baseline(g, args, device, label_array=None):

  
    dim = args.emb_size   # [한국어] feature dim. baseline 에서는 실제 사용되지 않지만 시그니처 호환 유지.

    # sampler = dgl.dataloading.MultiLayerNeighborSampler(
    #            [int(fanout) for fanout in args.fan_out.split(',')]
    #            )

    # [한국어] feat/label 별칭 부여 — 다른 코드 경로와 동일한 ndata 키 구성.
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    # [한국어] split mask 로부터 node id 추출.
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    in_feats = g.ndata['features'].shape[1]   # [한국어] in_feats = feature dim.

    sampler = None   # [한국어] 아래 분기에서 결정.

    # [한국어] === Sampler 선택 ===
    # NHS(Neighbor Hop Sampling): 표준 MultiLayerNeighborSampler.
    if (args.sample_type == 'NHS'):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in args.fan_out.split(',')]
            )

    # [한국어] ClusterGCN: 그래프를 num_partitions 개로 partition 하고 각 partition 을 배치로 사용.
    elif(args.sample_type == "ClusterGCN"):
        sampler = dgl.dataloading.ClusterGCNSampler(
            g,
            args.num_partitions
        )
        # [한국어] train_nid 를 partition id 시퀀스로 교체(ClusterGCN 의 "seed" 는 partition index).
        train_nid = torch.arange(args.num_partitions)
    #LADIES
    else:
        # [한국어] LADIES: layer-wise importance sampling. edge weight 정규화 필요.
        g.edata["w"] = normalized_edata(g)
        l_out = [int(_) for _ in args.ladouts.split(",")]   # [한국어] layer별 샘플 수 파싱.
        sampler = LadiesSampler(l_out)


    # [한국어] === DGL DataLoader (baseline) ===
    # use_uva=False → CPU 경로. use_prefetch_thread/pin_prefetcher 모두 off 로 하여
    # 공정 비교(모든 측정이 동일 thread/stream 에서 이뤄지게).
    train_dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=False,                    # [한국어] UVA 그래프 경로 비활성.
        use_prefetch_thread=False,        # [한국어] DGL prefetch 스레드 비활성.
        pin_prefetcher=False,             # [한국어] pinned memory prefetch 비활성.
        use_alternate_streams=False       # [한국어] 대체 stream 사용 안 함.

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

    if args.model_type == 'gcn':
        model = GCN(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'sage':
        model = SAGE(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'gat':
        model = GAT(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers, args.num_heads).to(device)

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay
        )

    # [한국어] warm_up_iter=1000 — baseline 은 mmap page cache warm-up 비용이 커서
    # GIDS 버전(warm_up=100) 보다 훨씬 긴 워밍업을 둠. eval_iter=100 은 본측정 구간 길이.
    warm_up_iter = 1000
    eval_iter = 100
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

        # [한국어] baseline DataLoader 는 (input_nodes, seeds, blocks) 3-tuple 반환(ret 없음).
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            #
            if(step % 20 == 0):
                print("step: ", step)   # [한국어] 진행 로그.
            if(step == warm_up_iter):
                print("warp up done")
                # [한국어] 통계 리셋 — 측정 구간의 시작점.
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                e2e_time_start = time.time()


            # Features are fetched by the baseline GIDS dataloader in ret

            # [한국어] --- feature fetch (baseline) ---
            # blocks[0].srcdata['feat'] 는 DGL 이 sampler 결과 node 집합에 대해 host(mmap)
            # 에서 읽어 담아준 텐서. 아직 host 에 있음.
            batch_inputs = blocks[0].srcdata['feat']
            transfer_start = time.time()

            # [한국어] 마지막 layer dst 라벨.
            batch_labels = blocks[-1].dstdata['labels']


            # [한국어] --- H2D 전송 ---
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = batch_inputs.to(device)  # [한국어] feature → GPU 복사(PCIe).
            batch_labels = batch_labels.to(device)
            transfer_time = transfer_time +  time.time()  - transfer_start

            # Model Training Stage
            # [한국어] --- forward/backward/step ---
            train_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
            # [한국어] baseline 만 배치마다 accuracy 도 즉시 누적(학습 수렴 체크용).
            train_acc += (sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(),
                batch_pred.argmax(1).detach().cpu().numpy())*100)
            train_time = train_time + time.time() - train_start

            # [한국어] warm_up + eval_iter 도달 시 측정 종료 후 early return.
            if(step == warm_up_iter + eval_iter):
                print("Performance for 100 iteration after 1000 iteration")
                e2e_time += time.time() - e2e_time_start 
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
        for _, _, blocks in test_dataloader:
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



       

# [한국어] __main__ - 베이스라인 학습 엔트리. CLI 파싱 → 데이터셋 로드 → CPU 메모리 락 프롬프트 → 학습 시작.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    # [한국어] 데이터셋 경로/스케일/클래스 수 — GIDS 버전과 동일.
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M',
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983, 172], help='number of classes')
    # [한국어] --in_memory: baseline 이므로 mmap(0) 사용이 주. 1 은 전체 RAM 로드.
    parser.add_argument('--in_memory', type=int, default=0,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)
    
    # Model
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gat', 'sage', 'gcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Sampling
    # [한국어] --sample_type: NHS(neighbor hop) / ClusterGCN(partition) / LADIES(layer-wise importance) 택 1.
    parser.add_argument('--sample_type', type=str, default='NHS',
                        choices=['NHS', 'ClusterGCN', 'LADIES'])
    # [한국어] --num_partitions: ClusterGCN 용 파티션 수.
    parser.add_argument('--num_partitions', type=int, default=1000)



    # Model parameters
    # [한국어] --fan_out: NHS 용 레이어별 이웃 수.
    parser.add_argument('--fan_out', type=str, default='10,5,5')
    # [한국어] --ladouts: LADIES 용 레이어별 샘플 크기.
    parser.add_argument('--ladouts', type=str, default='64,64,64')
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
    # [한국어] 아래 GIDS 인자들은 baseline 에서는 실제 사용되지 않으나, 공통 쉘 스크립트와 CLI 호환을 위해 수용.
    parser.add_argument('--GIDS', action='store_true', help='Enable GIDS Dataloader')
    parser.add_argument('--num_ssd', type=int, default=1)
    parser.add_argument('--cache_size', type=int, default=8)
    parser.add_argument('--uva', type=int, default=0)        # [한국어] baseline 은 UVA 끔(순수 host 경로).
    parser.add_argument('--uva_graph', type=int, default=0)
    parser.add_argument('--wb_size', type=int, default=6)

    parser.add_argument('--device', type=int, default=0)     # [한국어] CUDA device 번호.

    #GIDS Optimization
    parser.add_argument('--accumulator', action='store_true', help='Enable Storage Access Accmulator')
    parser.add_argument('--bw', type=float, default=5.8, help='SSD peak bandwidth in GB/s')
    parser.add_argument('--l_ssd', type=float, default=11.0, help='SSD latency in microseconds')
    parser.add_argument('--l_system', type=float, default=20.0, help='System latency in microseconds')
    parser.add_argument('--peak_percent', type=float, default=0.95)

    parser.add_argument('--num_iter', type=int, default=1)

    # [한국어] CPU buffer 관련 옵션(baseline 에서 무시).
    parser.add_argument('--cpu_buffer', action='store_true', help='Enable CPU Feature Buffer')
    parser.add_argument('--cpu_buffer_percent', type=float, default=0.2, help='CPU feature buffer size (0.1 for 10%)')
    parser.add_argument('--pin_file', type=str, default="/mnt/nvme16/pr_full.pt",
        help='Pytorch Tensor File for the list of nodes that will be pinned in the CPU feature buffer')

    parser.add_argument('--window_buffer', action='store_true', help='Enable Window Buffering')



    #GPU Software Cache Parameters
    # [한국어] BaM page_cache 관련(baseline 에서 미사용).
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
    print("Sample Type: ", args.sample_type)


    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    # [한국어] === 데이터셋 로드 & 포맷 결정 ===
    # LADIES 샘플러는 edge weight 기반 → COO(Coordinate) 포맷이 필요.
    # 그 외(NHS/ClusterGCN) 는 CSC 가 효율.
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        if(args.sample_type == "LADIES"):
            print("LADIES")
            g = g.formats('coo')
        else:
            g  = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
        if(args.sample_type == "LADIES"):
            print("LADIES")
            g = g.formats('coo')
            g = g.long()   # [한국어] OGB 는 int64 id 가 필요할 수 있음.
        else:
            g  = g.formats('csc')
    else:
        g=None
        dataset=None

    # [한국어] mlock 등 외부에서 CPU 메모리를 잠글 수 있도록 수동 대기. 사용자가 enter 치면 진행.
    # (공정 측정을 위해 page cache 고정 혹은 외부 락 유틸 실행 포인트.)
    sec = input('Lock CPU memory.\n')
    print("start training")

    # [한국어] 학습 시작.
    track_acc_Baseline(g, args, device, labels)
    #track_acc(g, args, device, labels)




