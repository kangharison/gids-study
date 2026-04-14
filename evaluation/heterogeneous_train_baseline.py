"""
[한국어 설명] 이종 그래프 mmap 베이스라인 학습 스크립트 (heterogeneous_train_baseline.py)

=== 파일의 역할 ===
이종 그래프(IGBH/OGB-MAG)에 대해 GIDS+BaM 을 쓰지 않고 mmap 기반 host feature 로
학습하여 GIDS 경로와 성능/메모리 비교를 수행한다. 동일 RGCN/RSAGE/RGAT 모델,
동일 DGL NeighborSampler를 사용하되, feature 는 `blocks[0].srcdata['feat']`
(dict[type → tensor]) 를 host 에서 받아 `.to(device)` 로 복사한다.

=== 전체 아키텍처에서의 위치 ===
"Python 학습 스크립트" 층의 baseline. 호출 체인: `python heterogeneous_train_baseline.py`
→ `__main__` → `track_acc_Baseline(g, category, args, device)` → DGL DataLoader
(`use_uva=False`, device=GPU) → sampler → host feature 로드 → model. GIDS/BaM
경로는 전혀 타지 않음.

=== 타 모듈과의 연결 ===
- `models.py`: RGCN/RSAGE/RGAT.
- `dataloader.py`: IGBHeteroDGLDataset(Massive), OGBHeteroDGLDatasetMassive.
- `GIDS`: 임포트 되지만 본 스크립트에선 호출 안 함(CLI 호환 목적).
- 데이터 흐름: mmap feature(host, dict) → H2D 복사 → model.

=== 주요 함수/구조체 요약 ===
- `fetch_data_chunk`, `print_times`: 유틸(미사용 혹은 로깅).
- `track_acc_Baseline(g, category, args, device, label_array)`: DGL DataLoader 로
  이종 batch 생성 → host feature → `.to(device)` → RGNN forward. warm-up 1000 iter
  이후 100 iter 성능 측정 후 early return.
- `__main__`: argparse → 데이터셋/포맷 로드 → 학습 호출.
"""

import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from dataloader import IGB260MDGLDataset, OGBDGLDataset
# [한국어] 이종 로더들(Massive 는 large/full 전용).
from dataloader import IGBHeteroDGLDataset, IGBHeteroDGLDatasetMassive, OGBHeteroDGLDatasetMassive

import csv
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

# [한국어] GIDS 는 import 만 — 실제 호출은 없음(CLI 호환용).
import GIDS

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")


# [한국어] fetch_data_chunk - NVTX 마커(미사용, 호환).
@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)


# [한국어] print_times - 단계별 시간 로깅.
def print_times(transfer_time, train_time, e2e_time):
    print("transfer time: ", transfer_time)
    print("train time: ", train_time)
    print("e2e time: ", e2e_time)



# [한국어]
# track_acc_Baseline - mmap 기반 이종 그래프 GNN 학습
#
# @param g: DGL heterograph(csc).
# @param category: 예측 대상 node type(예: 'paper').
# @param args: argparse 네임스페이스.
# @param device: "cuda:N".
# @param label_array: OGB 외부 label 텐서(옵션).
# @return: None(warm_up_iter=1000 + 100 측정 후 return).
#
# GIDS 경로와 달리 DataLoader(device=device, use_uva=False) 로 구성되어 내부에서
# DGL 자체 host→device transfer 수행. feature 는 dataset 이 제공한 host 텐서(dict).
def track_acc_Baseline(g, category, args, device, label_array=None):

  
    dim = args.emb_size   # [한국어] feature dim(기록 목적).

    # [한국어] 이종 NeighborSampler — edge type 별로 fanout 샘플링.
    sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])



    # [한국어] category node 의 split mask → train/val/test nid.
    train_nid = torch.nonzero(g.nodes[category].data['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.nodes[category].data['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.nodes[category].data['test_mask'], as_tuple=True)[0]

    # [한국어] baseline 은 host feature 가 접근 가능하므로 shape 으로 in_feats 직접 추출.
    in_feats = g.ndata['feat'][category].shape[1]

    # [한국어] === 이종 모델 선택 ===
    if args.model_type == 'rgcn':
        model = RGCN(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rsage':
        model = RSAGE(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rgat':
        # [한국어] baseline 은 RGAT 원본 클래스 사용(heterogeneous GIDS 는 RGNN 사용).
        model = RGAT(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers, args.num_heads).to(device)



    # [한국어] === DGL DataLoader (이종 baseline) ===
    # device=device 파라미터가 전달되어 DGL 내부에서 host→device 전송까지 담당(prefetch 스레드는 off).
    train_dataloader = dgl.dataloading.DataLoader(
        g,
        {category: train_nid},
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=False,                # [한국어] host 경로 고정.
        use_prefetch_thread=False,
        pin_prefetcher=False,
        use_alternate_streams=False,
        device=device                  # [한국어] DataLoader 가 자체적으로 device 텐서 반환.
    )

    val_dataloader = dgl.dataloading.DataLoader(
        g, {category: val_nid},sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, {category: test_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),
        lr=args.learning_rate)
    # [한국어] StepLR: 25 step 마다 lr *= 0.25. baseline 만 사용(원본 관례).
    sched = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.25)

    # [한국어] baseline 은 warm-up 1000(mmap page cache 안정화 필요).
    warm_up_iter = 1000
    # Setup is Done
    for epoch in tqdm.tqdm(range(args.epochs)):
        epoch_start = time.time()
        epoch_loss = 0
        train_acc = 0
        model.train()
        total_loss = 0

        batch_input_time = 0
        train_time = 0
        transfer_time = 0
        e2e_time = 0
        e2e_time_start = time.time()

        # [한국어] 이종 baseline DataLoader iter: (input_nodes, seeds, blocks) — ret 없음.
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            if(step % 20 == 0):
                print("step: ", step)
            if(step == warm_up_iter):
                print("warp up done")
                train_dataloader.print_timer()   # [한국어] DGL 내장 타이머(이종 fork 버전만 존재할 수 있음).
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                e2e_time_start = time.time()


            # Features are fetched by the baseline GIDS dataloader in ret

            # [한국어] blocks[0].srcdata['feat'] 는 dict[type → tensor]. 이미 device=device 지정이면 GPU 에 있음.
            batch_inputs = blocks[0].srcdata['feat']
            transfer_start = time.time()

            # [한국어] paper type 의 라벨만 사용(category='paper' 가정).
            y = blocks[-1].dstdata['label']['paper']

            blocks = [block.int().to(device) for block in blocks]

            transfer_time = transfer_time +  time.time()  - transfer_start

            # Model Training Stage
            # [한국어] --- forward/backward/step ---
            train_start = time.time()
            y_hat = model(blocks, batch_inputs)
            loss = loss_fcn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            cur_train_time = time.time() - train_start
            train_time += cur_train_time
          
            if(step == warm_up_iter + 100):
                print("Performance for 100 iteration after 1000 iteration")
                e2e_time += time.time() - e2e_time_start 
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



       

# [한국어] __main__ - 이종 baseline 엔트리. key_offset 없음(GIDS 경로 아님).
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M',
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983, 348,349, 350, 153, 152], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)

    # Model
    # [한국어] 이종 모델 선택. 기본 'gcn'은 choices 와 불일치(원본 버그 유지).
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['rgat', 'rsage', 'rgcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters
    parser.add_argument('--fan_out', type=str, default='10,15')       # [한국어] layer별 fanout.
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
    # [한국어] baseline 에서는 실제 호출되지 않지만 CLI 호환을 위해 수용.
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


    args = parser.parse_args()
    print("GIDS DataLoader Setting")
    print("GIDS: ", args.GIDS)
    print("CPU Feature Buffer: ", args.cpu_buffer)
    print("Window Buffering: ", args.window_buffer)
    print("Storage Access Accumulator: ", args.accumulator)


    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    # [한국어] === 데이터셋 선택 ===
    if(args.data =='OGB'):
        dataset = OGBHeteroDGLDatasetMassive(args)   # [한국어] OGB-MAG.

    else:
        # [한국어] IGB 계열: 크기에 따라 Massive/일반 로더 분기.
        if(args.dataset_size == 'full' or args.dataset_size == 'large'):
            dataset = IGBHeteroDGLDatasetMassive(args)
        else:
            dataset = IGBHeteroDGLDataset(args)
    g = dataset[0]
    print("format: ", g.formats())
    g  = g.formats('csc')   # [한국어] CSC 로 변환(NeighborSampler 효율).
    print("format: ", g.formats())
    # [한국어] category: 예측 대상 node type.
    category = g.predict

    # g_etypes = g.canonical_etypes
    # graph_dict = {}
    # for i in range(len(g_etypes)):
    #     graph_dict[g_etypes[i]] = ('csc', g.adj_sparse('csc',  etype=g_etypes[i]))

    # torch.save(graph_dict, "/mnt/nvme22/OGB_graph_csc.pth")

    # category = g.predict
    # print("g: ", g)



    # [한국어] 학습 진입(baseline). label_array 는 미지정.
    track_acc_Baseline(g, category, args, device)


  



