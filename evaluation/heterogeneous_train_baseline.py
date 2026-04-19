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

# [한국어] argparse: CLI 파싱 (--dataset_size/--fan_out 등 학습 스크립트 공통 인자 세트).
# datetime: 로깅 타임스탬프 후보(본 경로에선 미사용이지만 공통 import 세트 유지).
import argparse, datetime
# [한국어] DGL(Deep Graph Library): NeighborSampler + DataLoader — 이종 그래프도 동일 API 로 다룸.
#          본 baseline 은 use_uva=False 경로로만 사용.
import dgl
# [한국어] sklearn.metrics: test 단계 accuracy_score — 예측/정답 numpy 배열 비교.
import sklearn.metrics
# [한국어] torch: 텐서/자동미분. nn: 모델 계층 기반. optim: Adam+StepLR 스케줄러.
import torch, torch.nn as nn, torch.optim as optim
# [한국어] time: wall-clock 측정. tqdm: epoch 진행바. numpy: 예측/라벨 numpy concat.
import time, tqdm, numpy as np
# [한국어] evaluation/models.py 의 GNN 모델 클래스 전체 — RGCN/RSAGE/RGAT 포함.
from models import *
# [한국어] 동종 그래프 로더 — baseline 스크립트가 dataloader import 세트를 통일해 공유.
from dataloader import IGB260MDGLDataset, OGBDGLDataset
# [한국어] 이종 로더들(Massive 는 large/full 전용). 이종 baseline 은 feature 를 host 에서 mmap 으로 로드.
# IGBHeteroDGLDataset: small/medium/tiny — 일반 로더.
# IGBHeteroDGLDatasetMassive: large/full — 메모리 절감형 mmap 로더.
# OGBHeteroDGLDatasetMassive: OGB-MAG 대형 이종.
from dataloader import IGBHeteroDGLDataset, IGBHeteroDGLDatasetMassive, OGBHeteroDGLDatasetMassive

# [한국어] csv: 결과 로깅 유틸(현 경로 미사용). warnings: 실험 시 경고 억제.
import csv
import warnings

# [한국어] torch.cuda.nvtx: PyTorch 측 NVTX push/pop API. Nsight Systems 프로파일링 마커용.
import torch.cuda.nvtx as t_nvtx
# [한국어] nvtx: @nvtx.annotate 데코레이터 제공 — 함수 전체를 NVTX 구간으로 감쌈.
#          fetch_data_chunk 가 이 데코레이터를 사용 (baseline 에서는 실제 호출 안됨).
import nvtx
# [한국어] threading: DataLoader prefetch thread 실험 후보. 본 경로에선 미사용.
import gc
import threading

# [한국어] GIDS 는 import 만 — 실제 호출은 없음(CLI 호환용).
#          아래 argparse 의 --GIDS/--cpu_buffer 등은 공통 쉘 스크립트 호환 목적으로 수용만 한다.
import GIDS

# [한국어] OGB: Open Graph Benchmark 접근 API. Evaluator 는 현 baseline 경로 미사용.
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

# [한국어] 재현성 확보: torch 난수/ DGL sampler 난수 시드 고정.
#          경고: 이종 샘플러가 내부에서 np.random 를 쓰는 경로가 있으면 별도 시드가 필요할 수 있음.
torch.manual_seed(0)
dgl.seed(0)
# [한국어] DGL/OGB/PyTorch deprecation 경고 대량 발생 → 실험 로그 가독성 위해 묵음.
warnings.filterwarnings("ignore")


# [한국어]
# fetch_data_chunk - (baseline 미호출) BAM_Feature_Store 의 backing memory 청크를 out_t 에 복사
#
# @param test: C++ BAM_Feature_Store pybind 객체(float/long 템플릿 중 하나). baseline 경로에선 전달되지 않음.
# @param out_t: device 텐서. out_t.data_ptr() 로 디바이스 포인터를 C++ 로 넘긴다.
# @param page_size: BaM page_cache_t 의 페이지 크기 — 전송 단위.
# @param stream_id: CUDA stream index — 비동기 전송 스케줄링용.
# @return: None (in-place 복사).
#
# NVTX blue 마커로 감싸 Nsight Systems 에서 "fetch_data_chunk()" 구간이 가시화된다.
# baseline 은 GIDS 호출이 없어 dead code 지만, 공통 CLI/워크플로우 호환을 위해 보존.
# 실행 컨텍스트: Python main thread(pybind11 GIL 해제 경로).
@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    # [한국어] pybind C++ 메서드 호출. GIL 은 C++ 내부에서 해제될 수 있음(cudaMemcpy 등 장기 블로킹).
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)


# [한국어]
# print_times - 학습 루프 단계별 누적 시간을 stdout 에 출력
# @param transfer_time: blocks/feature → GPU device 복사 누적(초).
# @param train_time: forward+backward+optimizer.step 누적(초).
# @param e2e_time: 측정 구간 end-to-end wall-clock(초).
# @return: None. 호출 체인: track_acc_Baseline → print_times. warm-up 이후 100 iter 구간에서 1회 호출.
def print_times(transfer_time, train_time, e2e_time):
    print("transfer time: ", transfer_time)   # [한국어] H2D 전송 비용(page cache 미스 포함 시 큼).
    print("train time: ", train_time)         # [한국어] GPU 연산 비용 — feature dim/배치 크기에 의존.
    print("e2e time: ", e2e_time)             # [한국어] 전체 구간(샘플링+전송+학습). 진단용.



# [한국어]
# track_acc_Baseline - mmap 기반 이종 그래프 GNN 학습(GIDS/BaM 미사용)
#
# @param g: DGL heterograph(csc 포맷). 여러 node type(paper/author/fos/...) 보유.
#           ndata['feat'] 는 type 별 dict → 각 텐서는 host mmap 으로 열려 있음.
# @param category: 예측 대상 node type 이름(예: 'paper'). g.predict 로부터 전달.
# @param args: argparse 네임스페이스 — 모델/샘플러/하이퍼파라미터 단일 전달체.
# @param device: "cuda:N" 문자열.
# @param label_array: OGB 외부 label 텐서(옵션) — IGB 경로에선 미사용.
# @return: None — warm_up_iter=1000 경과 후 warm_up+100 에서 early return(벤치마크 모드).
#
# GIDS 경로와 달리 DataLoader(device=device, use_uva=False) 로 구성되어 DGL 내부 prefetch 로직이
# host feature(mmap) 을 GPU 로 복사한다. feature 는 dataset 이 제공한 host 텐서(type→tensor dict).
# 호출 체인: __main__ → track_acc_Baseline → DGL NeighborSampler → DataLoader iter
#              → blocks[0].srcdata['feat'] (type별 dict) → model(blocks, batch_inputs).
# 실행 컨텍스트: Python main thread + CUDA default stream. num_workers=0 이면 샘플링도 main thread.
# 에러 경로: mmap 파일 누락 → dataloader 내부 IndexError 발생 → Python traceback 으로 종료.
def track_acc_Baseline(g, category, args, device, label_array=None):


    dim = args.emb_size   # [한국어] feature dim(기록 목적, 실제 분기에는 미사용 — in_feats 가 shape 에서 직접 도출).

    # [한국어] 이종 NeighborSampler — 각 edge type 별로 fanout 개의 이웃을 샘플링.
    #          fan_out="10,15" → [10, 15] 리스트, layer 수 = len(list) = 2.
    #          이종 그래프에서는 DGL 이 edge type 별로 내부 MFG 를 생성.
    sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])



    # [한국어] category node 의 split mask → train/val/test nid.
    #          nonzero(as_tuple=True)[0]: True 위치의 index 1D 텐서 반환.
    #          이종 그래프이므로 g.nodes[category].data 로 dict 접근.
    train_nid = torch.nonzero(g.nodes[category].data['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.nodes[category].data['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.nodes[category].data['test_mask'], as_tuple=True)[0]

    # [한국어] baseline 은 host feature 가 접근 가능하므로 shape 으로 in_feats 직접 추출.
    #          GIDS 경로와 달리 emb_size 추정이 아닌 실제 텐서 차원 사용.
    in_feats = g.ndata['feat'][category].shape[1]

    # [한국어] === 이종 모델 선택 ===
    # R-prefix = Relational (각 edge type 별 별도 weight matrix 보유).
    if args.model_type == 'rgcn':
        # [한국어] RGCN: 각 edge type 별 W_r 매트릭스 + 이웃 평균.
        model = RGCN(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rsage':
        # [한국어] RSAGE: GraphSAGE 의 relational 변형.
        model = RSAGE(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rgat':
        # [한국어] baseline 은 RGAT 원본 클래스 사용(heterogeneous GIDS 경로는 mlperf_model 의 RGNN 사용).
        #          num_heads: attention head 수.
        model = RGAT(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers, args.num_heads).to(device)



    # [한국어] === DGL DataLoader (이종 baseline) ===
    # device=device 파라미터가 전달되어 DGL 내부에서 host→device 전송까지 담당(prefetch 스레드는 off).
    # seed 는 {category: train_nid} 이종 dict — sampler 가 이 type 의 노드부터 역방향으로 MFG 구성.
    train_dataloader = dgl.dataloading.DataLoader(
        g,
        {category: train_nid},          # [한국어] 이종 seed 는 반드시 dict. single-type batch 이면 1-key dict.
        sampler,
        batch_size=args.batch_size,      # [한국어] seed 개수(category 노드). fanout 따라 input_nodes 는 폭증.
        shuffle=True,                    # [한국어] epoch 마다 seed 순서 섞기(일반화).
        drop_last=False,                 # [한국어] 마지막 짜투리 배치도 사용.
        num_workers=args.num_workers,    # [한국어] 0이면 main thread 샘플링. >0 이면 fork 된 프로세스에서 샘플링.
        use_uva=False,                   # [한국어] host 경로 고정 — GIDS 와 달리 UVA/BaM 미사용.
        use_prefetch_thread=False,       # [한국어] DGL prefetch thread off — 공정 비교 목적.
        pin_prefetcher=False,            # [한국어] pinned memory prefetch off.
        use_alternate_streams=False,     # [한국어] 대체 stream 미사용 — default stream 에서만 H2D.
        device=device                     # [한국어] DataLoader 가 자체적으로 device 텐서 반환 — blocks[*].srcdata 가 이미 GPU.
    )

    # [한국어] validation DataLoader: 현재 벤치마크 모드에선 미사용(아래 warmup+100 구간에서 return 함).
    val_dataloader = dgl.dataloading.DataLoader(
        g, {category: val_nid},sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,   # [한국어] val 은 순서 고정(결정론적 평가).
        num_workers=args.num_workers)

    # [한국어] test DataLoader: 최종 evaluation 용. shuffle=True 는 원본 관례(결과 평균은 순서 무관).
    test_dataloader = dgl.dataloading.DataLoader(
        g, {category: test_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)

    # [한국어] CrossEntropy: multi-class 분류용 loss. .to(device) 로 내부 weight(없지만) device 정합.
    loss_fcn = nn.CrossEntropyLoss().to(device)
    # [한국어] Adam: 적응형 learning rate. weight_decay 없음(이종 baseline 원본 관례).
    optimizer = optim.Adam(model.parameters(),
        lr=args.learning_rate)
    # [한국어] StepLR: 25 step 마다 lr *= 0.25. baseline 만 사용(원본 관례).
    #          warmup+100 early return 경로에서는 실제 step 되지 않음.
    sched = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.25)

    # [한국어] baseline 은 warm-up 1000(mmap page cache 안정화 필요).
    #          GIDS 경로(100) 대비 훨씬 긴 이유: OS page cache 가 feature 파일을 완전히 캐싱하기까지 많은 miss 발생.
    warm_up_iter = 1000
    # Setup is Done
    # [한국어] === epoch 루프 === epochs 기본 3 이나 warmup+100 return 으로 1 epoch 내부에서 종료.
    for epoch in tqdm.tqdm(range(args.epochs)):
        epoch_start = time.time()   # [한국어] 참고용 타임스탬프(미사용).
        epoch_loss = 0               # [한국어] loss 누적(진단용).
        train_acc = 0                # [한국어] accuracy 누적 placeholder.
        model.train()                # [한국어] dropout/bn 학습 모드 활성.
        total_loss = 0               # [한국어] 측정 구간 loss 합산.

        batch_input_time = 0         # [한국어] feature fetch 시간 — baseline 은 DGL 내부에서 발생해 여기서 직접 측정 X.
        train_time = 0               # [한국어] forward+backward+step 누적.
        transfer_time = 0            # [한국어] blocks.to(device) 누적.
        e2e_time = 0                 # [한국어] 구간 end-to-end 시간.
        e2e_time_start = time.time() # [한국어] e2e 측정 시작점.

        # [한국어] 이종 baseline DataLoader iter: (input_nodes, seeds, blocks) — ret 없음(GIDS 전용).
        # input_nodes/seeds 는 type→tensor dict. blocks 는 이종 MFG 리스트.
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            if(step % 20 == 0):
                print("step: ", step)   # [한국어] 20 step 마다 진행 로그.
            if(step == warm_up_iter):
                # [한국어] warm-up 종료 경계 — mmap page cache 가 충분히 따뜻해진 시점.
                print("warp up done")
                train_dataloader.print_timer()   # [한국어] DGL 내장 타이머(이종 fork 버전에서 제공되는 보조 메서드).
                # [한국어] 측정 구간 재시작 — 통계 리셋.
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0
                e2e_time_start = time.time()


            # Features are fetched by the baseline GIDS dataloader in ret

            # [한국어] blocks[0].srcdata['feat'] 는 dict[type → tensor].
            # DataLoader(device=device) 지정이므로 이미 GPU 에 올려진 상태.
            # 이 dict 을 그대로 model 의 forward 에 넘기면 RGCN/RSAGE/RGAT 내부에서 type 별 선형변환 수행.
            batch_inputs = blocks[0].srcdata['feat']
            transfer_start = time.time()

            # [한국어] paper type 의 라벨만 사용(category='paper' 가정) — IGBH 표준 predict target.
            #          heterogeneous 이지만 분류는 단일 type 에서만 수행.
            y = blocks[-1].dstdata['label']['paper']

            # [한국어] blocks 의 int() 는 id dtype 을 int32 로 변환(메시지 passing 효율).
            #          .to(device) 는 DataLoader 가 이미 GPU 로 옮긴 경우 no-op 에 가깝지만 안전.
            blocks = [block.int().to(device) for block in blocks]

            transfer_time = transfer_time +  time.time()  - transfer_start

            # Model Training Stage
            # [한국어] --- forward/backward/step ---
            train_start = time.time()
            y_hat = model(blocks, batch_inputs)   # [한국어] 이종 MFG + type→feature dict 전달 → logits 반환.
            loss = loss_fcn(y_hat, y)              # [한국어] CrossEntropy loss.
            optimizer.zero_grad()                  # [한국어] 이전 배치 gradient 초기화.
            loss.backward()                        # [한국어] GPU 상 autograd backward.
            optimizer.step()                       # [한국어] Adam 파라미터 갱신.
            total_loss += loss.item()              # [한국어] scalar 추출(.item() 은 sync 유발).
            cur_train_time = time.time() - train_start
            train_time += cur_train_time

            if(step == warm_up_iter + 100):
                # [한국어] warm_up 이후 정확히 100 iter 측정 → 성능 지표 출력 후 종료.
                #          warm_up_iter(1000) + 100 = 1100 이 측정 종료 지점.
                print("Performance for 100 iteration after 1000 iteration")
                e2e_time += time.time() - e2e_time_start
                train_dataloader.print_timer()
                print_times(transfer_time, train_time, e2e_time)

                # [한국어] 통계 리셋(보고 후 정리 의도, 현재는 return 으로 즉시 종료).
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                e2e_time = 0

                #Just testing 100 iterations remove the next line if you do not want to halt
                # [한국어] 벤치마크 모드: 측정 후 즉시 종료. 실제 수렴 학습을 원하면 이 return 제거.
                return None




    # Evaluation
    # [한국어] === Evaluation (현 경로에서는 위 return 으로 도달하지 않음) ===

    model.eval()          # [한국어] dropout/bn eval 모드 — 확정적 평가.
    predictions = []      # [한국어] 배치별 argmax 결과 누적.
    labels = []           # [한국어] 배치별 정답 누적.
    with torch.no_grad():  # [한국어] gradient tracking 비활성 — 메모리/속도 최적화.
        for _, _, blocks in test_dataloader:
            # [한국어] test DataLoader 는 device= 지정 없음 → 수동으로 blocks.to(device).
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']   # [한국어] 이종 feature dict.

            if(args.data == 'IGB'):
                labels.append(blocks[-1].dstdata['label'].cpu().numpy())   # [한국어] IGBH paper 라벨.
            elif(args.data == 'OGB'):
                # [한국어] 주의: `b` 는 정의되지 않은 변수(원본 버그) — 이 분기 실제 진입 시 NameError.
                #          코드 수정 금지 원칙에 따라 그대로 유지.
                out_label = torch.index_select(label_array, 0, b[1]).flatten()
                labels.append(out_label.numpy())
            predict = model(blocks, inputs).argmax(1).cpu().numpy()   # [한국어] class argmax → CPU numpy.
            predictions.append(predict)

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        # [한국어] 전체 test set accuracy(%).
        test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
    print("Test Acc {:.2f}%".format(test_acc))



       

# [한국어] __main__ - 이종 baseline 엔트리. key_offset 없음(GIDS 경로가 아니므로 feature 는 dataset 내부 host 텐서).
# 실행 컨텍스트: 호스트 Python. CUDA context 는 model.to(device) 시 lazy 생성.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    # [한국어] --path: 데이터셋 루트 디렉토리(IGBH / OGB-MAG 파일 위치).
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M',
        help='path containing the datasets')
    # [한국어] --dataset_size: 스케일 — experimental/small/medium/large/full. large+ 는 Massive 로더.
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    # [한국어] --num_classes: 출력 class 수. IGBH paper 2983(논문 분야), 다른 크기별 348/349/350/153/152.
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983, 348,349, 350, 153, 152], help='number of classes')
    # [한국어] --in_memory: 0 = mmap 읽기 전용(권장), 1 = 전체 메모리 로드. baseline 은 0 이 주 대상.
    parser.add_argument('--in_memory', type=int, default=0,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    # [한국어] --synthetic: 1이면 실제 feature 대신 랜덤 텐서(디버그/성능 측정용).
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    # [한국어] --data: IGB or OGB 데이터셋 계열 선택.
    parser.add_argument('--data', type=str, default='IGB')
    # [한국어] --emb_size: feature 차원 기본 1024 — IGBH/OGB feature 인코딩 기본.
    parser.add_argument('--emb_size', type=int, default=1024)

    # Model
    # [한국어] 이종 모델 선택. 기본 'gcn'은 choices 와 불일치(원본 버그 유지 — --model_type 미지정 시 model 변수 미생성).
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['rgat', 'rsage', 'rgcn'])
    # [한국어] --modelpath/--model_save: 체크포인트 저장 경로/플래그(벤치마크 모드에선 미사용).
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters
    parser.add_argument('--fan_out', type=str, default='10,15')       # [한국어] layer별 fanout("10,15" → 2-layer 10/15 이웃).
    parser.add_argument('--batch_size', type=int, default=1024)       # [한국어] seed(category) 노드 수.
    parser.add_argument('--num_workers', type=int, default=0)         # [한국어] DataLoader subprocess 수(0=main thread).
    parser.add_argument('--hidden_channels', type=int, default=128)    # [한국어] 중간 레이어 폭.
    parser.add_argument('--learning_rate', type=float, default=0.01)   # [한국어] Adam 초기 lr.
    parser.add_argument('--decay', type=float, default=0.001)           # [한국어] L2 weight_decay(baseline 함수는 미전달).
    parser.add_argument('--epochs', type=int, default=3)                # [한국어] 총 epoch 수(벤치마크는 100 iter 후 return).
    parser.add_argument('--num_layers', type=int, default=6)            # [한국어] 모델 내부 레이어 수(fanout 길이와 별개로 사용되는 경우 있음).
    parser.add_argument('--num_heads', type=int, default=4)             # [한국어] RGAT attention head 수.
    parser.add_argument('--log_every', type=int, default=2)             # [한국어] 로깅 주기.

    #GIDS parameter
    # [한국어] baseline 에서는 실제 호출되지 않지만 CLI 호환을 위해 수용.
    # 공통 쉘 스크립트가 동일 인자 세트로 모든 스크립트를 호출하기 때문.
    parser.add_argument('--GIDS', action='store_true', help='Enable GIDS Dataloader')   # [한국어] 플래그 — baseline 에서 무시.
    parser.add_argument('--num_ssd', type=int, default=1)                                 # [한국어] 원래는 BaM Controller 수.
    parser.add_argument('--cache_size', type=int, default=8)                              # [한국어] 원래는 BaM page_cache MB.
    parser.add_argument('--uva', type=int, default=0)                                     # [한국어] 원래는 UVA feature 경로 플래그.
    parser.add_argument('--uva_graph', type=int, default=0)                               # [한국어] 원래는 UVA graph 경로 플래그.
    parser.add_argument('--wb_size', type=int, default=6)                                 # [한국어] 원래는 window buffer depth.

    parser.add_argument('--device', type=int, default=0)                                  # [한국어] CUDA device 번호.

    #GIDS Optimization
    # [한국어] Accumulator/CPU buffer 관련 — baseline 에서 전부 무시.
    parser.add_argument('--accumulator', action='store_true', help='Enable Storage Access Accmulator')
    parser.add_argument('--bw', type=float, default=5.8, help='SSD peak bandwidth in GB/s')
    parser.add_argument('--l_ssd', type=float, default=11.0, help='SSD latency in microseconds')
    parser.add_argument('--l_system', type=float, default=20.0, help='System latency in microseconds')
    parser.add_argument('--peak_percent', type=float, default=0.95)

    parser.add_argument('--num_iter', type=int, default=1)                                 # [한국어] 반복 측정 횟수(스위프용).

    parser.add_argument('--cpu_buffer', action='store_true', help='Enable CPU Feature Buffer')   # [한국어] Constant CPU Buffer 활성(baseline 무시).
    parser.add_argument('--cpu_buffer_percent', type=float, default=0.2, help='CPU feature buffer size (0.1 for 10%)')
    parser.add_argument('--pin_file', type=str, default="/mnt/nvme16/pr_full.pt",
        help='Pytorch Tensor File for the list of nodes that will be pinned in the CPU feature buffer')

    parser.add_argument('--window_buffer', action='store_true', help='Enable Window Buffering')   # [한국어] Window Buffering 활성(baseline 무시).



    #GPU Software Cache Parameters
    # [한국어] BaM page_cache 관련 파라미터 — baseline 에서 사용되지 않지만 CLI 호환 수용.
    parser.add_argument('--page_size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD')
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)')
    parser.add_argument('--cache_dim', type=int, default=1024) #CHECK


    args = parser.parse_args()
    # [한국어] 설정 요약 출력 — baseline 실행 로그에서도 GIDS 경로 여부 식별 가능.
    print("GIDS DataLoader Setting")
    print("GIDS: ", args.GIDS)
    print("CPU Feature Buffer: ", args.cpu_buffer)
    print("Window Buffering: ", args.window_buffer)
    print("Storage Access Accumulator: ", args.accumulator)


    labels = None   # [한국어] OGB 경로에서만 별도 label 텐서를 넘기는 용도 — 현 경로 None.
    # [한국어] device 문자열. CUDA 없으면 CPU 폴백(실전은 GPU 전제).
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    # [한국어] === 데이터셋 선택 ===
    if(args.data =='OGB'):
        dataset = OGBHeteroDGLDatasetMassive(args)   # [한국어] OGB-MAG 대형 이종(paper/author/institution/fieldofstudy).

    else:
        # [한국어] IGB 계열: 크기에 따라 Massive/일반 로더 분기.
        # large/full 은 Massive (mmap + lazy 로딩), 그 외는 일반 버전(보다 편의적 API).
        if(args.dataset_size == 'full' or args.dataset_size == 'large'):
            dataset = IGBHeteroDGLDatasetMassive(args)
        else:
            dataset = IGBHeteroDGLDataset(args)
    g = dataset[0]    # [한국어] 첫 graph 반환(단일 이종 graph).
    print("format: ", g.formats())
    g  = g.formats('csc')   # [한국어] CSC 로 변환(NeighborSampler 효율 — in-neighbor 샘플링 시 column access).
    print("format: ", g.formats())
    # [한국어] category: 예측 대상 node type(g.predict 는 DGL heterograph 에 저장된 예측 타겟).
    category = g.predict

    # g_etypes = g.canonical_etypes
    # graph_dict = {}
    # for i in range(len(g_etypes)):
    #     graph_dict[g_etypes[i]] = ('csc', g.adj_sparse('csc',  etype=g_etypes[i]))

    # torch.save(graph_dict, "/mnt/nvme22/OGB_graph_csc.pth")

    # category = g.predict
    # print("g: ", g)
    # [한국어] 위 주석 블록은 원본 저자가 edge type별 CSC adjacency 를 별도 파일로 저장하는 실험 흔적.



    # [한국어] 학습 진입(baseline). label_array 는 미지정(None) — IGB 경로 기본.
    track_acc_Baseline(g, category, args, device)


  



