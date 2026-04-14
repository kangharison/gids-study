"""
[한국어 설명] 이종(heterogeneous) 그래프용 GIDS+BaM 학습 스크립트 (heterogeneous_train.py)

=== 파일의 역할 ===
IGBH(IGB-Heterogeneous) / OGB-MAG 와 같이 노드 타입이 여러 개인 이종 그래프에 대해
GNN 을 학습한다. RGCN/RSAGE/RGAT(여기선 RGNN with rgat) 모델을 사용하며,
feature 는 node type 마다 SSD 상 서로 다른 오프셋에 저장되어 있다
(`key_offset`/`heterograph_map` 으로 전달). GIDS 는 BaM page_cache 로 단일
1D 주소공간을 가지므로, 각 node type 의 글로벌 index = type_offset + local_index 로
접근한다.

=== 전체 아키텍처에서의 위치 ===
동종 버전과 동일하게 "Python 학습 스크립트" 단계. 호출 체인:
`python heterogeneous_train.py` → `__main__` → `track_acc_GIDS(g, category, args, device, dim, labels, key_offset)`
→ `GIDS.GIDS(..., heterograph=True, heterograph_map=key_offset)` 로 C++ 측에
이종 모드 플래그 전달 → `GIDS_DGLDataLoader` 가 이종 샘플러의 input_nodes 를
type 별로 분리 → 각 type 의 node id + offset 을 합성하여 BaM 에서 fetch.
실행 컨텍스트: 호스트 Python / GPU CUDA.

=== 타 모듈과의 연결 ===
- `models.py`: RGCN/RSAGE/RGAT.
- `mlperf_model.py`: RGNN (MLPerf 레퍼런스 모델).
- `dataloader.py`: IGBHeteroDGLDataset(Massive), OGBHeteroDGLDatasetMassive.
- `GIDS`: heterograph 모드 활성화 시 `heterograph_map` 을 C++
  BAM_Feature_Store<float>::set_heterograph_map 등으로 전달.
- 데이터 흐름: SSD 내 type별 feature → BaM page_cache(GPU, 단일 페이지 캐시) →
  ret(dict 또는 concatenated) → model(blocks, batch_inputs).

=== 주요 함수/구조체 요약 ===
- `fetch_data_chunk`, `print_times`: 유틸.
- `track_acc_GIDS(g, category, args, device, dim, label_array, key_offset)`: 이종
  GNN 학습의 메인. category(예: 'paper') 가 예측 대상 node type.
- `__main__`: argparse + key_offset 테이블(데이터셋 크기별로 하드코딩) + 학습 호출.
  `key_offset` 은 node type → SSD feature 시작 인덱스(element 단위) 매핑이다.
"""

# [한국어] accuracy 계산용.
import sklearn.metrics

import dgl
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
# [한국어] MLPerf 레퍼런스 RGNN(rgat 등 포함).
from mlperf_model import RGNN
from dataloader import IGB260MDGLDataset, OGBDGLDataset
# [한국어] 이종 그래프 로더 — 크기별로 세 가지 클래스 제공.
from dataloader import IGBHeteroDGLDataset, IGBHeteroDGLDatasetMassive, OGBHeteroDGLDatasetMassive

import csv
import argparse, datetime
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

import GIDS
from GIDS import GIDS_DGLDataLoader

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")


# [한국어] fetch_data_chunk - NVTX 마커 래퍼(디버그용).
@nvtx.annotate("fetch_data_chunk()", color="blue")
def fetch_data_chunk(test, out_t, page_size, stream_id):
    test.fetch_from_backing_memory_chunk(out_t.data_ptr(), page_size, stream_id)


# [한국어] print_times - 단계별 시간 로깅.
def print_times(transfer_time, train_time, e2e_time):
    print("transfer time: ", transfer_time)
    print("train time: ", train_time)
    print("e2e time: ", e2e_time)


# [한국어]
# track_acc_GIDS - 이종 그래프 GNN 학습 메인
#
# @param g: DGL heterograph(csc). 여러 node type(예: paper/author/fos/institute/...).
# @param category: 예측 대상 node type 이름(예: 'paper'). g.predict 로부터 결정.
# @param args: argparse 네임스페이스.
# @param device: "cuda:N".
# @param dim: feature 차원(실제로는 args.emb_size 로 덮어씌워짐 — 아래 참조).
# @param label_array: OGB 외부 label 텐서(옵션).
# @param key_offset: {node_type: element_offset} — 각 type feature 의 SSD 내 시작 인덱스.
#                    GIDS heterograph_map 으로 전달되어 GPU fetch 커널이 올바른
#                    global index(= type_offset + local_idx) 를 계산하게 한다.
# @return: None(warm_up_iter=35000 이후 100 iter 측정 후 원래는 break 자리이나 현재 주석 처리됨,
#          epoch 루프 종료 후 evaluation 수행).
#
# 호출 체인: __main__ → track_acc_GIDS → GIDS.GIDS(heterograph=True) → 내부
#   BAM_Feature_Store 가 heterograph_map 을 가진 상태로 read_feature_kernel 호출.
#   read_feature_kernel 은 입력 node id 와 함께 type 정보를 사용해 bam_ptr.read(global_idx).
def track_acc_GIDS(g, category, args, device, dim , label_array=None, key_offset=None):

    # [한국어] === GIDS Loader 초기화 (heterograph=True) ===
    GIDS_Loader = None
    GIDS_Loader = GIDS.GIDS(
        page_size = args.page_size,
        off = args.offset,
        num_ele = args.num_ele,
        num_ssd = args.num_ssd,
        cache_size = args.cache_size,
        window_buffer = args.window_buffer,
        wb_size = args.wb_size,
        accumulator_flag = args.accumulator,
        cache_dim = args.cache_dim,
        #ssd_list=[5],
        heterograph = True,               # [한국어] 이종 그래프 모드 ON → BAM_Feature_Store 내부 타입 분기 활성.
        heterograph_map = key_offset      # [한국어] {type: offset} 딕셔너리. C++ 템플릿이 type별 start index 로 사용.
    )

    dim = args.emb_size   # [한국어] 인자로 넘어온 dim 무시하고 args.emb_size 로 덮어씀(코드 관습).

    # [한국어] Accumulator 파라미터 전달(옵션).
    if(args.accumulator):
        GIDS_Loader.set_required_storage_access(args.bw, args.l_ssd, args.l_system, args.num_ssd, args.peak_percent)


    # [한국어] CPU Backing Buffer(핫 노드 pin) — 이종 그래프에서도 동일 API.
    # 주의: g.number_of_nodes() 는 이종 그래프 전체 합. type 별 아님.
    if(args.cpu_buffer):
        num_nodes = g.number_of_nodes()
        num_pinned_nodes = int(num_nodes * args.cpu_buffer_percent)
        GIDS_Loader.cpu_backing_buffer(dim, num_pinned_nodes)
        pr_ten = torch.load(args.pin_file)
        GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned_nodes)


    # [한국어] 이종용 NeighborSampler — 각 레이어마다 edge type별로 fanout 이웃을 샘플링.
    sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])


    # g.ndata['features'] = g.ndata['feat']
    # g.ndata['labels'] = g.ndata['label']

    # [한국어] 예측 대상 type(=category) 의 split mask 로부터 node id 추출.
    train_nid = torch.nonzero(g.nodes[category].data['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.nodes[category].data['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.nodes[category].data['test_mask'], as_tuple=True)[0]

    in_feats = dim   # [한국어] 이종 모델은 모든 type feature 가 같은 dim 을 가진다고 가정.

    # [한국어] === 모델 선택 (R-prefix = Relational, 이종 그래프 대응 버전) ===
    if args.model_type == 'rgcn':
        # [한국어] RGCN: edge type별 별도 weight matrix.
        model = RGCN(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rsage':
        model = RSAGE(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rgat':
        # model = RGAT(g.etypes, in_feats, args.hidden_channels,
        #     args.num_classes, args.num_layers, 0.2, args.num_heads).to(device)

         # [한국어] MLPerf 레퍼런스 RGNN(rgat 모드). node_type='paper' 는 예측 대상 지정.
         model = RGNN(g.etypes,
               in_feats,
               args.hidden_channels,
               args.num_classes,
               num_layers=args.num_layers,
               dropout=0.2,
               model='rgat',
               heads=args.num_heads,
               node_type='paper').to(device)


    # [한국어] === 학습용 GIDS DataLoader (이종) ===
    # 시드는 {category: train_nid} 딕셔너리. 내부적으로 sampler 가 이종 input_nodes
    # (dict) 를 반환하고, GIDS_DGLDataLoader 가 heterograph_map 을 사용해 각 type 별
    # global index 로 변환 후 한 번의 BAM fetch 로 모든 feature 를 ret 로 얻는다.
    #train_dataloader = dgl.dataloading.DataLoader(
    train_dataloader =  GIDS_DGLDataLoader(
        g,
        {category: train_nid},
        sampler,
        args.batch_size,
        dim,
        GIDS_Loader,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_alternate_streams=False
    )

    # [한국어] validation: 일반 DGL DataLoader(현 경로 미사용).
    val_dataloader = dgl.dataloading.DataLoader(
        g, {category: val_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)


    # [한국어] test: 평가에서도 GIDS 경로 사용(대용량 feature 라서).
    test_dataloader =  GIDS_DGLDataLoader(
        g,
        {category: test_nid},
        sampler,
        args.batch_size,
        dim,
        GIDS_Loader,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_alternate_streams=False
    )


    # test_dataloader = dgl.dataloading.DataLoader(
    #     g, {category: test_nid}, sampler,
    #     batch_size=args.batch_size,
    #     shuffle=True, drop_last=False,
    #     num_workers=args.num_workers)


    # [한국어] CrossEntropy + Adam. weight_decay 는 이종 모델에선 주석 처리(대형 그래프 학습의 관례).
    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
        #, weight_decay=args.decay
        )

    # [한국어] warm_up_iter 가 35,000 로 매우 큼 — 이종 대형 그래프는 초반 hit/miss 분포 안정까지 오래 걸리기 때문.
    warm_up_iter = 35000
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

        # [한국어] 이종 GIDS_DGLDataLoader iter: input_nodes/seeds 는 dict(type → node id 텐서).
        # blocks 는 여러 edge type 을 포함한 MFG. ret 는 모든 type feature 를 합쳐 둔 1D/2D 텐서.
        for step, (input_nodes, seeds, blocks, ret) in enumerate(train_dataloader):


            # p = blocks[0].srcdata[dgl.NID]['paper'].cpu()
            # a = blocks[0].srcdata[dgl.NID]['author'].cpu()
            # print(f"paper node: {p} author node:{a} emb: {ret}")
            # p_orig_feat = g.ndata['feat']['paper'][p]
            # print(f"paper feat: {p_orig_feat}")
           
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

            # [한국어] ret: GPU 상 feature 텐서(이미 BaM fetch 완료).
            batch_inputs = ret
            transfer_start = time.time()

            # [한국어] 'paper' type 의 정답 label 만 추출(category='paper' 가정) → device로 이동.
            batch_labels = (blocks[-1].dstdata['label']['paper']).to(device)

            # [한국어] blocks 를 device 로. int() 는 id dtype 을 int32 로.
            blocks = [block.int().to(device) for block in blocks]
            transfer_time = transfer_time +  time.time()  - transfer_start

            # Model Training Stage
            # [한국어] --- forward/backward/step ---
            train_start = time.time()
            batch_pred = model(blocks, batch_inputs)   # [한국어] RGNN forward — 이종 MFG 처리.
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
                #break
         

            # [한국어] 1000 iter 마다 실시간 학습 정확도 출력 — 수렴 모니터링용.
            if (step % 1000 == 0):
                train_acc =sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(),
                  batch_pred.argmax(1).detach().cpu().numpy())*100
                print(f"Step {step}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.2f}%")

            # if(step == 5000):
            #     break




       
    # [한국어] === Evaluation (test set 전체 순회) ===
    model.eval()
    predictions = []
    labels = []
    counter = 0
    eval_acc = 0
    with torch.no_grad():
        #for _, _, blocks in test_dataloader:
        # [한국어] test_dataloader 도 GIDS 경로 — ret 로 feature 직접 수신.
        for step, (input_nodes, seeds, blocks, ret) in enumerate(test_dataloader):


            # blocks = [block.to(device) for block in blocks]
            # inputs = blocks[0].srcdata['feat']

            batch_labels = (blocks[-1].dstdata['label']['paper']).to(device)
            batch_inputs = ret

            blocks = [block.int().to(device) for block in blocks]


            if(args.data == 'IGB'):
                labels.append(blocks[-1].dstdata['label']['paper'].cpu().numpy())
            elif(args.data == 'OGB'):
                # [한국어] 주의: 여기의 `b` 는 미정의 변수(원본 버그). 코드 수정 금지 원칙 유지.
                out_label = torch.index_select(label_array, 0, b[1]).flatten()
                labels.append(out_label.numpy())
            #predict = model(blocks, inputs).argmax(1).cpu().numpy()
            predict = model(blocks, batch_inputs)
            #predictions.append(predict)

            # [한국어] 배치 단위 정확도 계산 후 누적(배치 개수로 평균).
            train_acc = sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(),
                  predict.argmax(1).detach().cpu().numpy())*100
            eval_acc += train_acc
            if(counter % 1000 == 0):
                print(f"Step {step}, Eval Acc: {train_acc:.2f}%")

            # if(counter == 5000):
            #     break

            counter += 1
    

        # predictions = np.concatenate(predictions)
        # labels = np.concatenate(labels)
        # test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100

        test_acc = eval_acc/counter
    print("Test Acc {:.2f}%".format(test_acc))




    
       

# [한국어] __main__ - 이종 그래프 학습 엔트리. key_offset(node type → SSD feature 시작 index) 구성이 핵심.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M',
        help='path containing the datasets')
    # [한국어] --dataset_size: 이종 버전은 tiny/small/medium/large/full 지원.
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['tiny', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    # [한국어] --num_classes: IGBH paper 기준 2983 / full 153 등 다양.
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983, 172, 348,349, 350, 153, 152], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)
    
    # Model
    # [한국어] 이종 모델 3종: rgat/rsage/rgcn. default 'gcn' 은 매칭되지 않아 모델 미설정 버그 가능(원본 유지).
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['rgat', 'rsage', 'rgcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters
    parser.add_argument('--fan_out', type=str, default='10,15')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    # [한국어] 이종 기본 hidden=512(동종보다 큼) — 모델 용량 확보.
    parser.add_argument('--hidden_channels', type=int, default=512)
    # [한국어] 이종 기본 lr=0.001(동종 0.01 보다 작게) — MLPerf 추천값.
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=2)

    #GIDS parameter
    # [한국어] GIDS 활성 / BaM 컨트롤러 수 / page cache 크기 등 — 동종 버전과 동일 의미.
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

    # [한국어] CPU Feature Buffer 설정.
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
    # [한국어] key_offset: {node_type: SSD feature 시작 element index}. GIDS heterograph_map 으로 전달.
    # 각 type 의 feature 는 type별 노드 수만큼의 element 를 차지하며, 뒤 type 은 앞 type 노드 수를 누적한 위치에 저장.
    key_offset = None
    dim = 1024
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        dim = 1024   # [한국어] IGBH feature dim 고정.
        print("Dataset: IGB")
        if(args.dataset_size == 'full' or args.dataset_size == 'large'):
            dataset = IGBHeteroDGLDatasetMassive(args)   # [한국어] 대형 데이터셋 전용 로더(mmap 최적화).
            # User need to fill this out for their dataset based how it is stored in SSD
            # [한국어] IGBH-full: paper 269,346,174 노드 → author는 그 뒤부터 시작, 이하 누적.
            if(args.dataset_size == 'full'):
                key_offset = {
                    'paper' : 0,                  # [한국어] paper feature 는 offset 0 부터.
                    'author' : 269346174,         # [한국어] paper 총 노드 수.
                    'fos' : 546567057,             # [한국어] paper + author.
                    'institute' : 547280017,
                    'journal' : 546593975,        # [한국어] 주의: 값 배치가 비오름차순으로 보이나 원본 유지.
                    'conference' : 546643027
                }
            else:
                # [한국어] IGBH-large: paper 100M 기준. 누적합 형식으로 명시.
                key_offset = {
                    'paper' : 0,
                    'author' : 100000000,
                    'fos' : 100000000 + 116959896,
                    'institute' : 100000000 + 116959896 + 649707,
                    'journal' : 100000000 + 116959896 + 649707 + 26524,
                    'conference' : 100000000 + 116959896 + 649707 + 26524 + 48820
                }

        else:
            # [한국어] small/medium/tiny: IGBHeteroDGLDataset (일반 버전).
            dataset = IGBHeteroDGLDataset(args)
            if(args.dataset_size == 'small'):
                # [한국어] small: paper 1M 기준. journal/conference 는 현재 subset 에 포함 안됨(주석 처리).
                key_offset = {
                    'paper' : 0,
                    'author' : 1000000,
                    'fos' : 1000000 + 192606,
                    'institute' : 1000000 + 192606 + 190449,
                    # 'journal' : 1000000 + 192606 + 190449 + 14751,
                    # 'conference' : 1000000 + 192606 + 190449 + 14751 + 15277
                }
            elif(args.dataset_size == 'medium'):
                key_offset = {
                    'paper' : 0,
                    'author' : 10000000,
                    'fos' : 10000000 + 15544654,
                    'institute' : 10000000 + 15544654 + 415054,
                    # 'journal' : 10000000 + 15544654 + 415054 + 23256,
                    # 'conference' : 10000000 + 15544654 + 415054 + 23256 + 37565
                }
            elif(args.dataset_size == 'tiny'):
                # [한국어] tiny: paper 100k 기준. 테스트용.
                key_offset = {
                    'paper' : 0,
                    'author' : 100000,
                    'fos' : 100000 + 357041,
                    'institute' : 100000 + 357041 + 84220
                }
            else:
                # [한국어] experimental 등 매칭 안되는 크기 → 즉시 종료.
                key_offset = None
                print("key_offset is not set")
                exit()

        g = dataset[0]
        g = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBHeteroDGLDatasetMassive(args)   # [한국어] OGB-MAG 대형 이종 로더.
        g = dataset[0]
        g = g.formats('csc')
    else:
        g=None
        dataset=None
    
    # nt = g.ntypes

    # for t in nt:
    #     num_t = g.num_nodes(t)
    #     print("type: ", t, " num: ", num_t)


    # [한국어] g.predict: DGL heterograph 에 내장된 "예측 대상 type" (예: 'paper').
    category = g.predict
    print(f"GIDS trainign start key pffset: {key_offset}")
    # [한국어] 학습 시작. key_offset 이 GIDS.GIDS 내부에서 heterograph_map 으로 전달됨.
    track_acc_GIDS(g, category, args, device, dim, labels, key_offset)
    #track_acc(g, args, device, labels)




