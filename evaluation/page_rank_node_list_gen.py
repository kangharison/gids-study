"""
[한국어] Constant CPU Buffer 용 PageRank 상위 노드 리스트 생성
        (page_rank_node_list_gen.py)

=== 파일의 역할 ===
IGB/OGB (homogeneous / heterogeneous) 그래프 전체에 대해 PageRank 를 일정
iteration(K) 동안 반복 계산한 뒤, **상위 60% 중요 노드의 인덱스**를 텐서로 저장
한다. 저장된 텐서(예: pr_full.pt) 는 학습 시 --pin_file 인자로 GIDS 에 전달되어
GIDS_CPU_buffer 가 host-pinned memory 에 고정하는 "핫 노드 feature" 집합으로
사용된다(= Constant CPU Buffer).

=== 전체 아키텍처에서의 위치 ===
(전처리) page_rank_node_list_gen.py → torch.save(indices, out_path) → 학습
시점 run_GIDS_IGBH.sh --pin_file <path> → GIDS.GIDS(cpu_buffer=True) →
BAM_Feature_Store::cpu_buffer_init → range_t::set_cpu_buffer 로 매핑.
CPU 버퍼에 올라간 노드는 GPU 커널이 SSD 대신 pinned host 로부터 feature 를
읽어 SSD IOPS 를 절약한다(read_feature_kernel_with_cpu_backing 경로).

=== 타 모듈과의 연결 ===
- dataloader.py: 동일한 DGLDataset 클래스들을 사용해 그래프를 로드.
- DGL update_all / multi_update_all: heterogeneous PageRank 구현 핵심.
- torch.topk: 상위 k 노드 선별. k = int(N × 0.6).
- GIDS.py 의 cpu_buffer_init 와 반드시 key_offset 매핑이 일관되어야 한다.

=== 주요 함수/구조체 요약 ===
- compute_pagerank_hetero(g, DAMP, K, key_offset):
    이종 그래프에서 node type 별 PV 벡터 초기화 + 각 edge type 별
    in-degree 정규화 후 multi_update_all 로 PV 갱신을 K 회 반복.
- compute_pagerank(g, DAMP, K, N):
    동종 그래프용 단순 PageRank (in-degree 정규화 + copy/sum).
- main:
    argparse → dataset 로드 → PR 계산 → topk → torch.save.
"""

import sklearn.metrics                               # [한국어] import 호환성용. 직접 사용은 없음.

import dgl                                           # [한국어] DGL. PageRank 메시지 전파 API 사용.
import argparse, datetime                            # [한국어] CLI 파싱 / 타임스탬프.

import torch, torch.nn as nn, torch.optim as optim    # [한국어] 텐서/선택적 모듈. PR 계산에 torch 텐서만 사용.
import time, tqdm, numpy as np                        # [한국어] 시간/진행표시/넘파이 배열.
from models import *                                   # [한국어] models 임포트 호환성 목적.
from dataloader import IGB260MDGLDataset, OGBDGLDataset, IGBHeteroDGLDataset, IGBHeteroDGLDatasetMassive, OGBHeteroDGLDatasetMassive
# [한국어] dataloader 의 모든 데이터셋 클래스 임포트 — PR 계산에 사용할 그래프를 구성.
import csv                                             # [한국어] 로그 CSV 후보.
import warnings                                         # [한국어] deprecation 경고 억제.


from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
# [한국어] OGB 벤치마크 접근용. 직접 사용은 미미하지만 평가 환경 일관성 위해 import.

import networkx as nx                                  # [한국어] NetworkX. 현재는 직접 사용 X — 원본 흔적.
import dgl.function as fn                              # [한국어] DGL 내장 메시지 함수 (copy_u, sum).

torch.manual_seed(0)                                    # [한국어] 재현성 — train/val/test mask 등 난수 시드.
dgl.seed(0)                                             # [한국어] DGL 내부 난수도 고정.
warnings.filterwarnings("ignore")                       # [한국어] 반복 경고 억제.

# [한국어]
# compute_pagerank_hetero - 이종 그래프에서 K 회 PageRank 반복.
# @param g         : DGL heterograph.
# @param DAMP      : damping factor (전형 0.85).
# @param K         : 반복 횟수.
# @param key_offset: 각 node type 이 전역 PV 벡터에서 차지하는 시작 오프셋.
# @return          : 초기 pv (원본 구현상 반환되지만, 실제로는 g.nodes[ntype].data['pv'] 에 결과 저장).
# 동작: 각 etype 별 in-degree 정규화 후 multi_update_all 로 PV 전파. 매 step
#       모든 ntype 에 대해 PV ← (1-DAMP)/N + DAMP·PV 업데이트.
def compute_pagerank_hetero(g, DAMP, K, key_offset):
    # Assuming N is the total number of nodes across all types
    N = sum([g.number_of_nodes(ntype) for ntype in g.ntypes])            # [한국어] 전체 노드 수(타입별 합).
    pv_init_value = torch.ones(N) / N                                     # [한국어] 초기 PV 균등 분포 1/N.
    degrees = torch.zeros(N, dtype=torch.float32)                         # [한국어] 현재 미사용. 원본의 placeholder.

    # Initialize PageRank values for each node type
    for ntype in g.ntypes:                                                # [한국어] 각 node type 마다 PV 슬라이스 할당.
        print("node type: ", ntype)                                       # [한국어] 로그.
        print("offset : ",key_offset[ntype])                              # [한국어] key_offset: ntype → 전역 PV 시작 인덱스.
        n = g.number_of_nodes(ntype)                                      # [한국어] 해당 ntype 노드 수.
        offset = key_offset[ntype]                                        # [한국어] 전역 offset.
        g.nodes[ntype].data['pv'] = pv_init_value[offset:offset+n]        # [한국어] PV 텐서를 slicing 해 ntype 에 저장.



    func_dict = {}                                                         # [한국어] etype → (msg_fn, reduce_fn) 매핑.
    for etype in g.etypes:                                                 # [한국어] 모든 etype 동일한 copy/sum 메시지 함수.
        func_dict[etype] = (fn.copy_u('pv','m'), fn.sum('m', 'pv'))
        # [한국어] src 의 pv 를 메시지로 복사 → dst 에서 합산 → pv 업데이트.


    for k in range(K):                                                     # [한국어] K 회 반복.
        for etype in g.etypes:                                             # [한국어] 각 edge type 에서 src PV 를 in-degree 로 정규화.
            #print("etype: ", etype)

            cur_degrees = g.in_degrees(v='__ALL__', etype=etype).type(torch.float32)
            # [한국어] 해당 etype 의 dst 별 in-degree. '__ALL__' = 전체 노드.
            if(etype == 'affiliated_to'):                                  # [한국어] src ntype=author → dst=institute.
                g.nodes['institute'].data['pv'] /=  cur_degrees
            elif(etype == 'cites'):                                        # [한국어] paper→paper self-relation.
                g.nodes['paper'].data['pv'] /= cur_degrees
            elif(etype == 'topic'):                                        # [한국어] paper→fos.
                g.nodes['fos'].data['pv'] /= cur_degrees
            elif(etype == 'written_by'):                                   # [한국어] paper→author.
                g.nodes['author'].data['pv'] /= cur_degrees
                # [한국어] 주: 구현 편의상 dst 노드의 pv 를 해당 etype in-degree 로 나눠
                #        이후 multi_update_all 에서 src→dst 전파 시 per-edge 정규화 효과.



        g.multi_update_all(func_dict, cross_reducer="sum")                 # [한국어] etype 별 msg 전파 후 서로 다른 etype 결과를 sum.

        for ntype in g.ntypes:                                             # [한국어] PageRank damping 보정.
            g.nodes[ntype].data['pv'] = (1 - DAMP) / N + DAMP * g.nodes[ntype].data['pv']
            # [한국어] 표준 PR 식: PV ← (1-d)/N + d · PV_new.
    return pv_init_value                                                    # [한국어] 초기값 반환(주의: 결과는 g.nodes[ntype].data['pv'] 에 누적됨).


#
# [한국어]
# compute_pagerank - 동종 그래프 전용 PageRank.
# @param g: homogeneous DGL graph.
# @param DAMP: damping factor.
# @param K: iteration.
# @param N: 노드 수.
# @return: g.ndata['pv'] 최종 PV 텐서.
def compute_pagerank(g, DAMP, K, N):
    g.ndata['pv'] = torch.ones(N) / N                                      # [한국어] 초기 PV.
    degrees = g.in_degrees(g.nodes()).type(torch.float32)                  # [한국어] 모든 노드 in-degree.
    for k in range(K):                                                     # [한국어] K 회 반복.
        g.ndata['pv'] = g.ndata['pv'] / degrees                            # [한국어] PV 를 in-degree 로 나눠 per-edge 기여도 계산.
        g.update_all(message_func=fn.copy_u('pv', 'm'),                    # [한국어] src pv 를 메시지로 전달.
                     reduce_func=fn.sum('m', 'pv'))                        # [한국어] dst 에서 합산 → 새 pv.
        g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['pv']              # [한국어] damping 보정.
    return g.ndata['pv']                                                    # [한국어] 최종 PV.

if __name__ == '__main__':
    # [한국어] CLI — 데이터셋 선택/PageRank 하이퍼/출력 경로.
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983, 171,172, 173], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--out_path', type=str, default='./pr_full.pt', 
        help='output path for the node list')
    parser.add_argument('--damp', type=float, default=0.85, 
        help='Damp value for the page rank algorithm')
    parser.add_argument('--K', type=int, default=20, 
        help='K value for the page rank algorithm')
    parser.add_argument('--device', type=int, default=0, 
        help='cuda device number')
    parser.add_argument('--data', type=str, default='IGB', 
        choices=['IGB', 'OGB'], help='Dataset type')
    parser.add_argument('--uva_graph', type=int, default=0,help='0:non-uva, 1:uva')
    parser.add_argument('--emb_size', type=int, default=1024)
    
    parser.add_argument('--hetero', action='store_true', help='Heterogenous Graph')
 

    args = parser.parse_args()                                             # [한국어] CLI 파싱.

    labels = None                                                           # [한국어] 후처리용 placeholder. 현재 사용 X.
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    # [한국어] GPU 가용 시 cuda:N. 현재 PR 은 CPU 연산이지만 device 변수는 유지.


    if(args.hetero):                                                        # [한국어] 이종 그래프 모드.
        if(args.data == 'IGB'):                                             # [한국어] IGB Heterogeneous (IGBH).
            print("Dataset: IGBH")
            if(args.dataset_size == 'full' or args.dataset_size == 'large'):
                dataset = IGBHeteroDGLDatasetMassive(args)                  # [한국어] journal/conference 포함 풀셋 로더.
            else:
                dataset = IGBHeteroDGLDataset(args)                         # [한국어] small/medium 용 경량 로더.

            g = dataset[0]                                                  # [한국어] heterograph 추출.
            #g  = g.formats('csc')
        elif(args.data == "OGB"):                                           # [한국어] OGB-MAG Heterogeneous.
            print("Dataset: MAG")
            dataset = OGBHeteroDGLDatasetMassive(args)
            g = dataset[0]
            #g  = g.formats('csc')
        else:
            g=None
            dataset=None


    else:                                                                   # [한국어] 동종 그래프 모드.
        if(args.data == 'IGB'):
            print("Dataset: IGB")
            dataset = IGB260MDGLDataset(args)
            g = dataset[0]
            g  = g.formats('csc')                                          # [한국어] CSC 포맷으로 변환 — in-degree 질의가 빠르다.
        elif(args.data == "OGB"):
            print("Dataset: OGB")
            dataset = OGBDGLDataset(args)
            g = dataset[0]
        else:
            g=None
            dataset=None


    pr_val = None                                                           # [한국어] placeholder — 최종 PR 값은 pv 변수에.
    N = 0                                                                    # [한국어] 노드 수.
    if(args.hetero):                                                        # [한국어] 이종 그래프 PR 경로.
        print("heterogeneous graph")

        print("g.number_of_nodes(): ", g.number_of_nodes())                 # [한국어] 전체 노드 수 로깅.
        # [한국어] 아래 key_offset 들은 각 ntype 의 전역 PV 벡터상 시작 인덱스.
        #          IGBH 의 각 데이터셋 크기마다 노드 수가 달라 상수 테이블로 유지.
        #          이 값은 GIDS.GIDS(heterograph_map=...) 의 SSD offset 규약과
        #          논리적으로 동형(isomorphic)이어야 한다.
        if(args.dataset_size == 'full'):
            print("full dataset")
            key_offset = {
                'paper' : 0,
                'author' : 269346174,
                'fos' : 546567057,
                'institute' : 547280017,
                'journal' : 546593975,
                'conference' : 546643027
            }
        elif(args.dataset_size == 'large'):
            print("large dataset")
            key_offset = {
                'paper' : 0,
                'author' : 100000000,
                'fos' : 100000000 + 116959896,
                'institute' : 100000000 + 116959896 + 649707,
                'journal' : 100000000 + 116959896 + 649707 + 26524,
                'conference' : 100000000 + 116959896 + 649707 + 26524 + 48820
            }
        elif(args.dataset_size == 'medium'):
            print("medium dataset")
            key_offset = {
                'paper' : 0,
                'author' : 10000000,
                'fos' : 10000000 + 15544654,
                'institute' : 10000000 + 15544654 + 415054,
                # 'journal' : 10000000 + 15544654 + 415054 + 23256,
                # 'conference' : 10000000 + 15544654 + 415054 + 23256 + 37565
            }
        elif(args.dataset_size == 'small'):
            print("small dataset")
            key_offset = {
                'paper' : 0,
                'author' : 1000000,
                'fos' : 1000000 + 192606,
                'institute' : 1000000 + 192606 + 190449,
                # 'journal' : 1000000 + 192606 + 190449 + 14751,
                # 'conference' : 1000000 + 192606 + 190449 + 14751 + 15277
            }
        elif(args.dataset_size == 'tiny'):
            print("tiny dataset")
            key_offset = {
                'paper' : 0,
                'author' : 100000,
                'fos' : 100000 + 357041,
                'institute' : 100000 + 357041 + 84220
            }
        else:
            key_offset = None                                               # [한국어] 지원하지 않는 크기면 중단.
            print("key_offset is not set")
            exit()

        pv = compute_pagerank_hetero(g, args.damp, args.K, key_offset)      # [한국어] 이종 PR 실행. pv 자체는 초기값이고, 결과는 g.nodes[ntype].data['pv'].
        N = len(pv)                                                         # [한국어] 전역 노드 수.
        # [한국어] 주의: 아래 topk 는 pv(초기 균등 벡터)에 대해 수행되어 실제로는
        #          "앞에서 60%" 인덱스를 뽑는 형태가 된다. 원본 구현의 한계이며
        #          본 문서에서는 코드를 수정하지 않으므로 동작은 그대로 둔다.
    else:
        N = g.number_of_nodes()                                             # [한국어] 동종 그래프 노드 수.
        pv = compute_pagerank(g, args.damp, args.K, N)                      # [한국어] 동종 PR 실행.

    print("N: ", N)                                                         # [한국어] 전체 노드 수 출력.
    topk = int(N * 0.6)                                                     # [한국어] 상위 60% 를 pin 대상으로 선정.
    _, indices = torch.topk(pv, k=topk, largest=True)                       # [한국어] pv 기준 상위 k 인덱스 추출. indices 가 핫 노드 집합.

    torch.save(indices, args.out_path)                                      # [한국어] 텐서 직렬화 저장. 학습 시 --pin_file 로 전달되어
                                                                            #          GIDS_CPU_buffer / range_t::set_cpu_buffer 에서 소비된다.

