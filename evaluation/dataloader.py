"""
[한국어] BaM baseline 및 GIDS 학습에 공통으로 쓰이는 DGL Dataset 로더
        (dataloader.py)

=== 파일의 역할 ===
IGB260M / IGBH / OGB-MAG 벤치마크의 그래프 구조(edge_index)와 feature/label
numpy 파일들을 DGL graph 객체로 변환해주는 **데이터셋 래퍼 모음**이다. 이
파일은 GIDS 고유 SSD-direct 경로와는 무관하게, **BaM 논문의 baseline 형태**로
feature 를 host-side(mmap 또는 in-memory) 로 들고 DGL graph.ndata['feat'] 에
직접 붙이는 전통 방식이다. GIDS 학습 스크립트는 이 클래스들을 그래프 구조
용도로만 사용하고, feature 는 런타임에 GIDS_DGLDataLoader 가 SSD 에서 GPU 로
가져오도록 우회한다. 즉 이 파일은 graph structure + mask 설정 + baseline
feature 소스를 담당하는 공용 모듈이다.

=== 전체 아키텍처에서의 위치 ===
evaluation/{heterogeneous,homogenous,...}_train*.py → 본 파일의 DGLDataset
구현체(IGB260MDGLDataset / OGBDGLDataset / IGBHeteroDGLDataset /
IGBHeteroDGLDatasetMassive / IGBHeteroDGLDatasetTest / OGBHeteroDGLDatasetMassive)
로부터 `dataset[0]` 로 DGL heterograph/graph 를 얻는다. 학습 스크립트가 DGL
sampler (NeighborSampler / LadiesSampler) 에 그래프를 전달하면 block 이
만들어지고, feature fetch 단계에서만 GIDS 경유 SSD → GPU 경로가 개입한다.

GIDS_Setup/GIDS/GIDS.py 의 GIDS_DGLDataLoader 와 비교 :
 - 본 파일(baseline): paper_node_features = numpy mmap 또는 로드 → DGL ndata
                       에 저장 → 학습 시 DGL 이 CPU→GPU copy.
 - GIDS.py(가속): DGL ndata 에는 더미/절반 placeholder, 실제 feature 는 매
                  iteration GIDS_Loader 가 input_nodes ID 기반으로 BAM_Feature_
                  Store → bam page_cache → NVMe SSD 로부터 직접 GPU 텐서로 fetch.
                  Constant CPU Buffer(페이지랭크 핫 노드 pinned) + Window
                  Buffering(다음 n 배치 프리패치) 도 GIDS 측에만 존재.
 동일한 IGBH dataset 상수(269346174 paper 등)를 공유해, GIDS 설정의 offset
 계산(heterograph_map)과 write_data*.sh 의 --loffset 값이 서로 일관되게 된다.

=== 타 모듈과의 연결 ===
- DGL: DGLDataset 상속, dgl.graph / dgl.heterograph / remove/add_self_loop / formats('csc').
- numpy / np.memmap: feature/label/edge_index .npy 를 host 메모리로 매핑.
- pandas: OGB 의 csv.gz split 파일 로딩.
- 학습 스크립트: self.graph 에 'feat','label','train_mask','val_mask','test_mask'
  등을 동일 인터페이스로 기대. DGL 기본 규약.
- GIDS.py / gids_module: 본 파일이 만든 graph 는 feature 자체를 SSD 경로로
  대체할 때 heterograph_map offsets 가 정확히 맞아야 한다.
  예) paper=0 → author=269346174 → fos=546567057 → institute=547280017.

=== 주요 함수/구조체 요약 ===
- _idx_to_mask(indices, total_samples)    : 인덱스 리스트 → bool mask.
- ogb_get_idx_split_mask(path, n)         : OGB csv.gz split 을 train/valid/test mask 로.
- class IGB260M(object)                    : raw numpy 파일(node_feat/label/edge)을 size 별로 접근.
- class IGB260MDGLDataset(DGLDataset)      : homogeneous IGB (paper→paper cites 단일 관계) + 마스크.
- class OGBDGLDataset(DGLDataset)          : homogeneous OGB (ogbn-papers 등).
- class IGBHeteroDGLDataset(DGLDataset)    : small/medium IGBH (paper/author/fos/institute).
- class IGBHeteroDGLDatasetMassive(DGLDataset): large/full IGBH (journal/conference 포함).
- class IGBHeteroDGLDatasetTest(DGLDataset): IGBH 디버그용 변형 (author/paper memmap 경로 고정).
- class OGBHeteroDGLDatasetMassive(DGLDataset): OGB-MAG Heterogeneous (paper/author/institute).
- main                                     : 단독 실행 시 IGBHeteroDGLDatasetMassive 동작 확인.

핵심 공통 필드(self.graph.*):
- ndata['feat']       : paper(혹은 node) feature 텐서 (memmap 기반 가능).
- ndata['label']      : 노드 라벨 (paper 만 분류 대상).
- ndata['train_mask'],['val_mask'],['test_mask']: 60/20/20 랜덤 분할 bool 마스크.
- predict              : 이종 그래프에서 분류 대상 ntype('paper').
- num_{ntype}_nodes    : 각 ntype 노드 수 (GIDS offset 계산 참고용).
"""

import argparse, time                           # [한국어] CLI / 시간 측정.
import numpy as np                              # [한국어] .npy / memmap 로딩.
import torch                                    # [한국어] torch.from_numpy, mask 텐서 생성.
import os.path as osp                           # [한국어] osp.join 으로 OS-독립 경로.
import pandas as pd                             # [한국어] OGB split csv.gz 로더.

import dgl                                       # [한국어] DGL graph/heterograph API.
from dgl.data import DGLDataset                 # [한국어] 커스텀 Dataset 베이스. process() 오버라이드 패턴.
import warnings                                  # [한국어] 경고 억제.
warnings.filterwarnings("ignore")                # [한국어] numpy memmap/DGL deprecation 알림 억제.


# [한국어]
# _idx_to_mask - 인덱스 배열을 길이 total_samples 의 boolean mask 로 변환.
# @param indices      : 1D LongTensor 또는 numpy int 배열.
# @param total_samples: 전체 노드 수.
# @return             : shape (total_samples,) 의 bool 텐서.
def _idx_to_mask(indices, total_samples):
        mask = torch.zeros(total_samples, dtype=torch.bool)    # [한국어] 기본 False.
        mask[indices] = True                                    # [한국어] 지정 인덱스만 True.
        return mask                                             # [한국어] DGL ndata 용 mask.

# [한국어]
# ogb_get_idx_split_mask - OGB 의 train/valid/test split csv.gz 를 읽어 mask 로 변환.
# @param path: split 디렉토리 경로.
# @param n   : 전체 노드 수.
# @return    : (train_mask, valid_mask, test_mask) 튜플.
def ogb_get_idx_split_mask(path, n):
    train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
    # [한국어] 첫 column 만 읽어 인덱스 배열화 (.values.T[0]).
    valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
    test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

    train_mask = _idx_to_mask(train_idx, n)                     # [한국어] 인덱스 → bool mask.
    valid_mask = _idx_to_mask(valid_idx, n)
    test_mask = _idx_to_mask(test_idx, n)

    return train_mask, valid_mask, test_mask                    # [한국어] DGL 관행: ndata[...] 에 바로 저장 가능.


class IGB260M(object):
    # [한국어] IGB260M 원본 numpy 파일들에 대한 얇은 접근 객체.
    #   - 데이터셋 size('experimental'/'small'/'medium'/'large'/'full')에 따라
    #     num_nodes / 파일 경로가 달라진다.
    #   - paper_feat / paper_label / paper_edge 프로퍼티가 실제 데이터를 반환.
    #   - in_memory=1 이면 np.load 로 full load, 0 이면 mmap_mode='r' 로 memmap.
    #   - data='OGB' 이면 IGB 경로 대신 OGB 파일명으로 분기.
    # 필드 설명:
    #   dir         : 루트 경로(arg.path).
    #   size        : 'experimental'/'small'/.../'full'.
    #   synthetic   : 1 이면 random feature 생성(테스트용).
    #   in_memory   : 0=memmap, 1=전체 로드.
    #   num_classes : 19 또는 2983 (IGB paper label 두 종류).
    #   emb_size    : feature 차원 (전형 1024).
    #   uva_graph   : graph 구조를 UVA 로 GPU 에 노출할지 — True 면 in_memory 유사 동작 요구.
    #   data        : 'IGB' 또는 'OGB'.
    # [한국어] 생성자. 위 필드 설명 참조.
    def __init__(self, root: str, size: str, in_memory: int, uva_graph: int,  \
            classes: int, synthetic: int, emb_size: int, data: str):
        self.dir = root                                         # [한국어] IGB/OGB 원본 디렉토리.
        self.size = size                                        # [한국어] 데이터셋 크기 키워드.
        self.synthetic = synthetic                              # [한국어] 1 이면 random feature.
        self.in_memory = in_memory                              # [한국어] 전체 로드 vs memmap.
        self.num_classes = classes                              # [한국어] label 타입 19/2983.
        self.emb_size = emb_size                                # [한국어] feature dim.
        self.uva_graph = uva_graph                              # [한국어] UVA graph 플래그.
        self.data = data                                        # [한국어] 'IGB' 또는 'OGB'.

    # [한국어] num_nodes - size/data 조합에 따른 paper 노드 수 반환.
    def num_nodes(self):
        if self.data == 'OGB':
            return 111059956                                    # [한국어] OGB-papers 약 111M.

        if self.size == 'experimental':
            return 100000                                        # [한국어] 10 만 노드 테스트용.
        elif self.size == 'small':
            return 1000000
        elif self.size == 'medium':
            return 10000000
        elif self.size == 'large':
            return 100000000
        elif self.size == 'full':
            return 269346174                                     # [한국어] IGB full paper 269.3M. GIDS heterograph_map offset 기준값.

    # [한국어] paper_feat - paper 노드 feature 배열 (numpy).
    #   경로는 size/data 에 따라 분기. in_memory=0 이면 mmap_mode='r' 로 가상
    #   메모리 매핑만 수행해 실제 페이지 fault 는 접근 시점에 일어난다.
    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()                                       # [한국어] size 기준 노드 수.
        # TODO: temp for bafs. large and full special case
        if self.data == 'OGB':                                              # [한국어] OGB 경로.
            path = osp.join(self.dir, 'node_feat.npy')
            if self.in_memory:
                emb = np.load(path)                                         # [한국어] 전체 로드.
            else:
                emb = np.load(path, mmap_mode='r')                          # [한국어] read-only memmap.

        elif self.size == 'large' or self.size == 'full':                   # [한국어] large/full: 특정 SSD 경로 하드코딩.
            path = '/mnt/nvme17/node_feat.npy'
            #path = '/mnt/raid0_2/node_feat.npy'
            if self.in_memory:
                emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024)).copy()
                # [한국어] memmap 후 .copy() → RAM 으로 완전 로드. 대용량에는 비권장.
            else:
                emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024))
                # [한국어] shape 명시 memmap. 파일 크기 = num_nodes*1024*4 byte.
        else:
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype('f')          # [한국어] 디버그용 랜덤 feature.
            else:
                if self.in_memory:
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode='r')

        return emb                                                          # [한국어] numpy 배열 (memmap 또는 실제 ndarray).

    # [한국어] paper_label - paper 노드 라벨 배열.
    #   num_classes(19 or 2983) 에 따라 파일이 다르다.
    @property
    def paper_label(self) -> np.ndarray:
        if(self.data == 'OGB'):
            path = osp.join(self.dir, 'node_label.npy')
            node_labels = np.load(path).flatten()                           # [한국어] (N,1) → (N,) flatten.
            return node_labels

        elif self.size == 'large' or self.size == 'full':
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
                path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_19_extended.npy'
                if(self.in_memory):
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes)).copy()
                else:
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 227130858
            else:
                path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_2K_extended.npy'
                
                if(self.in_memory):
                    node_labels = np.load(path)
                else:
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 157675969

        else:
            if self.num_classes == 19:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
            else:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
            if self.in_memory:
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode='r')
        return node_labels

    # [한국어] paper_edge - paper→paper cites edge_index (E,2).
    #   size 에 따라 파일 경로가 다르며, uva_graph=1 일 때는 DGL UVA 에 올릴 수
    #   있도록 memmap 대신 전체 로드가 필요(첫 번째 분기).
    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        if self.data == 'OGB':
            path = osp.join(self.dir, 'edge_index.npy')                     # [한국어] OGB: 단일 파일.
        elif self.size == 'full':
            path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy'
        elif self.size == 'large':
            path = '/mnt/nvme7/large/processed/paper__cites__paper/edge_index.npy'

        if self.in_memory or self.uva_graph:
            return np.load(path)                                            # [한국어] UVA 는 실 메모리 보유 요구.
        else:
            return np.load(path, mmap_mode='r')                             # [한국어] 디스크 기반 memmap.


class IGB260MDGLDataset(DGLDataset):
    # [한국어] 동종 IGB260M 용 DGLDataset.
    #   process() 내부에서 IGB260M 로 feature/edge/label 을 받아 DGL graph 를
    #   구성하고 ndata['feat'/'label'/'train_mask'/'val_mask'/'test_mask'] 설정.
    #   full 사이즈는 CSC 포맷으로 edge 를 사전 저장해 빠르게 로드한다.
    # [한국어] 생성자: args 보관 후 부모 DGLDataset.__init__ 호출 → process() 자동 호출.
    def __init__(self, args):
        self.dir = args.path                                                 # [한국어] IGB 루트.
        self.args = args                                                     # [한국어] 전체 args 보존.
        super().__init__(name='IGB260MDGLDataset')                           # [한국어] DGL 이 process() 를 호출해준다.

    # [한국어] process - DGL 이 __init__ 시 자동 호출. 그래프와 feature 세팅을 수행.
    def process(self):
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, uva_graph=self.args.uva_graph, \
            classes=self.args.num_classes, synthetic=self.args.synthetic, emb_size=self.args.emb_size, data=self.args.data)
        # [한국어] 원본 numpy 어댑터 생성.

        node_features = torch.from_numpy(dataset.paper_feat)                 # [한국어] numpy memmap/array → torch tensor (view).
        node_edges = torch.from_numpy(dataset.paper_edge)                    # [한국어] edge_index. shape (E,2).
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)   # [한국어] label 은 long 으로 캐스팅.

        print("node edge:", node_edges)                                       # [한국어] 디버그 출력.
        cur_path = osp.join(self.dir, self.args.dataset_size, 'processed')   # [한국어] CSC 보조 파일 루트.
        # cur_path = '/mnt/nvme16/IGB260M_part_2/full/processed'
        
#        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
        if self.args.dataset_size == 'full':                                 # [한국어] full 은 사전 변환된 CSC 로드.
            edge_row_idx = torch.from_numpy(np.load(cur_path + '/paper__cites__paper/edge_index_csc_row_idx.npy'))
            edge_col_idx = torch.from_numpy(np.load(cur_path + '/paper__cites__paper/edge_index_csc_col_idx.npy'))
            edge_idx = torch.from_numpy(np.load(cur_path + '/paper__cites__paper/edge_index_csc_edge_idx.npy'))
            # [한국어] CSC = Compressed Sparse Column. (colptr, rowidx, edge_id).
            self.graph = dgl.graph(('csc', (edge_col_idx,edge_row_idx,edge_idx)), num_nodes=node_features.shape[0])
            # [한국어] DGL graph CSC 생성.
            self.graph  = self.graph.formats('csc')                          # [한국어] 내부 포맷을 CSC 로 고정.
        else:
            self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
            # [한국어] (src,dst) 쌍으로 생성 (COO 형태).
        print("self graph: ", self.graph.formats())
        self.graph.ndata['feat'] = node_features                             # [한국어] feature 부착.
        self.graph.ndata['label'] = node_labels                              # [한국어] label 부착.
        print("self graph2: ", self.graph.formats())
        if self.args.dataset_size != 'full':                                 # [한국어] full 은 CSC 고정이라 self-loop 조작 생략.
            self.graph = dgl.remove_self_loop(self.graph)                    # [한국어] 중복 self-loop 제거.
            self.graph = dgl.add_self_loop(self.graph)                       # [한국어] GCN 계열 필수 self-loop 추가.
        print("self graph3: ", self.graph.formats())
        
        # [한국어] 아래는 train/val/test 마스크 랜덤 분할. full 은 라벨 있는 노드 수
        #          이내에서만 분할해 test_mask 범위를 제한한다.
        if self.args.dataset_size == 'full':
            #TODO: Put this is a meta.pt file
            if self.args.num_classes == 19:
                n_labeled_idx = 227130858                                    # [한국어] 19-class 라벨이 달린 paper 수.
            else:
                n_labeled_idx = 157675969                                    # [한국어] 2983-class 라벨 paper 수.

            n_nodes = node_features.shape[0]
            n_train = int(n_labeled_idx * 0.6)                               # [한국어] 60% train.
            n_val   = int(n_labeled_idx * 0.2)                               # [한국어] 20% val. 나머지 20% test.
            print("self graph4: ", self.graph.formats())
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            perm = torch.randperm(n_nodes)                                   # [한국어] 노드 무작위 순열. seed 는 호출자에서 설정.
            train_mask[perm[:n_train]] = True
            val_mask[perm[n_train:n_train + n_val]] = True
            test_mask[perm[n_train + n_val:n_labeled_idx]] = True            # [한국어] 라벨 범위에서만 test.
            print("self graph5: ", self.graph.formats())
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        else:
            # [한국어] 소형 데이터셋: 전체 노드 60/20/20 분할.
            n_nodes = node_features.shape[0]
            n_train = int(n_nodes * 0.6)
            n_val   = int(n_nodes * 0.2)
            perm = torch.randperm(n_nodes)

            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            train_mask[perm[:n_train]] = True
            val_mask[perm[n_train:n_train + n_val]] = True
            test_mask[perm[n_train + n_val:]] = True

            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask

    # [한국어] DGL Dataset 인덱스 프로토콜. 유일한 그래프 하나를 반환.
    def __getitem__(self, i):
        return self.graph

    # [한국어] 주의: len 은 self.graphs(복수형) 을 참조하나 이 클래스는 graph 단수만 보유하므로
    #          실제 호출 시 AttributeError. 학습 스크립트는 dataset[0] 만 쓰기에 무해.
    def __len__(self):
        return len(self.graphs)

class OGBDGLDataset(DGLDataset):
    # [한국어] OGB homogeneous 버전. IGB260M 어댑터를 재사용하되 OGB 경로로 분기된 feature/edge 를 사용.
    # [한국어] 생성자 — IGB260MDGLDataset 과 동일 패턴.
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    # [한국어] process - OGB homogeneous graph 구성. OGB edge_index 는 (2,E) 형태라 [0,:]/[1,:] 사용.
    def process(self):
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, uva_graph=self.args.uva_graph, \
            classes=self.args.num_classes, synthetic=self.args.synthetic, emb_size=self.args.emb_size, data=self.args.data)

        node_features = torch.from_numpy(dataset.paper_feat)                  # [한국어] OGB 경로 feature.
        node_edges = torch.from_numpy(dataset.paper_edge)                     # [한국어] OGB edge_index shape (2,E).
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        self.graph = dgl.graph((node_edges[0,:],node_edges[1,:]), num_nodes=node_features.shape[0])
        # [한국어] OGB 포맷: 첫 row=src, 둘째 row=dst.

        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        self.graph = dgl.remove_self_loop(self.graph)                        # [한국어] 중복 self-loop 제거 후 추가 — GCN 관행.
        self.graph = dgl.add_self_loop(self.graph)


        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val   = int(n_nodes * 0.2)
        # [한국어] 위 n_train/n_val 은 실제 mask 생성에는 사용되지 않으며 아래 split 파일이 우선.

        split_path = osp.join(self.dir,'../split', 'time')                   # [한국어] OGB time-based split.
        print("split path: ", split_path)
        train_mask, val_mask, test_mask = ogb_get_idx_split_mask(split_path, n_nodes)
        # [한국어] OGB 공식 split 에서 mask 를 가져온다.

        #train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
        #train_mask[:n_train] = True
        #val_mask[n_train:n_train + n_val] = True
        #test_mask[n_train + n_val:] = True
            
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    # [한국어] DGL Dataset 프로토콜.
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)                                              # [한국어] 동일 원인 — 단일 그래프라 실사용 시 오류 가능.


class IGBHeteroDGLDataset(DGLDataset):
    # [한국어] IGBH (small/medium) 이종 그래프 로더.
    # node types: paper / author / fos / institute.
    # edge types: cites, written_by, affiliated_to, topic + 각 역방향 relation.
    # process() 에서 edge_index → heterograph 생성, 각 ntype feature/label 부착,
    # paper 노드에 train/val/test mask 60/20/20 랜덤 분할 생성.
    # [한국어] 생성자. DGL 부모가 process() 자동 호출.
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')
    # [한국어] process - 4개 edge type 의 edge_index .npy 를 로드해 heterograph 구성.
    #          각 edge_index 는 (E,2) 형태. in_memory=1 이면 전체 로드, 0 이면 memmap.
    def process(self):

        if self.args.in_memory:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy')))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy')))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy')))

        else:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))

        # graph_data = {
        #     ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
        #     ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
        #     ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
        #     ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1])
        # }

        # [한국어] edge type 별 src/dst 쌍 dict. DGL heterograph 생성자 입력.
        #          forward relation 과 역방향 relation 쌍을 함께 선언해
        #          양방향 message passing 을 가능하게 한다.
        graph_data = {
            ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
            ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
            ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1]),
            ('paper', 'citied_by', 'paper'): (paper_paper_edges[:, 1], paper_paper_edges[:, 0]),
            ('author', 'rev_written_by', 'paper'): (author_paper_edges[:, 1], author_paper_edges[:, 0]),
            ('institute', 'rev_affiliated_to', 'author'): (affiliation_author_edges[:, 1], affiliation_author_edges[:, 0]),
            ('fos', 'rev_topic', 'paper'): (paper_fos_edges[:, 1], paper_fos_edges[:, 0])
        }

        self.graph = dgl.heterograph(graph_data)                              # [한국어] heterograph 생성.
        self.graph.predict = 'paper'                                           # [한국어] 분류 대상 ntype.

        # [한국어] paper feature/label 로드 (num_classes=19).
        if self.args.in_memory:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper', 'node_feat.npy')))
            paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper', 'node_label_19.npy'))).to(torch.long)
        else:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper', 'node_feat.npy'), mmap_mode='r'))
            paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper', 'node_label_19.npy'), mmap_mode='r')).to(torch.long)

        self.graph.nodes['paper'].data['feat'] = paper_node_features          # [한국어] paper feature 부착.
        self.graph.num_paper_nodes = paper_node_features.shape[0]             # [한국어] 편의 필드 — GIDS offset 매핑 참고.
        self.graph.nodes['paper'].data['label'] = paper_node_labels
        # [한국어] 이하 author/institute/fos 각각 node_feat 로드 후 ndata['feat'] 부착.
        #          in_memory 분기 패턴은 paper 와 동일.
        if self.args.in_memory:
            author_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'author', 'node_feat.npy')))
        else:
            author_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author', 'node_feat.npy'), mmap_mode='r'))
        self.graph.nodes['author'].data['feat'] = author_node_features
        self.graph.num_author_nodes = author_node_features.shape[0]

        if self.args.in_memory:
            institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'institute', 'node_feat.npy')))       
        else:
            institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'institute', 'node_feat.npy'), mmap_mode='r'))
        self.graph.nodes['institute'].data['feat'] = institute_node_features
        self.graph.num_institute_nodes = institute_node_features.shape[0]

        if self.args.in_memory:
            fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'fos', 'node_feat.npy')))       
        else:
            fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'fos', 'node_feat.npy'), mmap_mode='r'))
        self.graph.nodes['fos'].data['feat'] = fos_node_features
        self.graph.num_fos_nodes = fos_node_features.shape[0]
        
        # [한국어] 'cites' edge type 에 self-loop 조작 — GCN 안정성 위해.
        self.graph = dgl.remove_self_loop(self.graph, etype='cites')
        self.graph = dgl.add_self_loop(self.graph, etype='cites')

        n_nodes = paper_node_features.shape[0]                                # [한국어] paper 노드 수 기준으로 mask 분할.

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        perm = torch.randperm(n_nodes)                                        # [한국어] paper 노드 랜덤 순열.

        train_mask[perm[:n_train]] = True                                     # [한국어] 앞 60% = train.
        val_mask[perm[n_train:n_train + n_val]] = True                        # [한국어] 다음 20% = val.
        test_mask[perm[n_train + n_val:]] = True                              # [한국어] 나머지 = test.

        self.graph.nodes['paper'].data['train_mask'] = train_mask             # [한국어] paper 노드에만 mask 부착 (분류 대상).
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask


    # [한국어] DGL Dataset 프로토콜. 단일 heterograph 반환.
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
class IGBHeteroDGLDatasetMassive(DGLDataset):
    # [한국어] IGBH large/full 을 위한 로더. 기본 4개 ntype 에 journal/conference 를
    #          추가해 총 6개 ntype, 12개(=6 정방향+6 역방향) edge type 구성.
    #          feature 는 memmap 기반 대용량 접근 전제(269M paper 등).
    #          DGLDataset.__init__ 을 직접 호출하지 않고 self.process() 를 생성자
    #          안에서 바로 수행하도록 구현된 점이 특이하다.
    # [한국어] 생성자 — 부모 DGLDataset.__init__ 을 호출하지 않고 본문에서 직접 load.
    #          이유: massive 는 초기화 경로에 시간이 오래 걸려 DGLDataset 의 캐싱
    #          로직을 건너뛰려는 의도.
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        self.graph = None
        #super().__init__(name='IGB260M')

    # def process(self):
        # [한국어] 아래 블록이 실질 process(). uva_graph 플래그에 따라 전체 로드/memmap 분기.
        print("processs IGBH Dataset")
        if self.args.uva_graph:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__cites__paper', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy')))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy')))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy')))
            paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__published__journal', 'edge_index.npy'))) 
            paper__venue__conference = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__venue__conference', 'edge_index.npy'))) 

        else:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))
            paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__published__journal', 'edge_index.npy'), mmap_mode='r'))
            paper__venue__conference = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__venue__conference', 'edge_index.npy'), mmap_mode='r'))

        print("loading feature data")
        # [한국어] 데이터셋 크기별 paper/author 노드 수 상수. GIDS heterograph_map 및
        #          write_data_full.sh 의 --loffset 계산과 동일 값이어야 한다.
        if self.args.dataset_size == "full":
            num_paper_nodes = 269346174                                       # [한국어] IGB full paper 수.
            paper_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
            'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,1024)))
            if self.args.num_classes == 19:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
                'paper', 'node_label_19.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            elif self.args.num_classes == 2983:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
                'paper', 'node_label_2K.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            num_author_nodes = 277220883
            author_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "full", 'processed', 
            'author', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_author_nodes,1024)))

          
        elif self.args.dataset_size == "large":
            num_paper_nodes = 100000000
            paper_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "large", 'processed', 
            'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,1024)))
            if self.args.num_classes == 19:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "large", 'processed', 
                'paper', 'node_label_19.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            elif self.args.num_classes == 2983:
                paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir, "large", 'processed', 
                'paper', 'node_label_2K.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes))).to(torch.long)
            num_author_nodes = 116959896
            author_node_features = torch.from_numpy(np.memmap(osp.join(self.dir, "large", 'processed', 
            'author', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_author_nodes,1024)))

        institute_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'institute', 'node_feat.npy'), mmap_mode='r'))
        fos_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'fos', 'node_feat.npy'), mmap_mode='r'))

        journal_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'journal', 'node_feat.npy'), mmap_mode='r'))
        conference_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
        'conference', 'node_feat.npy'), mmap_mode='r'))
               
        # [한국어] heterograph 각 ntype 의 노드 수를 dict 로 요약. DGL heterograph
        #          생성자에 전달되어 node id 범위를 확정한다.
        num_nodes_dict = {'paper': num_paper_nodes, 'author': num_author_nodes, 'institute': len(institute_node_features), 'fos': len(fos_node_features), 'journal': len(journal_node_features), 'conference': len(conference_node_features)  }
        print(f"Setting the graph structure {num_nodes_dict}")
        # graph_data = {
        #     ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
        #     ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
        #     ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
        #     ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1])
        # }

        # [한국어] massive 버전 edge dict — 기본 relation 4종 + journal/conference 2종 +
        #          각각의 역방향 relation. 총 12 relation 이 선언된다.
        graph_data = {
            ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
            ('paper', 'rev_cites', 'paper'): (paper_paper_edges[:, 1], paper_paper_edges[:, 0]),
            ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
            ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1]),
            ('author', 'rev_written_by', 'paper'): (author_paper_edges[:, 1], author_paper_edges[:, 0]),
            ('institute', 'rev_affiliated_to', 'author'): (affiliation_author_edges[:, 1], affiliation_author_edges[:, 0]),
            ('fos', 'rev_topic', 'paper'): (paper_fos_edges[:, 1], paper_fos_edges[:, 0]),
            ('paper', 'published', 'journal'): (paper_published_journal[:, 0], paper_published_journal[:, 1]),
            ('paper', 'venue', 'conference'): (paper__venue__conference[:, 0], paper__venue__conference[:, 1]),
            ('journal', 'rev_published', 'paper'): (paper_published_journal[:, 1], paper_published_journal[:, 0]),
            ('conference', 'rev_venue', 'paper'): (paper__venue__conference[:, 1], paper__venue__conference[:, 0])
        }

        # graph_data = {
        #     ('paper', 'cites', 'paper'): (torch.cat([paper_paper_edges[1, :], paper_paper_edges[0, :]]), torch.cat([paper_paper_edges[0, :], paper_paper_edges[1, :]])),
        #     ('paper', 'written_by', 'author'): author_paper_edges,
        #     ('author', 'affiliated_to', 'institute'): affiliation_author_edges,
        #     ('paper', 'topic', 'fos'): paper_fos_edges,
        #     ('author', 'rev_written_by', 'paper'): (author_paper_edges[1, :], author_paper_edges[0, :]),
        #     ('institute', 'rev_affiliated_to', 'author'): (affiliation_author_edges[1, :], affiliation_author_edges[0, :]),
        #     ('fos', 'rev_topic', 'paper'): (paper_fos_edges[1, :], paper_fos_edges[0, :])
        # }

        #graph_data = torch.load("/mnt/nvme22/IGBH_csc.pth")
        print("dgl.heterograph init starting")
        self.graph = dgl.heterograph(graph_data, num_nodes_dict)             # [한국어] heterograph 생성.
        self.graph = dgl.remove_self_loop(self.graph, etype='cites')         # [한국어] GCN 안정성.
        self.graph = dgl.add_self_loop(self.graph, etype='cites')
        self.graph = self.graph.formats('csc')                               # [한국어] 이웃 샘플링에 유리한 CSC 포맷.
        print(self.graph.formats())
        print("dgl.heterograph init successful")
        self.graph.predict = 'paper'                                          # [한국어] 분류 대상 ntype.


        # [한국어] 이하 각 ntype 에 feature / label / 노드 수 메타데이터 부착.
        #          num_{ntype}_nodes 는 학습 스크립트가 GIDS offset/logging 에 사용.
        self.graph.nodes['paper'].data['feat'] = paper_node_features
        # self.graph.num_paper_nodes = paper_node_features.shape[0]

        self.graph.num_paper_nodes = num_paper_nodes

        self.graph.nodes['paper'].data['label'] = paper_node_labels
        self.graph.nodes['author'].data['feat'] = author_node_features
        # self.graph.num_author_nodes = author_node_features.shape[0]
        self.graph.num_author_nodes = num_author_nodes

        self.graph.nodes['institute'].data['feat'] = institute_node_features
        self.graph.num_institute_nodes = institute_node_features.shape[0]

        self.graph.nodes['fos'].data['feat'] = fos_node_features
        self.graph.num_fos_nodes = fos_node_features.shape[0]

        #ADDED
        # [한국어] massive 전용 추가 ntype (journal/conference).

        self.graph.nodes['journal'].data['feat'] = journal_node_features
        self.graph.num_journal_nodes = journal_node_features.shape[0]

        self.graph.nodes['conference'].data['feat'] = conference_node_features
        self.graph.num_conference_nodes = conference_node_features.shape[0]

        n_nodes = num_paper_nodes                                             # [한국어] paper 노드 기준으로 mask 분할.

        # n_nodes = paper_node_features.shape[0]

        n_train = int(n_nodes * 0.6)                                        # [한국어] 60% = train. paper 노드만 분류 대상이므로 paper 수 기준.
        n_val = int(n_nodes * 0.2)                                          # [한국어] 20% = val. 나머지 20% = test.


        train_mask = torch.zeros(n_nodes, dtype=torch.bool)                 # [한국어] 초기 False. 이후 perm 으로 선택된 인덱스만 True.
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        perm = torch.randperm(n_nodes)                                       # [한국어] 무작위 순열. torch 기본 시드 사용 — 학습 스크립트가 set_seed 로 고정해야 재현 가능.

        train_mask[perm[:n_train]] = True                                    # [한국어] 앞 60% 슬라이스 → train.
        val_mask[perm[n_train:n_train + n_val]] = True                       # [한국어] 다음 20% → val.
        test_mask[perm[n_train + n_val:]] = True                             # [한국어] 나머지 → test.

        self.graph.nodes['paper'].data['train_mask'] = train_mask            # [한국어] DGL 에서 paper 노드에만 mask 부착 — 분류 head 가 paper 만 예측.
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask

    # [한국어] DGL Dataset 프로토콜 — 학습 스크립트 dataset[0] 호출 시 heterograph 반환. i 는 무시(단일 그래프).
    def __getitem__(self, i):
        return self.graph

    # [한국어] DGL Dataset 표준 길이. IGBH massive 는 단일 graph 만 보유하므로 1 고정.
    def __len__(self):
        return 1


class IGBHeteroDGLDatasetTest(DGLDataset):
    # [한국어] IGBHeteroDGLDataset 의 디버그/실험 변형.
    #   - 기본 4 edge type 로드 후, paper/author feature 는 강제로 /mnt/raid0/full
    #     memmap 을 덮어씀 (num_paper_nodes=269346174, num_author_nodes=277220883 하드코딩).
    #   - small/medium edge 구조 + full 크기 feature 를 섞어 테스트하는 목적.
    #   - 동작: 주요 상수만 차이, 나머지 process 흐름은 IGBHeteroDGLDataset 와 동일.
    # [한국어] 생성자.
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')
    # [한국어] process - 상기 테스트 변형. paper/author feature 를 하드코딩 경로 memmap 으로 덮어쓴다.
    def process(self):

        if self.args.in_memory:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy')))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy')))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy')))

        else:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed',
            'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))

        # graph_data = {
        #     ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
        #     ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
        #     ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
        #     ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1])
        # }

        graph_data = {
            ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
            ('paper', 'written_by', 'author'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
            ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1]),
            ('paper', 'topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1]),
            ('paper', 'citied_by', 'paper'): (paper_paper_edges[:, 1], paper_paper_edges[:, 0]),
            ('author', 'rev_written_by', 'paper'): (author_paper_edges[:, 1], author_paper_edges[:, 0]),
            ('institute', 'rev_affiliated_to', 'author'): (affiliation_author_edges[:, 1], affiliation_author_edges[:, 0]),
            ('fos', 'rev_topic', 'paper'): (paper_fos_edges[:, 1], paper_fos_edges[:, 0])
        }

        self.graph = dgl.heterograph(graph_data)     
        self.graph.predict = 'paper'

        if self.args.in_memory:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_feat.npy')))
            paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_label_19.npy'))).to(torch.long)
        else:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_feat.npy'), mmap_mode='r'))
            paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, self.args.dataset_size, 'processed', 
            'paper', 'node_label_19.npy'), mmap_mode='r')).to(torch.long)  


        # [한국어] 아래는 테스트용 하드코딩 — small/medium edge 에 full 크기 memmap 을 강제로 덮어씀.
        #          실제 인덱스 범위와 불일치할 수 있으므로 production 학습용 아님.
        num_paper_nodes = 269346174
        paper_node_features = torch.from_numpy(np.memmap(osp.join('/mnt/raid0/full', 'processed',
        'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,1024)))
        num_author_nodes = 277220883
        author_node_features = torch.from_numpy(np.memmap(osp.join('/mnt/raid0/full', 'processed', 
        'author', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_author_nodes,1024)))

        self.graph.nodes['paper'].data['feat'] = paper_node_features
        self.graph.num_paper_nodes = num_paper_nodes
        self.graph.nodes['paper'].data['label'] = paper_node_labels
        
        self.graph.nodes['author'].data['feat'] = author_node_features
        self.graph.num_author_nodes = num_author_nodes

        institute_node_features = torch.from_numpy(np.load(osp.join('/mnt/raid0/full', 'processed', 
        'institute', 'node_feat.npy'), mmap_mode='r'))
        fos_node_features = torch.from_numpy(np.load(osp.join('/mnt/raid0/full', 'processed', 
        'fos', 'node_feat.npy'), mmap_mode='r'))
        num_nodes_dict = {'paper': num_paper_nodes, 'author': num_author_nodes, 'institute': len(institute_node_features), 'fos': len(fos_node_features)}
       
        self.graph.nodes['fos'].data['feat'] = fos_node_features
        self.graph.num_fos_nodes = fos_node_features.shape[0]
        self.graph.nodes['institute'].data['feat'] = institute_node_features  # [한국어] institute feature 부착.
        self.graph.num_institute_nodes = institute_node_features.shape[0]      # [한국어] GIDS offset 참고용.
        self.graph = dgl.remove_self_loop(self.graph, etype='cites')          # [한국어] GCN 안정성 — paper→paper cites 만 대상.
        self.graph = dgl.add_self_loop(self.graph, etype='cites')             # [한국어] add_self_loop 로 i→i 엣지 보강.

        n_nodes = paper_node_features.shape[0]                                 # [한국어] paper 노드 기준으로 mask 분할 (분류 대상).

        n_train = int(n_nodes * 0.6)                                           # [한국어] 60% train.
        n_val = int(n_nodes * 0.2)                                             # [한국어] 20% val / 20% test.

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)                    # [한국어] False 초기화.
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        perm = torch.randperm(n_nodes)                                          # [한국어] 무작위 순열 — IGBHeteroDGLDatasetTest 전용 분할.
        train_mask[perm[:n_train]] = True                                       # [한국어] 앞 60% 선택.
        val_mask[perm[n_train:n_train + n_val]] = True                          # [한국어] 다음 20%.
        test_mask[perm[n_train + n_val:]] = True                                # [한국어] 나머지 20%.

        self.graph.nodes['paper'].data['train_mask'] = train_mask               # [한국어] paper 노드에만 부착.
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask


    # [한국어] DGL Dataset 프로토콜 — heterograph 1개 반환.
    def __getitem__(self, i):
        return self.graph

    # [한국어] 단일 그래프 고정 길이 1. DGLDataset 베이스 요구사항.
    def __len__(self):
        return 1


class OGBHeteroDGLDatasetMassive(DGLDataset):
    # [한국어] OGB-MAG Heterogeneous 로더 (paper/author/institute, papers=121,751,666).
    #   - MAG feature 는 768 차원 (IGBH 1024 와 다름).
    #   - graph_data 는 사전 저장된 torch CSC 파일(/mnt/nvme22/OGB_csc.pth) 을 로드.
    #   - paper 노드에 대해 60/20/20 순차 분할 mask 생성 (randperm 아님).
    # [한국어] 생성자.
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260M')

    # [한국어] process - MAG heterograph 구성.
    def process(self):
        # [한국어] edge_index 로드. OGB-MAG 디렉토리 규칙 (triple underscore 구분자).
        if(self.args.uva_graph or self.args.in_memory):
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir,  'processed','author___affiliated_with___institution', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, 'processed', 'author___writes___paper', 'edge_index.npy')))
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir,  'processed',          'paper___cites___paper', 'edge_index.npy')))


        else:
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir,  'processed','author___affiliated_with___institution', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, 'processed', 'author___writes___paper', 'edge_index.npy'), mmap_mode='r'))
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir,  'processed',          'paper___cites___paper', 'edge_index.npy'), mmap_mode='r'))

        num_paper_nodes = 121751666                                          # [한국어] OGB-MAG paper 노드 수 상수.


#        print("a u e: ", affiliation_author_edges[0])
#        print("affiliation_author_edges: ", len(affiliation_author_edges))
#        print("autho paper len: ", len(author_paper_edges))
#        print("paper_paper_edges: ", len(paper_paper_edges))

        print("paper node feature load")
        #paper_node_features = torch.from_numpy(np.load(osp.join(self.dir, 'processed', 'paper', 'node_feat.npy'), mmap_mode='r')).to(torch.float32)
        
        #paper_node_features = torch.from_numpy(np.memmap(osp.join(self.dir,   'processed',  'paper', 'node_feat.npy'), dtype='float32', mode='r',  shape=(num_paper_nodes,768)))
        # [한국어] MAG feature 는 768 차원 (IGBH 와 다름). 사전 생성한 memmap 파일에서 로드.
        paper_node_features = torch.from_numpy(np.memmap("/mnt/nvme22/MAG_node_feat_mmap.npy",  dtype='float32', mode='r',  shape=(num_paper_nodes,768)))
        print("node feature tensor dim: ", paper_node_features.shape)
#        paper_node_labels = torch.from_numpy(np.memmap(osp.join(self.dir,  'processed', 'paper', 'node_label.npy'), dtype='long', mode='r',  shape=(num_paper_nodes))).to(torch.long)
        paper_node_labels = torch.from_numpy(np.load(osp.join(self.dir, 'processed', 'paper', 'node_label.npy'), mmap_mode='r')).to(torch.long)
        paper_node_labels[paper_node_labels<0]=0                              # [한국어] 음수 라벨(결측) → 0 으로 치환해 cross-entropy 안전화.
        num_author_nodes = 122383112
        num_institute = 25721
        
        print("min label: ", torch.min(paper_node_labels))
        print("max label: ", torch.max(paper_node_labels))
        #num_author_nodes = 0
        #num_institute = 0
        #num_fos = 0


        num_nodes_dict = {'paper': num_paper_nodes, 'author': num_author_nodes, 'institute': num_institute  }
        #graph_data = {
       #     ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
       #     ('author', 'writes', 'paper'): (author_paper_edges[:, 0], author_paper_edges[:, 1]),
       #     ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[:, 0], affiliation_author_edges[:, 1])
       # }

       # graph_data = {
       #     ('paper', 'cites', 'paper'): (paper_paper_edges[0,:], paper_paper_edges[1,:]),
       #     ('author', 'writes', 'paper'): (author_paper_edges[0,:], author_paper_edges[1,:]),
       #     ('author', 'affiliated_to', 'institute'): (affiliation_author_edges[0,:], affiliation_author_edges[1,:])
       # }

        # [한국어] 사전 변환된 DGL heterograph CSC 구조 로드 — edge_index 재처리 비용 절약.
        graph_data = torch.load("/mnt/nvme22/OGB_csc.pth")
        print("dgl.heterograph init starting")
        self.graph = dgl.heterograph(graph_data, num_nodes_dict)

        self.graph =  self.graph.formats('csc')                              # [한국어] sampler 에 유리한 포맷.

        print("dgl.heterograph init successful")
        self.graph.predict = 'paper'
#        self.graph = dgl.remove_self_loop(self.graph, etype='cites')
#        self.graph = dgl.add_self_loop(self.graph, etype='cites')
        
        self.graph.nodes['paper'].data['feat'] = paper_node_features
        self.graph.num_paper_nodes = paper_node_features.shape[0]
        self.graph.nodes['paper'].data['label'] = paper_node_labels
        
        n_nodes = paper_node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        # [한국어] OGB-MAG 은 순차 분할(랜덤 X). time-based split 의 단순 근사.
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True

        self.graph.nodes['paper'].data['train_mask'] = train_mask
        self.graph.nodes['paper'].data['val_mask'] = val_mask
        self.graph.nodes['paper'].data['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


if __name__ == '__main__':
    # [한국어] 단독 실행 시 IGBHeteroDGLDatasetMassive 생성/변환을 타이밍하는 smoke 테스트.
    #          실 학습은 heterogeneous_train.py 에서 import 하여 사용.
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mnt/nvme16/IGB260M_part_2',
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='full',
        choices=['experimental', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=2983, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    args = parser.parse_args()
    start = time.time()                                                      # [한국어] 총 로딩 시간 측정 시작.

    dataset =  IGBHeteroDGLDatasetMassive(args)                              # [한국어] massive 로더 생성 (생성자에서 process 수행).
    g = dataset[0]                                                            # [한국어] heterograph 추출.
    print(g)                                                                  # [한국어] 구조 출력.
    homo_g = dgl.to_homogeneous(g)                                            # [한국어] 이종→동종 변환 — 크기 추정 smoke.
    print("Time taken: ", time.time() - start)                                # [한국어] 총 경과 시간.
