"""
[한국어] evaluation 학습 스크립트가 사용하는 GNN 모델 모음 (models.py)

=== 파일의 역할 ===
homogeneous/heterogeneous 두 계열의 GNN 모델(SAGE, GCN, GAT / RGCN, RSAGE,
RGAT) 을 DGL 레이어로 구현한다. 각 모델의 forward 는 DGL 의 NeighborSampler 가
만들어준 block(Message Flow Graph) 리스트를 받아, layer 별로 소스 노드
feature → 대상 노드 feature 로 메시지를 전달하며 차곡차곡 임베딩을 축약한다.

=== 전체 아키텍처에서의 위치 ===
homogenous_train*.py / heterogeneous_train*.py → <Model>(in_feats, h_feats,
num_classes, …) → forward(blocks, x). x 는 GIDS/BaM dataloader 가 SSD 에서
fetch 해 GPU 로 올려준 feature 텐서(동종) 또는 {ntype: Tensor} dict(이종).
출력은 분류 로짓. evaluation/mlperf_model.py 는 MLPerf 스펙의 축약형 RGNN 만
포함하고, 이 파일은 baseline/변형 비교 실험용 모델 집합을 제공한다.

=== 타 모듈과의 연결 ===
- DGL: SAGEConv/GraphConv/GATConv/HeteroGraphConv + apply_each.
- 학습 스크립트: `from models import *` 로 전 심볼 import.
- blocks 는 DGL sampler/GIDS_DGLDataLoader 가 공통적으로 만든다.

=== 주요 함수/구조체 요약 ===
- SAGE(nn.Module): GraphSAGE homogeneous. mean aggregator 기반.
- GCN (nn.Module): Kipf'17 GCN. 정규화 인접행렬 전파.
- GAT (nn.Module): multi-head attention. 마지막 layer mean over heads.
- RGCN(nn.Module): HeteroGraphConv({etype: GraphConv}, aggregate='mean').
- RSAGE(nn.Module): 같은 구조에 SAGEConv 'gcn' aggregator.
- RGAT(nn.Module): edge type 별 GATConv(h/heads, heads), flatten after.

수학적으로 각 message passing 은
  h_v^{(l+1)} = σ( AGG_{u∈N(v)} MSG_W(h_u^{(l)}, h_v^{(l)}) )
형태이며, AGG 는 mean(SAGE/GCN/RSAGE) 또는 attention weighted sum(GAT/RGAT).
"""

import torch.nn as nn                                   # [한국어] nn.Module / nn.ModuleList / nn.Dropout / nn.Linear.
import torch.nn.functional as F                         # [한국어] F.relu 등 stateless activation.
from dgl import apply_each                              # [한국어] {ntype: Tensor} 각각에 함수 일괄 적용 헬퍼 (heterograph 용).
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, HeteroGraphConv
# [한국어] DGL GNN layer 들. 각 레이어는 내부에서 DGL 의 message/reduce API 를 구현한다.


class SAGE(nn.Module):
    # [한국어] GraphSAGE (homogeneous).
    # 수학식: h_v^{(l+1)} = W · mean({h_u^{(l)} : u ∈ N(v)} ∪ {h_v^{(l)}}).
    # aggregator='mean' 으로 이웃 평균 → 선형변환.
    # [한국어]
    # __init__ - SAGE GNN 레이어 스택 구성.
    # @param in_feats    : 입력 feature 차원.
    # @param h_feats     : hidden dim.
    # @param num_classes : 출력 클래스 수.
    # @param num_layers  : 레이어 수 (>=2 가정. fan_out 와 일치해야 함).
    # @param dropout     : dropout 확률.
    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super(SAGE, self).__init__()                                 # [한국어] nn.Module 초기화.
        self.layers = nn.ModuleList()                                # [한국어] 파라미터 자동 등록 되는 layer 리스트.
        self.layers.append(SAGEConv(in_feats, h_feats, aggregator_type='mean'))
        # [한국어] 첫 layer: in_feats → h_feats. 이웃 평균 + Linear.
        for _ in range(num_layers-2):
            self.layers.append(SAGEConv(h_feats, h_feats, aggregator_type='mean'))
            # [한국어] 중간 layer 들: h_feats → h_feats 유지.
        self.layers.append(SAGEConv(h_feats, num_classes, aggregator_type='mean'))
        # [한국어] 마지막 layer: h_feats → num_classes 로 직접 로짓 출력.
        self.dropout = nn.Dropout(dropout)                           # [한국어] 중간 layer activation dropout.

    # [한국어]
    # forward - block 리스트를 따라 message passing 을 순차 수행.
    # @param blocks: DGL sampler 가 만든 MFG 리스트 (layer 수와 동일 길이).
    # @param x     : src 노드 feature 텐서.
    # @return      : 최상위 대상 노드에 대한 로짓.
    def forward(self, blocks, x):
        h = x                                                         # [한국어] layer0 입력.
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]                         # [한국어] block 의 dst 노드 슬라이스.
                                                                      #          DGL bipartite MFG: 첫 num_dst_nodes() 는 dst 노드들.
            h = layer(block, (h, h_dst))                              # [한국어] (src_feat, dst_feat) 튜플로 conv. 반환은 dst 노드 embed.
            if l != len(self.layers) - 1:                             # [한국어] 마지막 제외 비선형+dropout.
                h = self.dropout(h)                                   # [한국어] dropout 적용.
                h = F.relu(h)                                         # [한국어] ReLU 활성.


        return h                                                      # [한국어] 마지막 layer 결과 = 로짓.

class GCN(nn.Module):
    # [한국어] Kipf'17 GCN (homogeneous).
    # 수학식: H^{(l+1)} = σ( D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)} ),  Ã = A + I.
    # [한국어] __init__ - GCN 스택 (파라미터 의미는 SAGE 동일).
    def __init__(self, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super(GCN, self).__init__()                                    # [한국어] Module 초기화.
        self.layers = nn.ModuleList()                                  # [한국어] layer 리스트.
        self.layers.append(GraphConv(in_feats, h_feats))               # [한국어] 첫 GCN 레이어. 정규화 인접행렬 전파 + Linear.
        for _ in range(num_layers-2):
            self.layers.append(GraphConv(h_feats, h_feats))            # [한국어] 중간 동일 차원 GCN.
        self.layers.append(GraphConv(h_feats, num_classes))            # [한국어] 마지막 → 클래스 로짓.
        self.dropout = nn.Dropout(dropout)                             # [한국어] dropout.

    # [한국어] forward - block 별로 GCN 전파. SAGE.forward 와 구조 동일.
    def forward(self, blocks, x):
        h = x                                                           # [한국어] 초기 feature.
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]                           # [한국어] dst 노드 feature 슬라이스.
            h = layer(block, (h, h_dst))                                # [한국어] GCN 적용. (src, dst) 튜플 bipartite forward.
            if l != len(self.layers) - 1:                               # [한국어] 마지막 제외 비선형.
                h = self.dropout(h)                                     # [한국어] dropout.
                h = F.relu(h)                                           # [한국어] ReLU.
        return h                                                        # [한국어] 로짓 반환.

class GAT(nn.Module):
    # [한국어] Graph Attention Network (homogeneous).
    # 수학식: α_{uv} = softmax_u( LeakyReLU(a^T [W h_u || W h_v]) ),
    #         h_v'   = ∥_{k=1..K} σ( Σ_u α^{(k)}_{uv} W^{(k)} h_u )  (hidden layer: concat)
    #         마지막 layer 는 head 평균: h_v' = σ( mean_k Σ α^{(k)} W^{(k)} h_u ).
    # [한국어] __init__ - GAT 스택 구성. 내부 차원 = h_feats * num_heads (concat 처리).
    def __init__(self, in_feats, h_feats, num_classes, num_heads, num_layers=2, dropout=0.2):
        super(GAT, self).__init__()                                          # [한국어] Module 초기화.
        self.layers = nn.ModuleList()                                        # [한국어] layer 리스트.
        self.layers.append(GATConv(in_feats, h_feats, num_heads))            # [한국어] 첫 GAT: in → (heads, h_feats).
        for _ in range(num_layers-2):
            self.layers.append(GATConv(h_feats * num_heads, h_feats, num_heads))
            # [한국어] 중간 레이어 입력은 이전 flatten(heads*h) 결과.
        self.layers.append(GATConv(h_feats * num_heads, num_classes, num_heads))
        # [한국어] 마지막 GAT: heads×num_classes. 이후 forward 에서 head 평균을 취해 num_classes 로짓.
        self.dropout = nn.Dropout(dropout)                                   # [한국어] dropout.

    # [한국어] forward - 중간 layer 는 head concat, 마지막 layer 는 head 평균.
    def forward(self, blocks, x):
        h = x                                                                # [한국어] 초기 feature.
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]                                # [한국어] dst 슬라이스.
            if l < len(self.layers) - 1:                                     # [한국어] 중간 layer.
                h = layer(block, (h, h_dst)).flatten(1)                      # [한국어] (N,heads,D)→(N,heads*D) concat.
                h = F.relu(h)                                                # [한국어] 비선형.
                h = self.dropout(h)                                          # [한국어] dropout.
            else:
                h = layer(block, (h, h_dst)).mean(1)                         # [한국어] 마지막: head 축(mean) 으로 평균.
        return h                                                             # [한국어] 로짓.


class RGCN(nn.Module):
    # [한국어] Relational GCN (Schlichtkrull'18) - 이종 그래프.
    # edge type 마다 별도 GraphConv 를 두고 HeteroGraphConv 가 'mean' 으로 집계.
    # 수학식: h_v^{(l+1)} = σ( Σ_r∈R (1/|N_r(v)|) Σ_{u∈N_r(v)} W_r h_u^{(l)} ).
    # [한국어] __init__ - edge type 별 GraphConv 를 HeteroGraphConv 로 묶어 3단계 쌓는다.
    #          마지막 layer 도 h_feats 유지 → self.linear 로 분류.
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super().__init__()                                                    # [한국어] Module 초기화.
        self.layers = nn.ModuleList()                                         # [한국어] layer 리스트.
        self.layers.append(HeteroGraphConv({
            etype: GraphConv(in_feats, h_feats)
            for etype in etypes}, aggregate='mean'))
        # [한국어] 첫 레이어: 각 edge type 별 in→h GraphConv. aggregate='mean' 는 type 간 평균.
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GraphConv(h_feats, h_feats)
                for etype in etypes}, aggregate='mean'))
            # [한국어] 중간 layer 들: h→h 유지.
        self.layers.append(HeteroGraphConv({
            etype: GraphConv(h_feats, h_feats)
            for etype in etypes}, aggregate='mean'))
        # [한국어] 마지막 layer 도 h→h (로짓은 아래 Linear 에서).
        self.dropout = nn.Dropout(dropout)                                    # [한국어] dropout.
        self.linear = nn.Linear(h_feats, num_classes)                         # [한국어] 최종 분류기. 'paper' 타입에만 적용.

    # [한국어] forward - {ntype: feat} dict 를 layer 별 전파.
    def forward(self, blocks, x):
        h = x                                                                  # [한국어] 초기 node feature dict.
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)                                                # [한국어] HeteroGraphConv 적용. 결과도 dict.
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))
            # [한국어] view 는 shape 재확인 (일부 DGL 버전이 (N,1,D) 로 주는 경우 squeeze 효과).
            if l != len(self.layers) - 1:                                      # [한국어] 마지막 제외.
                h = apply_each(h, F.relu)                                      # [한국어] 각 ntype 에 ReLU.
                h = apply_each(h, self.dropout)                                # [한국어] dropout.
        return self.linear(h['paper'])                                         # [한국어] 분류 대상 'paper' 만 Linear 로.

class RSAGE(nn.Module):
    # [한국어] Relational GraphSAGE (이종). SAGEConv 'gcn' aggregator 사용.
    # 각 edge type 별로 GraphSAGE 를 돌리고 HeteroGraphConv 기본 집계(sum).
    # [한국어] __init__ - RSAGE 스택. edge type 별 SAGEConv('gcn') 를 HeteroGraphConv 로 묶음.
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2):
        super().__init__()                                                    # [한국어] Module 초기화.
        self.layers = nn.ModuleList()                                         # [한국어] layer 리스트.
        self.layers.append(HeteroGraphConv({
            etype: SAGEConv(in_feats, h_feats, 'gcn')
            for etype in etypes}))
        # [한국어] 첫 layer: in→h, SAGE 'gcn' aggregator (Σ_u W h_u / (d+1) 유사).
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: SAGEConv(h_feats, h_feats, 'gcn')
                for etype in etypes}))
            # [한국어] 중간 layer 들: h→h.
        self.layers.append(HeteroGraphConv({
            etype: SAGEConv(h_feats, h_feats, 'gcn')
            for etype in etypes}))
        # [한국어] 마지막 layer: h→h. 로짓은 self.linear 담당.
        self.dropout = nn.Dropout(dropout)                                    # [한국어] dropout.
        self.linear = nn.Linear(h_feats, num_classes)                         # [한국어] 'paper' 분류기.

    # [한국어] forward - RGCN.forward 와 구조 동일. HeteroGraphConv 만 차이.
    def forward(self, blocks, x):
        h = x                                                                  # [한국어] 초기 feature dict.
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)                                                # [한국어] RSAGE 전파.
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1]))        # [한국어] shape 정리.
            if l != len(self.layers) - 1:                                      # [한국어] 마지막 제외.
                h = apply_each(h, F.relu)                                      # [한국어] ReLU.
                h = apply_each(h, self.dropout)                                # [한국어] dropout.
        return self.linear(h['paper'])                                         # [한국어] 분류.

class RGAT(nn.Module):
    # [한국어] Relational GAT (이종). edge type 별 GATConv(h//heads, heads).
    # HeteroGraphConv 기본 집계(sum) 로 multi-relation attention 결합.
    # [한국어] __init__ - RGAT 스택. 각 layer 에서 GATConv(in, h/heads, heads).
    def __init__(self, etypes, in_feats, h_feats, num_classes, num_layers=2, dropout=0.2, n_heads=4):
        super().__init__()                                                    # [한국어] Module 초기화.
        self.layers = nn.ModuleList()                                         # [한국어] layer 리스트.
        self.layers.append(HeteroGraphConv({
            etype: GATConv(in_feats, h_feats // n_heads, n_heads)
            for etype in etypes}))
        # [한국어] 첫 layer: in_feats → (heads, h_feats/heads). concat 후 h_feats.
        for _ in range(num_layers-2):
            self.layers.append(HeteroGraphConv({
                etype: GATConv(h_feats, h_feats // n_heads, n_heads)
                for etype in etypes}))
            # [한국어] 중간 layer: h_feats → (heads, h/heads) → concat h_feats.
        self.layers.append(HeteroGraphConv({
            etype: GATConv(h_feats, h_feats // n_heads, n_heads)
            for etype in etypes}))
        # [한국어] 마지막 layer: 동일 차원 유지. 최종 분류는 self.linear.
        self.dropout = nn.Dropout(dropout)                                    # [한국어] dropout.
        self.linear = nn.Linear(h_feats, num_classes)                         # [한국어] 'paper' 로짓 분류.

    # [한국어] forward - RGAT. GATConv 출력 (N,heads,D/heads) 를 flatten 으로 (N, heads*D/heads)=h_feats.
    def forward(self, blocks, x):
        h = x                                                                  # [한국어] 초기 feature dict.
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)                                                # [한국어] RGAT 전파. 결과 dict {ntype:(N,heads,D)}.
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            # [한국어] multi-head 축 flatten → (N, heads*D).
            if l != len(self.layers) - 1:                                      # [한국어] 마지막 제외.
                h = apply_each(h, F.relu)                                      # [한국어] ReLU.
                h = apply_each(h, self.dropout)                                # [한국어] dropout.
        return self.linear(h['paper'])                                         # [한국어] 분류.


