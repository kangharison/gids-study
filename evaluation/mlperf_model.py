"""
[한국어] MLPerf 참조용 Relational GNN 모델 정의 (mlperf_model.py)

=== 파일의 역할 ===
MLPerf-style IGBH 벤치마크에서 사용되는 **이종(heterogeneous) 관계형 GNN 모델**
클래스 `RGNN` 을 정의한다. edge type(관계)별로 별도 convolution 을 두고
DGL 의 HeteroGraphConv 로 묶어 message passing 을 수행하며, 'rsage'(SAGEConv)
와 'rgat'(GATConv) 두 가지 백본을 옵션으로 제공한다. 최종 분류는 'paper' 노드
타입에 대해 Linear 로 사상한다.

=== 전체 아키텍처에서의 위치 ===
heterogeneous_train*.py (MLPerf 설정) → RGNN(forward(blocks, x)) → DGL
HeteroGraphConv → SAGEConv/GATConv → message passing on block(MFG).
feature x 는 GIDS_DGLDataLoader 가 SSD 에서 fetch 해 GPU 에 올려준 텐서.

=== 타 모듈과의 연결 ===
- `evaluation/models.py` 의 RSAGE/RGAT/RGCN 과 중복되는 관계형 모델이지만,
  MLPerf spec 에 맞춰 단순화된 버전. 학습 스크립트가 어느 것을 import 할지
  선택한다.
- DGL: HeteroGraphConv, SAGEConv, GATConv, apply_each.
- 입력 blocks: DGL NeighborSampler 가 만든 layer 별 MFG 리스트.

=== 주요 함수/구조체 요약 ===
- class RGNN(nn.Module):
    * __init__(etypes, in_dim, h_dim, out_dim, num_layers, dropout, model,
               heads, node_type, with_trim):
        edge type 리스트 `etypes` 각각에 대해 동일 구조의 conv 를 만들고
        HeteroGraphConv 로 묶어 `self.convs` 에 쌓는다.
    * forward(blocks, x):
        layer 별로 block 기반 heterograph conv → flatten(multi-head 차원) →
        ReLU + Dropout 를 반복, 마지막에 node_type='paper' 의 히든을 Linear.
"""

import torch                                 # [한국어] PyTorch 기본 텐서/Module API.
import torch.nn.functional as F              # [한국어] F.relu 등 functional 연산.

# from torch_geometric.nn import HeteroConv, GATConv, GCNConv, SAGEConv
# from torch_geometric.utils import trim_to_layer
# [한국어] PyG(Torch Geometric) 대안 구현 주석 — 원본 저자가 비교용으로 남겨둔 흔적.
#          본 파일은 DGL 경로만 사용.

from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, HeteroGraphConv
# [한국어] DGL 의 PyTorch 백엔드 GNN 레이어:
#          - GATConv  : Graph Attention Network (Velickovic'18) - attention 기반 message weighting.
#          - GraphConv: 기본 GCN (Kipf'17) - 정규화 인접행렬 전파.
#          - SAGEConv : GraphSAGE (Hamilton'17) - neighbor 샘플+집계.
#          - HeteroGraphConv: 이종 그래프에서 edge type 별 conv dict 를 묶어 호출.
from dgl import apply_each                   # [한국어] node-type dict {ntype: Tensor} 에 동일 함수를 일괄 적용하는 헬퍼.


class RGNN(torch.nn.Module):
#  r""" [Relational GNN model](https://arxiv.org/abs/1703.06103).

#   Args:
#     etypes: edge types.
#     in_dim: input size.
#     h_dim: Dimension of hidden layer.
#     out_dim: Output dimension.
#     num_layers: Number of conv layers.
#     dropout: Dropout probability for hidden layers.
#     model: "rsage" or "rgat".
#     heads: Number of multi-head-attentions for GAT.
#     node_type: The predict node type for node classification.

#   """
    # [한국어] 필드 역할:
    #   - self.node_type  : 분류 대상 노드 타입명. None 이면 마지막 conv 가 out_dim
    #                        까지 직접 투영, 아니면 별도 Linear(self.lin)가 맡는다.
    #   - self.lin        : node_type 지정 시 h_dim→out_dim 분류기.
    #   - self.convs      : HeteroGraphConv 레이어들의 ModuleList (layer 수=num_layers).
    #   - self.dropout    : 중간 레이어 활성값에 적용되는 Dropout.
    #   - self.with_trim  : PyG 의 trim_to_layer 유사 최적화 플래그 (현재 forward 에서 미사용).
    #
    # [한국어]
    # __init__ - RGNN 구성자. 관계형 이종 GNN 을 layer 수만큼 쌓는다.
    #
    # @param etypes   : DGL heterograph 의 edge type 리스트 (dataloader 로부터 전달).
    # @param in_dim   : 첫 레이어 입력 feature 차원 (paper feat dim = 1024).
    # @param h_dim    : hidden 차원.
    # @param out_dim  : 최종 분류 클래스 수.
    # @param num_layers: 레이어 수 (fan_out 길이와 일치해야 함).
    # @param dropout  : dropout 확률.
    # @param model    : 'rsage' 또는 'rgat' - 각 edge type 별 백본 선택.
    # @param heads    : 'rgat' 에서 multi-head attention 수. h_dim 이 heads 로 나누어 떨어져야 함.
    # @param node_type: 분류 대상 노드 타입('paper'). None 이면 마지막 layer out_dim 으로 직접.
    # @param with_trim: PyG trim 최적화 호환성 플래그(현재 forward 경로엔 미반영).
    #
    # 호출 체인:
    #   heterogeneous_train.py → RGNN(...) → 각 HeteroGraphConv → SAGEConv/GATConv.
    def __init__(self, etypes, in_dim, h_dim, out_dim, num_layers=2, dropout=0.2, model='rgat', heads=4, node_type=None, with_trim=False):
        super().__init__()                                   # [한국어] nn.Module 기본 초기화.
        self.node_type = node_type                           # [한국어] 분류 노드 타입 저장.
        if node_type is not None:                            # [한국어] 지정 시 최종 Linear 필요.
            self.lin = torch.nn.Linear(h_dim, out_dim)       # [한국어] h_dim→out_dim 선형 분류기 y = xW^T + b.

        self.convs = torch.nn.ModuleList()                   # [한국어] layer 리스트 준비. nn.Module 들의 파라미터 자동 등록.
        for i in range(num_layers):                          # [한국어] 각 레이어 생성 루프.
            in_dim = in_dim if i == 0 else h_dim             # [한국어] 첫 레이어는 외부 in_dim, 이후는 직전 h_dim.
            h_dim = out_dim if (i == (num_layers - 1) and node_type is None) else h_dim
                                                              # [한국어] node_type 미지정 + 마지막 레이어면 출력을 바로 out_dim 으로.
            if model == 'rsage':                             # [한국어] Relational GraphSAGE 백본.
                self.convs.append(HeteroGraphConv({          # [한국어] edge type 별 SAGEConv 묶음.
                    etype: SAGEConv(in_dim, h_dim, root_weight=False)  # [한국어] aggregator='mean'(기본), self loop 미사용.
                    for etype in etypes}))
            elif model == 'rgat':                            # [한국어] Relational GAT 백본.
                self.convs.append(HeteroGraphConv({          # [한국어] edge type 별 multi-head attention conv.
                    etype: GATConv(in_dim, h_dim // heads, heads)       # [한국어] 출력 shape = (N, heads, h_dim/heads).
                    for etype in etypes}))
        self.dropout = torch.nn.Dropout(dropout)             # [한국어] 중간 layer activation dropout.
        self.with_trim = with_trim                           # [한국어] 플래그 보관 — 현재 forward 에서는 소비되지 않음.

    # [한국어]
    # forward - blocks(MFG 리스트)와 입력 feature dict x 를 받아 노드 임베딩을 계산한다.
    #
    # @param blocks: DGL NeighborSampler 가 만든 layer 별 block(MFG) 리스트.
    #                blocks[i] = i 번째 layer 의 message flow graph.
    # @param x     : {ntype: Tensor(N_src, feat_dim)} - 초기 feature dict
    #                (GIDS 로더가 SSD → GPU 로 채움).
    # @return      : Tensor(batch_size, out_dim) - 'paper' 노드에 대한 로짓.
    #
    # 동작:
    #   각 layer 에서
    #     (1) HeteroGraphConv 로 message passing 수행 (edge type 별 conv 후 집계).
    #     (2) 다중헤드 축(heads × per-head dim) 을 하나로 flatten.
    #     (3) 마지막 layer 가 아니면 ReLU + Dropout.
    #   최종적으로 'paper' 타입 임베딩을 Linear 로 분류기 적용.
    # 실행 컨텍스트: PyTorch forward, GPU. autograd 기록 대상.
    def forward(self, blocks, x):
        h = x                                                # [한국어] 첫 입력을 layer 0 입력으로 설정.
        for l, (layer, block) in enumerate(zip(self.convs, blocks)):
            # [한국어] layer 별 HeteroGraphConv 호출. block 은 해당 layer 의 MFG,
            #          message 는 src→dst 방향으로 전파되며 결과는 dst node 에 모임.
            h = layer(block, h)
            # [한국어] {ntype: Tensor(N, heads, D)} 를 Tensor(N, heads*D) 로 평탄화.
            #          GATConv 의 multi-head 출력 차원을 단일 벡터로 합치기.
            h = apply_each(h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.convs) - 1:                     # [한국어] 마지막 layer 가 아니면 비선형+dropout.
                h = apply_each(h, F.relu)                    # [한국어] 각 ntype 텐서에 ReLU.
                h = apply_each(h, self.dropout)              # [한국어] dropout 으로 과적합 완화.
        return self.lin(h['paper'])                          # [한국어] 'paper' 노드 hidden → 클래스 로짓.
