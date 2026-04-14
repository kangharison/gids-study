# referenced the following implementation: https://github.com/BarclayII/dgl/blob/ladies/examples/pytorch/ladies/ladies2.py
"""
[한국어] LADIES (Layer-wise Adaptive Importance Sampling) 샘플러 (ladies_sampler.py)

=== 파일의 역할 ===
GNN 미니배치 학습에서 **layer-wise importance sampling** 기반 이웃 샘플링을
수행하는 커스텀 DGL BlockSampler. layer 마다 고정된 수의 노드만을 중요도에
비례 확률로 뽑아 node-explosion 을 억제한다. 노드 선택 후에는 선택된 노드 간
실제 edge 로 구성된 bipartite block(MFG) 을 만들고, 각 edge 에 importance
reweighting(W_tilde) 를 부여한다.

=== 전체 아키텍처에서의 위치 ===
DGL DataLoader / GIDS_DGLDataLoader → LadiesSampler.sample_blocks(g,
seed_nodes) → blocks 리스트 (학습 모델 forward 로 전달). NeighborSampler
대안으로, IGB/OGB 실험에서 노드 수 제약이 엄격할 때 선택된다.

=== 타 모듈과의 연결 ===
- dgl (compact_graphs, in_subgraph, reverse, ops.copy_e_sum 등).
- 학습 스크립트에서 sampler=LadiesSampler(nodes_per_layer=[...]) 로 주입.
- fetch 대상 노드는 block.srcdata[dgl.NID] 로 추출해 GIDS 가 SSD 에서 읽는다.

=== 주요 함수/구조체 요약 ===
- find_indices_in(a,b)   : 배열 a 의 각 원소가 b 안에서 갖는 인덱스(없으면 0 fallback).
- union(*arrays)         : torch.cat 후 unique.
- normalized_edata(g, w) : 각 edge 를 dst in-degree 로 정규화한 가중치 반환.
- class LadiesSampler(BlockSampler):
    * __init__(nodes_per_layer, importance_sampling, weight, out_weight, replace)
    * compute_prob(g, seed_nodes, weight, num):
        candidate subgraph + 각 후보 노드 중요도(prob) 계산.
    * select_neighbors(prob, num):
        multinomial 로 `num` 개 이웃 인덱스 추출.
    * generate_block(insg, neighbor_nodes_idx, seed_nodes, P_sg, W_sg):
        선택된 이웃 + seed 로 block 구성, edge 가중치 W_tilde 부여.
    * sample_blocks(g, seed_nodes, exclude_eids):
        layer 역순으로 반복해 blocks 리스트 구성.
"""

import dgl                                       # [한국어] DGL core. heterograph 아닌 homogeneous 그래프 전제.
import dgl.function as fn                         # [한국어] 메시지 함수 (fn.copy_e, fn.sum 등).
import torch                                      # [한국어] 텐서 / 다항분포 샘플링.


# [한국어]
# find_indices_in - 배열 a 의 각 원소가 배열 b 에서 처음 등장하는 위치를 찾는다.
#   정렬 + searchsorted 로 O(N log N) 구현. b 에 존재하지 않는 원소는 잘못된
#   인덱스를 낼 수 있어 `indices.shape[0]` 초과 시 0 으로 clamp.
# @param a, b: 1D LongTensor.
# @return: a 와 같은 길이의 인덱스 텐서.
def find_indices_in(a, b):
    b_sorted, indices = torch.sort(b)                                  # [한국어] b 를 오름차순 정렬. indices = 원본 인덱스.
    sorted_indices = torch.searchsorted(b_sorted, a)                   # [한국어] a 각 원소가 정렬된 b 에서 삽입될 위치.
    sorted_indices[sorted_indices >= indices.shape[0]] = 0             # [한국어] out-of-range 는 0 fallback (미존재 원소 방어).
    return indices[sorted_indices]                                     # [한국어] 정렬 전 원 인덱스로 환원해 반환.


# [한국어]
# union - 임의 개수의 1D tensor 들을 concat 후 unique 값만 반환.
def union(*arrays):
    return torch.unique(torch.cat(arrays))                             # [한국어] cat → unique. 순서는 오름차순 정렬됨.


# [한국어]
# normalized_edata - 각 edge 에 1/deg(dst) 가중치를 부여해 반환.
# @param g     : DGL 그래프.
# @param weight: edge weight 이름 (기본 None → 모두 1 로 초기화하여 "W" 사용).
# @return      : edge 당 정규화 가중치 텐서.
# 실행 컨텍스트: 샘플링 전처리 단계. g.local_scope() 블록 안에서 임시 수정만 수행.
def normalized_edata(g, weight=None):
    with g.local_scope():                                              # [한국어] 이 블록 내 변경은 종료 시 롤백.
        if weight is None:                                             # [한국어] 가중치 미지정 시 1 로 초기화.
            weight = "W"
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, "v"))   # [한국어] 각 dst 노드에 들어오는 edge weight 합산 → ndata['v'].
        g.apply_edges(lambda edges: {"w": 1 / edges.dst["v"]})         # [한국어] 각 edge 를 dst 합으로 나눔 (= row-stochastic).
        return g.edata["w"]                                            # [한국어] 정규화된 edge weight 반환.


class LadiesSampler(dgl.dataloading.BlockSampler):
    # [한국어] DGL BlockSampler 상속. DataLoader 가 layer 별 block 을 요청할 때
    #          sample_blocks 가 호출된다.
    # 필드:
    #   nodes_per_layer     : layer 별 샘플 노드 수 리스트.
    #   importance_sampling : True 면 edge_weight^2 기반 importance, False 면 uniform.
    #   edge_weight         : g.edata 에서 가중치 키.
    #   output_weight       : 생성된 block 에 붙일 edge weight 키.
    #   replace             : multinomial 복원 여부.
    # [한국어]
    # __init__ - LADIES 파라미터 저장.
    # @param nodes_per_layer: layer 별 뽑을 노드 수(root 층부터의 역순으로 해석).
    # @param importance_sampling: True 면 edge_weight^2 기반 importance sampling.
    # @param weight      : 입력 edge weight 키.
    # @param out_weight  : 생성된 block edge weight 키.
    # @param replace     : multinomial 복원 추출 여부.
    def __init__(
        self,
        nodes_per_layer,
        importance_sampling=True,
        weight="w",
        out_weight="edge_weights",
        replace=False,
    ):
        super().__init__()                                              # [한국어] BlockSampler 기본 초기화.
        self.nodes_per_layer = nodes_per_layer                          # [한국어] layer 별 이웃 후보 노드 수.
        self.importance_sampling = importance_sampling                  # [한국어] importance vs uniform.
        self.edge_weight = weight                                       # [한국어] 입력 edge weight 이름.
        self.output_weight = out_weight                                 # [한국어] 출력 block edge weight 이름.
        self.replace = replace                                          # [한국어] multinomial 복원 플래그.

    # [한국어]
    # compute_prob - seed_nodes 의 in-subgraph 를 compact 해 후보 노드 집합을 만들고,
    #                각 후보의 중요도 확률을 계산한다.
    # @param g        : 전체 그래프.
    # @param seed_nodes: 현재 layer dst 노드.
    # @param weight   : edge weight 텐서.
    # @param num      : 목표 샘플 수 (현재 함수 내에서는 참조 안 됨, API 호환).
    # @return (prob, insg) : 후보 확률(1D) + compact 된 in-subgraph.
    # 실행 컨텍스트: DataLoader worker / 메인 프로세스. CPU 기반 연산.
    def compute_prob(self, g, seed_nodes, weight, num):
        """
        g : the whole graph
        seed_nodes : the output nodes for the current layer
        weight : the weight of the edges
        return : the unnormalized probability of the candidate nodes, as well as the subgraph
                 containing all the edges from the candidate nodes to the output nodes.
        """
       
        seed_nodes = seed_nodes.to("cpu")                               # [한국어] CPU 로 이동 (DGL in_subgraph/compact 가 CPU 경로에 최적).
        insg = dgl.in_subgraph(g, seed_nodes)                           # [한국어] seed_nodes 로 들어오는 edge 를 가진 서브그래프 추출.

        insg = dgl.compact_graphs(insg, seed_nodes)                     # [한국어] 고립 노드 제거 + 노드 재번호(compact). seed_nodes 의 ID 는 보존.

        if self.importance_sampling:                                    # [한국어] edge_weight^2 기반 importance.

            out_frontier = dgl.reverse(insg, copy_edata=True)           # [한국어] edge 방향 반전 — 각 후보 노드가 "보낼" 합을 계산하기 위함.
            weight = weight[out_frontier.edata[dgl.EID].long()]         # [한국어] 원본 edge weight 중 현재 subgraph edge 에 해당하는 것만 선택.
            prob = dgl.ops.copy_e_sum(out_frontier, weight**2)          # [한국어] 각 노드로 모이는 weight^2 의 합 → 중요도.
            # prob = torch.sqrt(prob)
            # [한국어] 논문에 따라 sqrt 가 적용될 수도 있으나 여기선 제곱합 자체를 확률로 사용.
        else:
            prob = torch.ones(insg.num_nodes())                         # [한국어] uniform 샘플링.
            prob[insg.out_degrees() == 0] = 0                           # [한국어] out-degree 0 인 노드는 선택 제외 (확률 0).
        return prob, insg                                               # [한국어] 후보 확률 + compact subgraph 반환.

    # [한국어]
    # select_neighbors - 주어진 확률로 min(num, 후보수) 개의 이웃 인덱스를 뽑는다.
    # @param prob: 후보 노드 확률.
    # @param num : 원하는 샘플 수.
    # @return    : 샘플된 인덱스(subgraph local).
    def select_neighbors(self, prob, num):
        """
        seed_nodes : output nodes
        cand_nodes : candidate nodes.  Must contain all output nodes in @seed_nodes
        prob : unnormalized probability of each candidate node
        num : number of neighbors to sample
        return : the set of input nodes in terms of their indices in @cand_nodes, and also the indices of
                 seed nodes in the selected nodes.
        """
        # The returned nodes should be a union of seed_nodes plus @num nodes from cand_nodes.
        # Because compute_prob returns a compacted subgraph and a list of probabilities,
        # we need to find the corresponding local IDs of the resulting union in the subgraph
        # so that we can compute the edge weights of the block.
        # This is why we need a find_indices_in() function.
       
        min_val = min(int(num), int(prob.shape[0]))                    # [한국어] 후보 노드 수보다 num 이 크면 clamp.
        neighbor_nodes_idx = torch.multinomial(
            prob, min_val, replacement=self.replace
        )
        # [한국어] 다항분포에서 min_val 개 샘플. replacement=False 이면 비복원.
        return neighbor_nodes_idx                                       # [한국어] subgraph 내 local 인덱스 반환.

    # [한국어]
    # generate_block - 선택된 이웃 + seed 로 bipartite block 을 만들고 edge 가중치를 부여한다.
    # @param insg              : compute_prob 가 돌려준 compact subgraph.
    # @param neighbor_nodes_idx: select_neighbors 로 뽑힌 이웃 인덱스.
    # @param seed_nodes        : 현재 layer dst 노드 (원본 NID).
    # @param P_sg              : 각 노드 확률.
    # @param W_sg              : edge weight.
    # @return                  : DGL block (MFG) — srcdata/dstdata 에 NID 매핑, edata[output_weight] 부여.
    def generate_block(self, insg, neighbor_nodes_idx, seed_nodes, P_sg, W_sg):
        """
        insg : the subgraph yielded by compute_prob()
        neighbor_nodes_idx : the sampled nodes from the subgraph @insg, yielded by select_neighbors()
        seed_nodes_local_idx : the indices of seed nodes in the selected neighbor nodes, also yielded
                               by select_neighbors()
        P_sg : unnormalized probability of each node being sampled, yielded by compute_prob()
        W_sg : edge weights of @insg
        return : the block.
        """
        seed_nodes_idx = find_indices_in(seed_nodes, insg.ndata[dgl.NID])   # [한국어] seed_nodes 가 compact 서브그래프 안에서의 local 인덱스.
        u_nodes = union(neighbor_nodes_idx, seed_nodes_idx)                  # [한국어] 이웃 ∪ seed = 샘플 노드 집합 (최종 block src).
        sg = insg.subgraph(u_nodes.type(insg.idtype))                        # [한국어] u_nodes 만 남긴 서브그래프.
        u, v = sg.edges()                                                     # [한국어] src(u), dst(v) 인덱스 배열.
        lu = sg.ndata[dgl.NID][u.long()]                                      # [한국어] src 노드의 compact NID (insg 기준).
        s = find_indices_in(lu, neighbor_nodes_idx)                           # [한국어] 각 src 가 neighbor_nodes_idx 에서 갖는 인덱스.
        eg = dgl.edge_subgraph(
            sg, lu == neighbor_nodes_idx[s], relabel_nodes=False
        )
        # [한국어] edge 가 "실제 선택된 이웃에서 온" 것만 남기는 필터링.
        eg.ndata[dgl.NID] = sg.ndata[dgl.NID][: eg.num_nodes()]               # [한국어] NID 재매핑 보존.
        eg.edata[dgl.EID] = sg.edata[dgl.EID][eg.edata[dgl.EID].long()]       # [한국어] EID 재매핑 보존.
        sg = eg                                                                # [한국어] 이후는 eg 를 sg 로 사용.
        nids = insg.ndata[dgl.NID][sg.ndata[dgl.NID].long()]                   # [한국어] 원본 그래프 ID 까지 끌어올림.
        P = P_sg[u_nodes.long()]                                               # [한국어] 선택 노드들의 확률 벡터.
        W = W_sg[sg.edata[dgl.EID].long()]                                     # [한국어] 선택 edge 의 weight.
        W_tilde = dgl.ops.e_div_u(sg, W, P)                                    # [한국어] W / P_u — importance 역가중(역확률 보정).
        W_tilde_sum = dgl.ops.copy_e_sum(sg, W_tilde)                          # [한국어] dst 노드별 합산.
        d = sg.in_degrees()                                                    # [한국어] 실제 in-degree.
        W_tilde = dgl.ops.e_mul_v(sg, W_tilde, d / W_tilde_sum)                # [한국어] 재정규화: W_tilde · d_v / Σ W_tilde → 평균 유지하며 합=1.

        block = dgl.to_block(sg, seed_nodes_idx.type(sg.idtype))               # [한국어] sg 를 bipartite block(MFG) 로 변환. dst=seed.
        block.edata[self.output_weight] = W_tilde                              # [한국어] 최종 edge 가중치를 block 에 기록.
        # correct node ID mapping
        block.srcdata[dgl.NID] = nids[block.srcdata[dgl.NID].long()]           # [한국어] src NID 를 원본 ID 로 복원.
        block.dstdata[dgl.NID] = nids[block.dstdata[dgl.NID].long()]           # [한국어] dst NID 복원.

        sg_eids = insg.edata[dgl.EID][sg.edata[dgl.EID].long()]                # [한국어] 원본 EID 트레이스.
        block.edata[dgl.EID] = sg_eids[block.edata[dgl.EID].long()]            # [한국어] block EID 복원.
        return block                                                            # [한국어] 완성된 block.

    # [한국어]
    # sample_blocks - 주어진 seed_nodes 로부터 layer 수만큼 역순으로 sample 하여
    #                 blocks 리스트를 구성해 반환한다. DGL DataLoader 인터페이스.
    # @param g           : 전체 그래프.
    # @param seed_nodes  : 배치의 root dst 노드.
    # @param exclude_eids: 링크 예측 용 제외 edge (미사용).
    # @return (input_nodes, output_nodes, blocks)
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes.to("cpu")                                   # [한국어] 배치의 최종 dst 노드 저장 (CPU 로 복사).
        blocks = []                                                            # [한국어] layer 별 block 누적.
        for block_id in reversed(range(len(self.nodes_per_layer))):            # [한국어] 가장 바깥 layer 부터 역순으로 샘플링.
            num_nodes_to_sample = self.nodes_per_layer[block_id]              # [한국어] 이번 layer 의 타겟 샘플 수.
            W = g.edata[self.edge_weight]                                      # [한국어] 전체 그래프의 edge weight.
            prob, insg = self.compute_prob(
                g, seed_nodes, W, num_nodes_to_sample
            )
            # [한국어] 후보 확률 / compact in-subgraph 획득.

            neighbor_nodes_idx = self.select_neighbors(
                prob, num_nodes_to_sample
            )
            # [한국어] 확률에 따라 num_nodes_to_sample 개 이웃 선택.
            block = self.generate_block(
                insg,
                neighbor_nodes_idx.type(g.idtype),
                seed_nodes.type(g.idtype),
                prob,
                W[insg.edata[dgl.EID].long()],
            )
            # [한국어] 선택 결과로 block 생성 + edge weight 부여.
            seed_nodes = block.srcdata[dgl.NID]                                # [한국어] 다음(더 깊은) layer 의 dst 는 이번 block 의 src.
            blocks.insert(0, block)                                            # [한국어] layer 순서대로 최종 리스트 앞쪽에 삽입.
        return seed_nodes, output_nodes, blocks                                # [한국어] 입력 노드(가장 깊은 layer src), 출력 노드, blocks 반환.

