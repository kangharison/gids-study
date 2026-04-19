# CLAUDE.md — gids-study 프로젝트 지침

이 파일은 `gids-study/` 디렉토리 내에서 주석 작업 시 `company/CLAUDE.md`(공통 방법론)와 함께 읽히는 **프로젝트별 보충 지침**이다. 공통 지침의 필수 4섹션(파일의 역할 / 전체 아키텍처에서의 위치 / 타 모듈과의 연결 / 주요 함수·구조체 요약), 모든 함수·모든 구조체 필드·모든 실행 라인 주석 요구사항은 그대로 적용된다.

## 1. 프로젝트 개요

- **이름**: GIDS (GPU-Initiated Direct Storage) Dataloader.
- **원본**: https://github.com/jeongminpark417/GIDS (`upstream` 리모트로 고정).
- **작업 리포**: https://github.com/kangharison/gids-study (`origin` = main 브랜치).
- **목표**: 테라바이트 규모 GNN(Graph Neural Network) 학습에서, 특성(feature) 텐서를 **GPU가 직접 NVMe SSD에서 읽어오도록** 하여 CPU/메모리 병목을 우회한다.
- **기반 스택**: BaM(=본 저장소의 `bam/` 서브모듈. 이미 파악된 것으로 간주, 주석 작업 제외), DGL(Graph Neural Network 프레임워크), PyTorch, pybind11, CUDA 11+.
- **핵심 논문**: Park 외, "Accelerating Sampling and Aggregation Operations in GNN Frameworks with GPU-Initiated Direct Storage Accesses", VLDB 2024.

## 2. 전체 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│ Python 학습 스크립트 (evaluation/*.py)                               │
│   homogenous_train.py / heterogeneous_train.py / ClusterGCN / …     │
│     └─ DGL sampler → mini-batch의 node ID 리스트 생성                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ node IDs (GPU tensor)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ GIDS_Setup/GIDS/GIDS.py                                              │
│   GIDS_DGLDataLoader / _PrefetchingIter / GIDS class                │
│   - DGL DataLoader 래퍼. sampler가 만든 배치에서 feature fetch 호출 │
│   - Constant CPU Buffer (핫 노드 pin), Window Buffering, CPU-backing│
└───────────────────────────────┬─────────────────────────────────────┘
                                │ pybind11 호출
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ gids_module/gids_nvme.cu (호스트)                                    │
│   GIDS_Controllers  : BaM Controller 객체 생성/관리 (libnvm* 디바이스)│
│   BAM_Feature_Store : page_cache_t, range_t, array_t 인스턴스화     │
│   read_feature*()   : CUDA 커널 런칭                                 │
│   store_tensor()    : 특성 데이터를 SSD에 쓰기                      │
│   PYBIND11_MODULE   : BAM_Feature_Store_float / _long 바인딩        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ __global__ launch
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ gids_module/gids_kernel.cu (GPU device code)                         │
│   read_feature_kernel                    (BaM 단독 경로)            │
│   read_feature_kernel_with_cpu_backing   (CPU 버퍼 hybrid 경로)     │
│   set_cpu_buffer_kernel / _data_kernel   (핫 노드 매핑 설정)        │
│   set_window_buffering_kernel            (윈도우 프리패치)          │
│   write_feature_kernel / _kernel2        (특성 쓰기)                │
│   seq_read_kernel                        (디버그용 시퀀셜 리드)     │
│   - 각 warp가 하나의 node index를 담당 → bam_ptr.read()로 SSD 접근  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ BaM API (bam_ptr, array_d_t, page_cache_d_t)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ bam/ 서브모듈 (주석 작업 제외)                                       │
│   - page_cache, range_t, array_t, Controller, nvm_queue, doorbell…   │
│   - GPU-resident SQ/CQ, libnvm 커널 모듈(/dev/libnvm*)               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ PCIe BAR0 (doorbell), DMA (SQE/CQE/PRP)
                                ▼
                      NVMe SSD(들) (스트라이핑 가능)
```

## 3. 핵심 자료구조 관계도

```
GIDS (Python, GIDS_Setup/GIDS/GIDS.py)
 ├─ BAM_Feature_Store.GIDS_Controllers   ── libnvm* 디바이스를 묶어 Controller 벡터 보유
 └─ BAM_Feature_Store.BAM_Feature_Store_{float,long}
      │
      ▼ (init_controllers)
BAM_Feature_Store<TYPE>  (gids_module/include/bam_nvme.h)
 ├─ std::vector<Controller*>  ctrls        — BaM Controller (SSD당 1개)
 ├─ page_cache_t*              h_pc        — GPU-resident 페이지 캐시 (BaM)
 ├─ range_t<TYPE>*             h_range     — 논리 주소 ↔ SSD offset 매핑
 ├─ array_t<TYPE>*             a           — GPU 커널이 보는 어레이 뷰 (array_d_t)
 ├─ GIDS_CPU_buffer<TYPE>      CPU_buffer  — Constant CPU Buffer (핫 노드)
 ├─ uint64_t*                  offset_array — 샘플링 오프셋 3-tuple
 ├─ bool cpu_buffer_flag, seq_flag         — 경로 선택 플래그
 └─ unsigned int* d_cpu_access             — CPU 버퍼 히트 카운터 (device)

range_t  ── get_cpu_offset(row) / set_cpu_buffer() 로 node → CPU 버퍼 슬롯 매핑
array_t  ── bam_ptr 생성 시 내부 참조됨. read()가 page_cache 경유 SSD fetch
```

## 4. BaM 연결성 (본 프로젝트 학습 목표)

GIDS가 BaM을 어떻게 쓰는지를 주석에서 반드시 명시한다.

- **Controller 생성**: `GIDS_Controllers::init_GIDS_controllers` 에서 `/dev/libnvm{0..5}` 중 `ssd_list`로 선택된 디바이스마다 `new Controller(...)` — BaM의 `Controller` 생성자가 admin queue 생성, I/O queue 개수(=`numQueues`) 및 큐 깊이(`queueDepth`) 만큼 GPU-resident SQ/CQ 할당.
- **페이지 캐시**: `page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0], 64, ctrls)` — BaM의 GPU 페이지 캐시. n_pages = `cache_size_MB × 1024 × 1024 / page_size`.
- **주소 매핑**: `range_t<TYPE>(start=0, n_elems, read_off, n_pages, 0, page_size, h_pc, dev, STRIPE)` — STRIPE 모드이면 다중 SSD에 페이지 단위 스트라이핑.
- **GPU 측 read**: `bam_ptr<T> ptr(dr)` → `ptr.read(logical_idx)` 가 BaM 페이지 캐시 hit/miss 판단 및 miss 시 GPU에서 NVMe SQE 제출, CQE polling 수행.
- **CPU 버퍼 hybrid**: GIDS 고유 최적화. `range_t::get_cpu_offset`로 "이 node가 CPU 버퍼에 올려져 있는가"를 1비트로 판정, hit면 `cudaHostAlloc(cudaHostAllocMapped)` 기반 영역에서 직접 읽어 SSD 접근 회피.
- **Window Buffering**: 다음 배치에 올 예정인 노드들을 미리 페이지 캐시에 끌어올리는 프리패치(`set_window_buffering_kernel`).

주석 작성 시 각 단계가 **BaM의 어떤 API를 부르는지** 함께 쓴다 (예: "`ptr.read()` 는 BaM `array_d_t<T>::seq_read` → `page_cache_d_t::acquire_page` → miss 시 `nvm_queue.sq_enqueue`").

## 5. 프로젝트 특화 주석 요구사항

- **GNN 도메인 용어**: 처음 등장 시 풀어 설명.
  - `DGL` = Deep Graph Library, `GNN` = Graph Neural Network, `CSC` = Compressed Sparse Column, `MFG`/`block` = Message Flow Graph (DGL mini-batch), `heterograph` = 이종 그래프(노드 타입 여러 개), `feature`/`node embedding`, `sampler`(NeighborSampler 등), `IGB`/`IGBH`/`OGB`/`MAG` = 벤치마크 데이터셋.
- **GIDS 고유 용어**:
  - `Constant CPU Buffer` = 학습 전 기간 동안 host-pinned memory에 고정하는 "자주 쓰이는 노드 feature" 버퍼.
  - `Window Buffering` = 다음 n배치에서 필요할 것으로 예측되는 노드의 페이지를 선행 프리패치.
  - `heterograph_map` = 이종 그래프에서 node type별 SSD 내 offset(bytes) 매핑.
  - `ssd_list` = striping 순서를 지정하는 `/dev/libnvm*` 인덱스 리스트.
- **CUDA 커널 주석**에서는 반드시:
  - block/warp ↔ node index 매핑을 적는다 (한 warp = 한 node, warp 내 32 스레드가 feature dim 차원 분할 처리).
  - `bam_ptr.read()`가 어떤 BaM 경로로 확장되는지(히트 시 캐시, 미스 시 NVMe 제출).
  - `atomicAdd(d_cpu_access, 1)`처럼 통계용 경쟁 연산은 "통계 목적이므로 순서 보장 불필요" 같은 문맥을 명시.
- **Python 측 pybind 호출** 주석: 타입 디스패치(`BAM_Feature_Store_float` vs `_long`)와 C++ 템플릿 `BAM_Feature_Store<float>` / `<int64_t>` 대응 관계를 1회 이상 언급.

## 6. 디테일 수준 가이드 (공통 표의 프로젝트 구체화)

| 경로 | 유형 | 상단 블록 | 함수 주석 | 인라인 | 구조체 필드 |
|---|---|---|---|---|---|
| `gids_module/gids_nvme.cu` | 핵심 호스트 구현 | 매우 상세 | 모든 | 모든 라인 | 모든 필드 |
| `gids_module/gids_kernel.cu` | 핵심 GPU 커널 | 매우 상세 | 모든 | 모든 라인 | — |
| `gids_module/include/bam_nvme.h` | 핵심 헤더 | 매우 상세 | 모든 선언 | 해당 없음 | 모든 필드 |
| `gids_module/include/example.h` | 예제 헤더 | 상세 | 모든 | — | 모든 필드 |
| `gids_module/include/page_cache_backup.h` | BaM 백업본 | **간결** (상단 블록 + 섹션 경계 주석만, "bam/page_cache.h의 백업 사본"임을 명시) | 주요만 | 핵심만 | — |
| `gids_module/CMakeLists.txt` | 빌드 | 상세 | — | 모든 라인 | — |
| `gids_module/BAM_Feature_Store/*`, `example/*` | pybind 패키징 | 상세 | 모든 | 모든 라인 | — |
| `GIDS_Setup/GIDS/GIDS.py` | Python 메인 API | 매우 상세 | 모든 | 모든 라인 | 모든 필드 |
| `GIDS_Setup/GIDS/__init__.py`, `test.py`, `setup.py` | 패키징/테스트 | 상세 | 모든 | 모든 라인 | — |
| `evaluation/dataloader.py` | BaM baseline loader | 매우 상세 | 모든 | 모든 라인 | 모든 필드 |
| `evaluation/homogenous_train*.py` / `heterogeneous_train*.py` / `ClusterGCN` | 학습 스크립트 | 매우 상세 | 모든 | 모든 라인 | — |
| `evaluation/models.py`, `mlperf_model.py` | GNN 모델 정의 | 상세 | 모든 | 모든 라인 | — |
| `evaluation/ladies_sampler.py` | 커스텀 샘플러 | 상세 | 모든 | 모든 라인 | — |
| `evaluation/page_rank_node_list_gen.py` | 핫 노드 리스트 생성 | 상세 | 모든 | 모든 라인 | — |
| `evaluation/tensor_write.py` | 특성 → SSD 쓰기 | 상세 | 모든 | 모든 라인 | — |
| `evaluation/lock_mem.cpp` | 메모리 락 유틸 | 상세 | 모든 | 모든 라인 | — |
| `evaluation/*.sh` | 실행 스크립트 | 상세 | — | 모든 라인 | — |
| `evaluation/GIDS_unit_test.py`, `gids_unit_test.sh` | 단위 테스트 | 상세 | 모든 | 모든 라인 | — |

## 7. 작업 제외 대상

- **`bam/` 서브모듈 전체** — 이미 파악되어 있다는 전제. 주석 작업 금지.
- **`GIDS_Setup/build/`, `GIDS_Setup/dist/`** — 빌드 산출물. 작업 제외.
- **`.git/`, `head.html`, `README.md`, `.gitmodules`, `.gitignore`** — 문서/메타. 작업 제외.

## 8. 주석 작업 진행 현황

| 디렉토리 | 파일 수 | 완료 | 비고 |
|---|---|---|---|
| `gids_module/` (C++/CUDA + pybind 패키지) | 10 | ● | gids_nvme.cu ●, CMakeLists.txt ●, gids_kernel.cu ●, include/bam_nvme.h ●, include/example.h ●, include/page_cache_backup.h ●(간결), BAM_Feature_Store/__init__.py ●, BAM_Feature_Store/setup.py ●, example/__init__.py ●, example/setup.py ● |
| `GIDS_Setup/GIDS/` | 4 | ● | GIDS.py ●, __init__.py ●, test.py ●, setup.py ● |
| `evaluation/` Python | 14 | ● | dataloader.py ●, heterogeneous_train.py ●, heterogeneous_train_baseline.py ●, homogenous_train.py ●, homogenous_train_baseline.py ●, homogenous_train_ClusterGCN.py ●, GIDS_unit_test.py ●, models.py ●, mlperf_model.py ●, ladies_sampler.py ●, page_rank_node_list_gen.py ●, tensor_write.py ● |
| `evaluation/` Shell+CPP | 7 | ● | lock_mem.cpp ●, gids_unit_test.sh ●, run_base_IGBH.sh ●, run_GIDS_IGBH.sh ●, run_BaM_IGBH.sh ●, write_data.sh ●, write_data_full.sh ● |

범례: ● 신기준 완료 / ◐ 일부 완료 / ⬜ 미진행.
신기준 완료 파일(28): `GIDS_Setup/GIDS/GIDS.py`, `GIDS_Setup/GIDS/__init__.py`, `GIDS_Setup/GIDS/test.py`, `GIDS_Setup/setup.py`, `gids_module/gids_nvme.cu`, `gids_module/gids_kernel.cu`, `gids_module/include/bam_nvme.h`, `gids_module/include/example.h`, `gids_module/CMakeLists.txt`, `gids_module/BAM_Feature_Store/__init__.py`, `gids_module/BAM_Feature_Store/setup.py`, `gids_module/example/__init__.py`, `gids_module/example/setup.py`, `evaluation/dataloader.py`, `evaluation/heterogeneous_train.py`, `evaluation/heterogeneous_train_baseline.py`, `evaluation/homogenous_train.py`, `evaluation/homogenous_train_baseline.py`, `evaluation/homogenous_train_ClusterGCN.py`, `evaluation/GIDS_unit_test.py`, `evaluation/models.py`, `evaluation/mlperf_model.py`, `evaluation/ladies_sampler.py`, `evaluation/page_rank_node_list_gen.py`, `evaluation/tensor_write.py`, `evaluation/lock_mem.cpp`, `evaluation/gids_unit_test.sh`, `evaluation/run_base_IGBH.sh`, `evaluation/run_GIDS_IGBH.sh`, `evaluation/run_BaM_IGBH.sh`, `evaluation/write_data.sh`, `evaluation/write_data_full.sh`. 간결 처리(1): `gids_module/include/page_cache_backup.h`.

## 9. 빌드 참고

```bash
# 1) bam 먼저 빌드 (bam/README 참조 — libnvm.so + libnvm kernel module)
# 2) gids_module 빌드
cd gids_module && mkdir -p build && cd build
cmake .. && make -j
cd BAM_Feature_Store && python setup.py install
# 3) Python 인터페이스 설치
cd ../../../GIDS_Setup && pip install .
# 4) 데이터 사전 기록
#    - bam/benchmarks/readwrite_stripe (권장) 또는 evaluation/tensor_write.py
# 5) 학습 실행
cd evaluation && ./run_GIDS_IGBH.sh
```

## 10. 커밋 메시지

```
Add Korean annotations to <대상>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```
