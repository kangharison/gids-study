/*
 * [한국어] GIDS 호스트 측 핵심 구조체 선언 헤더 (bam_nvme.h)
 *
 * === 파일의 역할 ===
 * GIDS(GPU-Initiated Direct Storage) 의 호스트 측 핵심 클래스 3종을 선언한다:
 *   (1) GIDS_Controllers       — BaM Controller 여러 개를 SSD 리스트로 묶어 관리.
 *   (2) GIDS_CPU_buffer<TYPE>   — Constant CPU Buffer(핫 노드 feature) 를 위한 host-pinned 메모리 디스크립터.
 *   (3) BAM_Feature_Store<TYPE> — BaM 페이지 캐시/range/array 인스턴스를 소유하고
 *                                 read_feature* / store_tensor 등 파이썬에 노출되는 상위 API 를 제공.
 * 선언만 있고 정의는 gids_nvme.cu 에 있다. 본 헤더는 BaM(`bam/include/*`) 헤더들을
 * 먼저 include 해 Controller/page_cache_t/range_t/array_t 등의 전방 선언을 확보한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 전체 흐름:
 *   Python 학습 스크립트 → GIDS_DGLDataLoader(Python) → pybind11 → 이 헤더가 정의한 C++ 구조체
 *   → BaM page_cache → NVMe SSD.
 * 호출자: gids_nvme.cu 의 템플릿 정의, 그리고 pybind11 바인딩(PYBIND11_MODULE).
 * 피호출자: BaM 의 Controller, page_cache_t, range_t, array_t. 실행 컨텍스트는
 * 호스트 유저스페이스 C++이며, 멤버 포인터 일부(a->d_array_ptr, d_range 등)는
 * GPU 디바이스 메모리를 가리켜 커널 인자로 전달된다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존(아래 include): libnvm 사용자 헤더(nvm_admin/cmd/ctrl/error/io/queue/types/util),
 *   BaM C++ 상위 래퍼(ctrl.h, event.h, page_cache.h, queue.h, buffer.h, util.h),
 *   시스템 헤더(cuda.h, fcntl.h, sys/mman.h, unistd.h, stdio.h).
 * - 사용처: gids_nvme.cu (모든 멤버 정의), gids_kernel.cu (GIDS_CPU_buffer 를 커널
 *   파라미터로 수신), evaluation/Python 코드(pybind11 경유).
 * - 공유 상태: std::vector<Controller*> ctrls 는 GIDS_Controllers 에서 생성되어
 *   BAM_Feature_Store::init_controllers 호출 시 멤버로 복사 공유된다.
 *
 * === 주요 함수/구조체 요약 ===
 * - GIDS_Controllers::init_GIDS_controllers : /dev/libnvm* 디바이스를 선택해 BaM Controller 벡터 구성.
 * - BAM_Feature_Store::init_controllers     : page_cache_t + range_t + array_t 생성(STRIPE 모드).
 * - BAM_Feature_Store::read_feature*        : mini-batch 노드 ID 들을 받아 GPU 커널 런칭.
 * - BAM_Feature_Store::store_tensor         : 특성 텐서를 SSD로 기록(write_feature_kernel2 실행).
 * - BAM_Feature_Store::cpu_backing_buffer / set_cpu_buffer : Constant CPU Buffer 최적화 설정.
 * - BAM_Feature_Store::set_window_buffering : 다음 배치 프리패치(window buffering).
 */

#ifndef BAMNVME_H
#define BAMNVME_H

/* [한국어] libnvm 사용자 헬퍼: 호스트 측 DMA 버퍼(`nvm_dma_t`) 래핑, pinned memory 매핑. */
#include <buffer.h>
/* [한국어] CUDA Driver API(cuCtx, cuMemAlloc) 진입점. BaM 이 GPU 메모리를 직접 매핑할 때 필요. */
#include <cuda.h>
/* [한국어] /dev/libnvm* 장치를 open 하기 위한 O_RDWR 플래그 등 POSIX 파일 제어. */
#include <fcntl.h>
/* [한국어] NVMe admin 큐 명령(Set/Get Features, Identify Controller). Controller 초기화 경로. */
#include <nvm_admin.h>
/* [한국어] NVMe I/O 커맨드 빌더(nvm_cmd_rw_blks 등). SQE 직접 조립 시 사용. */
#include <nvm_cmd.h>
/* [한국어] Controller 저수준 핸들(nvm_ctrl_t). BAR0 MMIO 매핑 정보 포함. */
#include <nvm_ctrl.h>
/* [한국어] libnvm 에러 코드 → 문자열 변환 헬퍼. 초기화 실패 로그에 사용. */
#include <nvm_error.h>
/* [한국어] 동기 블록 I/O 인터페이스(nvm_write/read). 호스트 측 벤치마크 경로 전용. */
#include <nvm_io.h>
/* [한국어] GPU-resident 큐 관리 구조. page_cache 가 내부에서 이 큐로 SQE 를 enqueue. */
#include <nvm_parallel_queue.h>
/* [한국어] 단일 호스트 큐 구조(nvm_queue_t). admin 경로에서 주로 사용. */
#include <nvm_queue.h>
/* [한국어] NVMe 기본 타입(sqe/cqe, nvm_aq_ref 등) 및 상수(오피코드) 정의. */
#include <nvm_types.h>
/* [한국어] DMA 주소 계산/정렬, LBA ↔ 바이트 변환 매크로. */
#include <nvm_util.h>
/* [한국어] printf/stderr 진단 로그 용도. */
#include <stdio.h>
/* [한국어] mlock/mmap — host-pinned buffer 를 잠그거나 /dev/libnvm* 를 mmap 할 때. */
#include <sys/mman.h>
/* [한국어] close(), read(), sysconf 등 POSIX 기본. */
#include <unistd.h>
/* [한국어] BaM 공용 유틸(cuda_err_chk 등 매크로). */
#include <util.h>

/* [한국어] BaM C++ Controller 상위 래퍼 — 생성자에서 admin queue 생성, I/O 큐 할당을 수행. */
#include <ctrl.h>
/* [한국어] BaM 이벤트/통계(print_reset_stats 등) — Controller/page_cache 에 구현 주입. */
#include <event.h>
/* [한국어] GIDS 의 핵심: GPU 페이지 캐시(page_cache_t), range_t, array_t, bam_ptr 정의. */
#include <page_cache.h>
/* [한국어] BaM 큐 C++ 래퍼(QueuePair 등). ctrl.h 가 내부적으로 사용. */
#include <queue.h>

//#define TYPE float
/* [한국어] (위 주석 원본 유지) — 과거엔 전역 TYPE 매크로로 feature 타입을 고정했으나,
 * 현재는 BAM_Feature_Store<TYPE> 템플릿으로 float/int64_t 를 모두 지원한다. */

/*
 * [한국어]
 * struct GIDS_Controllers - BaM Controller 객체들의 소유자/팩토리.
 *
 * 파이썬 쪽 GIDS_Controllers 바인딩에 1:1 대응한다. 여러 /dev/libnvm* 디바이스를
 * 묶어 하나의 벡터로 관리하고, BAM_Feature_Store::init_controllers 가 이 벡터를
 * 받아 자신의 ctrls 멤버로 복사한다. 실행 컨텍스트는 호스트 유저스페이스.
 * 생성/초기화는 단일 Python 스레드에서 수행(동기화 불필요).
 */
struct GIDS_Controllers {
  const char *const ctrls_paths[6] = {"/dev/libnvm0","/dev/libnvm1","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4","/dev/libnvm5"};
  /* [한국어] 선택 가능한 libnvm 캐릭터 디바이스 경로 테이블(최대 6개).
   * 설정자: 컴파일 시 고정 상수로 초기화. 런타임 변경 불가(const).
   * 읽는 자: init_GIDS_controllers 가 ssd_list[i] 로 인덱싱해 Controller 생성자에 전달.
   * 값 범위: 정확히 6개의 경로, 인덱스 0..5. ssd_list 가 이 범위를 넘으면 UB.
   * 동기화: 읽기 전용이므로 락 불필요. */

  std::vector<Controller *> ctrls;
  /* [한국어] BaM Controller(호스트 C++ 래퍼) 포인터 벡터. 각 원소는 /dev/libnvm{i} 하나에 대응.
   * 설정자: init_GIDS_controllers 가 new Controller(...) 결과를 push_back.
   * 읽는 자: BAM_Feature_Store::init_controllers 가 ctrls = GIDS_ctrl.ctrls 로 공유 복사.
   *         gids_kernel.cu 의 write_feature_kernel 이 Controller** ctrls 형태로 GPU 전달.
   * 값 범위: size == 초기화 시점의 n_ctrls. 각 원소는 생성 성공한 Controller* (NULL 아님).
   * 동기화: 초기화 후에는 거의 read-only. 내부 queue 는 BaM 쪽 락/GPU atomic 으로 보호. */

  uint32_t n_ctrls = 1;
  /* [한국어] 실제 생성된 Controller 수(ssd_list 길이와 동일).
   * 설정자: init_GIDS_controllers 가 num_ctrls 로 갱신.
   * 읽는 자: BAM_Feature_Store::init_controllers 에서 ctrls 루프 횟수로 사용.
   * 값 범위: 1..6 (ctrls_paths 길이 상한).
   * 동기화: 초기화 한 번, 이후 읽기 전용. */

  uint64_t queueDepth = 1024;
  /* [한국어] 각 Controller 에 할당할 NVMe I/O 큐(SQ/CQ)의 엔트리 개수.
   * 설정자: init_GIDS_controllers 의 q_depth 인자로 덮어씀.
   * 읽는 자: Controller 생성자(BaM) 가 GPU-resident SQ/CQ 메모리 크기 계산에 사용.
   * 값 범위: 1..장치 MQES+1. 기본 1024는 대부분 NVMe SSD 에서 유효.
   * 동기화: 초기화 후 불변. */

  uint64_t numQueues = 128;
  /* [한국어] Controller 당 생성할 I/O 큐 쌍(SQ/CQ) 개수. 많을수록 워프 간 큐 경합이 줄어듦.
   * 설정자: init_GIDS_controllers 의 num_q 인자.
   * 읽는 자: Controller 생성자가 큐 메모리 할당 및 doorbell 오프셋 계산에 사용.
   * 값 범위: 1..장치 지원 최대 I/O queue 수. 128은 엔터프라이즈 NVMe 일반값.
   * 동기화: 초기화 후 불변. */

  uint32_t cudaDevice = 0;
  /* [한국어] BaM Controller 가 바인딩될 CUDA 디바이스 인덱스(GPU #).
   * 설정자: 기본 0, init_GIDS_controllers 는 변경하지 않음(Python 측에서 사전 설정 필요 시 수정).
   * 읽는 자: Controller 생성자 — GPU BAR 매핑/pinned DMA 에 사용.
   * 값 범위: 0..cudaDeviceCount-1.
   * 동기화: 초기화 후 불변. */

  uint32_t nvmNamespace = 1;
  /* [한국어] NVMe namespace ID (NSID). 대부분 단일 네임스페이스 SSD 는 1.
   * 설정자: 기본 1. 멀티 NS 장치에서 다른 NS 를 쓰려면 수정 필요.
   * 읽는 자: Controller 생성자가 admin Identify NS 명령 발행 시 사용.
   * 값 범위: 1..FFFFFFFEh(유효 NSID). 0/FFFFFFFFh 는 예약.
   * 동기화: 초기화 후 불변. */

  //member functions
  /*
   * [한국어]
   * init_GIDS_controllers - Controller 벡터를 생성해 ctrls 를 채운다.
   *
   * @param num_ctrls: 사용할 SSD 수. n_ctrls 로 저장되고 루프 횟수가 됨.
   * @param q_depth:   각 큐의 엔트리 수(queueDepth 로 저장).
   * @param num_q:     큐 쌍 개수(numQueues 로 저장).
   * @param ssd_list:  실제 사용할 /dev/libnvm* 인덱스 목록. 길이 == num_ctrls 이어야 함.
   *
   * Python GIDS 래퍼의 GIDS_Controllers().init_GIDS_controllers(...) 로 호출됨.
   * 내부에서 BaM의 Controller 생성자를 각 SSD 마다 호출하여 admin queue → I/O 큐
   * 할당까지 수행한다. 에러 발생 시 BaM 예외가 전파되어 Python RuntimeError 로 승격.
   *
   * 호출 체인:
   *   Python GIDS.__init__ → [이 함수] → BaM Controller(ctor) → libnvm ioctl → 커널 모듈
   */
  void init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q,  const std::vector<int>& ssd_list);

};

/*
 * [한국어]
 * struct GIDS_CPU_buffer<TYPE> - Constant CPU Buffer(핫 노드 feature) 디스크립터.
 *
 * GIDS 최적화 "Constant CPU Buffer" 에서, 자주 쓰이는 노드의 feature 를 학습
 * 전 기간 동안 host-pinned 메모리에 고정한다. cudaHostAlloc(cudaHostAllocMapped)
 * 로 host 주소(cpu_buffer) 와 device 주소(device_cpu_buffer) 두 뷰를 동시에 확보해,
 * GPU 커널이 PCIe 를 통해 직접 읽게 함으로써 SSD 접근 자체를 회피한다.
 * 단일 BAM_Feature_Store 인스턴스가 단독 소유.
 */
template <typename TYPE>
struct GIDS_CPU_buffer {
    TYPE* cpu_buffer;
    /* [한국어] 호스트 가상 주소로 본 버퍼 포인터(cudaHostAlloc 반환값).
     * 설정자: BAM_Feature_Store::cpu_backing_buffer 가 cudaHostAlloc 결과로 할당.
     * 읽는 자: 현재 호스트 쪽 직접 사용 경로는 없음(디버그/확장 여지).
     * 값 범위: 유효 페이지잠금 호스트 포인터 또는 nullptr(미초기화 상태).
     * 동기화: 초기화 이후 read-only. 내부 데이터는 Python 측에서 CPU memcpy 로 채움. */

    TYPE* device_cpu_buffer;
    /* [한국어] 동일 버퍼의 GPU 디바이스 가상 주소(cudaHostGetDevicePointer 반환값).
     * 설정자: cpu_backing_buffer 가 cudaHostGetDevicePointer 로 획득.
     * 읽는 자: gids_kernel.cu 의 read_feature_kernel_with_cpu_backing_memory 가
     *         cpu_off 가 hit 일 때 이 포인터로 feature 를 읽음.
     * 값 범위: host 버퍼와 매핑된 device-visible 주소. kernel arg 로 전달.
     * 동기화: 쓰기 경로가 set_cpu_buffer_data_kernel(일회성) 이후 읽기만. 동시 쓰기 없음. */

    uint64_t cpu_buffer_dim;
    /* [한국어] 버퍼에 저장되는 각 노드 feature 벡터의 차원(= feature_size / sizeof(TYPE)).
     * 설정자: cpu_backing_buffer(dim, len) 가 인자 dim 으로 설정.
     * 읽는 자: 커널 쪽에서는 인자로 별도 전달되는 cache_dim 이 사실상 동일 역할.
     * 값 범위: >0. 보통 feature 차원 (e.g., 128, 1024).
     * 동기화: 초기화 후 불변. */

    uint64_t cpu_buffer_len;
    /* [한국어] 버퍼에 보유 가능한 노드 수(row 수). 총 바이트 = len * dim * sizeof(TYPE).
     * 설정자: cpu_backing_buffer(dim, len) 가 인자 len 으로 설정.
     * 읽는 자: read_feature_kernel_with_cpu_backing_memory 의 seq_flag 경로에서
     *         row_index < cpu_buffer_len 히트 판정에 직접 사용.
     * 값 범위: 0..메모리 한도. 통상 핫 노드 수(수십만~수백만).
     * 동기화: 초기화 후 불변. */
};


/*
 * [한국어]
 * struct BAM_Feature_Store<TYPE> - GIDS 의 주 특성 저장/조회 엔진.
 *
 * 하나의 BAM_Feature_Store 인스턴스는 BaM 페이지 캐시(page_cache_t) 하나,
 * 주소 매핑(range_t<TYPE>) 하나, 그리고 커널이 보는 array_t<TYPE> 하나를 소유한다.
 * Python 쪽 BAM_Feature_Store_float / BAM_Feature_Store_long 두 클래스가 각각
 * TYPE=float, TYPE=int64_t 로 pybind11 바인딩된다 (PYBIND11_MODULE 참조).
 *
 * 데이터 흐름 요약:
 *   Python(GIDS.DataLoader) → read_feature(node_ids) → read_feature_kernel
 *   → bam_ptr.read() → page_cache hit 시 GPU memcpy / miss 시 nvm_queue SQ enqueue
 *   → NVMe SSD → 완료 CQE 폴링 → 데이터 반환.
 * CPU 버퍼 hybrid 경로: cpu_buffer_flag=true 면 read_feature_kernel_with_cpu_backing_memory
 * 로 분기, range_t::get_cpu_offset 결과의 최하위 비트로 CPU 캐시 hit 판정.
 */
template <typename TYPE>
struct BAM_Feature_Store {


  GIDS_CPU_buffer<TYPE> CPU_buffer;
  /* [한국어] 이 스토어가 쓰는 Constant CPU Buffer 디스크립터.
   * 설정자: cpu_backing_buffer() 가 dim/len/포인터 필드를 채움.
   * 읽는 자: 모든 read_feature_kernel_with_cpu_backing_memory 호출이 값으로 전달받음.
   * 값 범위: GIDS_CPU_buffer<TYPE> 의 초기화 상태. 초기 상태는 모든 필드 0/NULL.
   * 동기화: 초기화 1회 후 커널에서는 읽기만. */

  //GIDS optimization flasg
  bool cpu_buffer_flag = false;
  /* [한국어] Constant CPU Buffer 최적화 활성 여부.
   * 설정자: cpu_backing_buffer() 호출 시 true 로 전환.
   * 읽는 자: read_feature/read_feature_hetero 가 커널 분기 선택에 사용.
   * 값 범위: true/false.
   * 동기화: Python 메인 스레드에서만 변경, 동시성 없음. */

  bool seq_flag = true;
  /* [한국어] CPU 버퍼 인덱싱을 "row_index 범위(true)"로 할지 "get_cpu_offset 비트(false)"로 할지 결정.
   * 설정자: 초기 true. set_cpu_buffer() 호출 후 false 로 전환 → 부분 mapping 모드.
   * 읽는 자: read_feature_kernel_with_cpu_backing_memory 내부 분기.
   * 값 범위: true/false.
   * 동기화: 단일 스레드 설정. */

  //Sampling Offsets
  uint64_t* offset_array;
  /* [한국어] DGL sampler ↔ GIDS 간 공유되는 3-튜플 오프셋 (input/index/data).
   * 설정자: set_offsets() 가 new uint64_t[3] 할당 후 값 저장.
   * 읽는 자: get_offset_array() 가 포인터를 uint64_t 로 Python 에 반환, Python 쪽 cffi 가 tensor 로 해석.
   * 값 범위: 각 원소는 바이트 오프셋(>=0).
   * 동기화: 초기화 후 값 고정, Python 단일 스레드. */

  int dim;
  /* [한국어] feature 차원 = pageSize / sizeof(TYPE) (이번 스토어에서 한 페이지에 들어가는 원소 수).
   * 설정자: init_controllers 가 pageSize 기반으로 설정.
   * 읽는 자: store_tensor 에서 write_feature_kernel2 차원 인자로 전달.
   * 값 범위: >0. 기본 4096/sizeof(float)=1024.
   * 동기화: 초기화 후 불변. */

  uint64_t total_access;
  /* [한국어] 누적된 노드 feature 요청 수(통계 용도).
   * 설정자: read_feature* 계열이 num_index 만큼 증가.
   * 읽는 자: print_stats() 가 출력 후 0 으로 리셋.
   * 값 범위: 0..2^64-1.
   * 동기화: 단일 Python 스레드 호출만 가정, atomic 불필요. */

  unsigned int cpu_access_count = 0;
  /* [한국어] 최근 커널에서 CPU 버퍼 히트 수를 받아둔 호스트 사본.
   * 설정자: read_feature 계열이 cudaMemcpy(d_cpu_access → cpu_access_count).
   * 읽는 자: get_cpu_access_count() 를 통해 Python 이 통계 조회.
   * 값 범위: 0..해당 배치의 워프 수.
   * 동기화: 단일 스레드 호스트 값. */

  unsigned int* d_cpu_access;
  /* [한국어] CPU 버퍼 히트 카운터의 디바이스 메모리 사본.
   * 설정자: init_controllers 가 cudaMalloc+cudaMemset 으로 준비.
   * 읽는 자: CPU-backed 커널이 atomicAdd 로 갱신, 호스트가 cudaMemcpy 로 가져감.
   * 값 범위: GPU 메모리의 unsigned int 포인터.
   * 동기화: GPU atomicAdd 로 스레드 경합 해소. 통계 목적이므로 순서 불필요. */

  //BAM parameters
  uint32_t cudaDevice = 0;
  /* [한국어] 페이지 캐시가 상주하는 GPU 인덱스.
   * 설정자: 기본 0. Python 다중 GPU 환경에선 사전에 세팅 필요.
   * 읽는 자: page_cache_t/range_t 생성자에 전달.
   * 값 범위: 0..cudaDeviceCount-1.
   * 동기화: 초기화 후 불변. */

  size_t numPages = 262144 * 8;
  /* [한국어] 페이지 캐시가 보유하는 페이지 수. init_controllers 에서 cache_size_MB 로 재계산.
   * 설정자: init_controllers 가 cache_size * 1MB / pageSize 로 갱신.
   * 읽는 자: page_cache_t 생성자, print_stats.
   * 값 범위: >0. 기본값 2,097,152.
   * 동기화: 초기화 후 불변. */

  bool stats = false;
  /* [한국어] 상세 통계 출력 여부 플래그(현재 소스에서는 print_stats 분기에만 흔적).
   * 설정자: 기본 false, 외부에서 직접 수정 가능.
   * 읽는 자: 현재 코드에서 직접 사용 X(확장 여지).
   * 값 범위: true/false. */

  size_t numThreads = 64;
  /* [한국어] (예약) BaM 이 호스트 측 작업자 스레드를 쓸 경우 개수. 현 GIDS 경로에선 미사용.
   * 설정자: 기본 64.
   * 읽는 자: 없음(문서화 용). */

  uint32_t domain = 0;
  /* [한국어] (예약) PCIe BDF(Bus/Device/Function) 중 domain. 현재 미사용.
   * 설정자: 기본 0. 읽는 자: 없음. */

  uint32_t bus = 0;
  /* [한국어] (예약) PCIe bus 번호. 다중 SSD 식별용 예약 필드. 값 범위: 0..255. */

  uint32_t devfn = 0;
  /* [한국어] (예약) PCIe device/function 합성 번호. 현재 미사용. 값 범위: 0..255. */

  uint32_t n_ctrls = 1;
  /* [한국어] 이 스토어가 사용하는 SSD 개수.
   * 설정자: init_controllers 가 num_ssd 로 갱신.
   * 읽는 자: print_stats, store_tensor(write_feature_kernel2 의 num_ctrls 인자).
   * 값 범위: 1..GIDS_Controllers::ctrls.size(). */

  size_t blkSize = 128;
  /* [한국어] CUDA 블록당 스레드 수. read_feature 커널 런칭에서 block dim 으로 사용.
   * 설정자: 기본 128 (= 4 워프).
   * 읽는 자: read_feature* 의 b_size.
   * 값 범위: 32 의 배수, 보통 128/256/512. */

  size_t queueDepth = 1024;
  /* [한국어] (문서화) 이 스토어가 전제하는 NVMe 큐 깊이. 실제로는 GIDS_Controllers 쪽이 권위.
   * 값 범위: 장치 MQES+1 이하. */

  size_t numQueues = 128;
  /* [한국어] (문서화) 전제 큐 개수. 실제로는 GIDS_Controllers 쪽이 권위. */

  uint32_t pageSize = 4096 ;
  /* [한국어] 페이지 캐시의 페이지 크기(바이트). feature 하나의 크기와 정렬되어야 한다.
   * 설정자: init_controllers 의 ps 인자.
   * 읽는 자: page_cache_t/range_t 생성자 및 dim 계산(pageSize/sizeof(TYPE)).
   * 값 범위: 512, 4096, 8192 등 NVMe LBA 와 정렬되는 값. */

  uint64_t numElems = 300LL*1000*1000*1024;
  /* [한국어] 전체 feature 텐서의 원소(TYPE 단위) 수. range 의 끝 지점.
   * 설정자: init_controllers 의 num_ele.
   * 읽는 자: range_t 생성자, array_t 생성자.
   * 값 범위: >0. 기본 ~3.07e11 (수백 GB). */

  uint64_t read_offset = 0;
  /* [한국어] SSD 시작 오프셋(바이트) — 데이터 영역이 디스크의 0 이 아닌 지점에서 시작할 때.
   * 설정자: init_controllers 의 read_off.
   * 읽는 자: range_t 생성자가 물리 LBA 계산에 사용.
   * 값 범위: 0..SSD 용량. */

  std::vector<Controller *> ctrls;
  /* [한국어] BaM Controller 포인터 벡터 (GIDS_Controllers 에서 공유 복사).
   * 설정자: init_controllers 가 GIDS_ctrl.ctrls 를 복사 대입.
   * 읽는 자: page_cache_t 생성자에 전달, print_stats 출력, store_tensor 가 n_ctrls 참조.
   * 값 범위: size == n_ctrls.
   * 동기화: 생성 후 읽기 전용. */

  page_cache_t *h_pc;
  /* [한국어] BaM GPU 페이지 캐시(호스트 핸들).
   * 설정자: init_controllers 가 new page_cache_t(...) 로 생성.
   * 읽는 자: read_feature 커널 경로, store_tensor(write_feature_kernel2), flush_cache.
   *         내부의 d_pc_ptr 가 device-side page_cache_d_t 에 대한 포인터.
   * 값 범위: 유효 포인터(초기화 후) 또는 dangling(소멸 후 — 본 구조체에 소멸자 없음).
   * 동기화: BaM 내부에서 GPU atomic 으로 캐시 라인 보호. */

  range_t<TYPE> *h_range;
  /* [한국어] 논리 인덱스 ↔ SSD offset 매핑 (호스트 핸들). STRIPE 모드로 생성.
   * 설정자: init_controllers 가 new range_t<TYPE>(..., STRIPE) 생성.
   * 읽는 자: vr[0] 에 저장되어 array_t 생성 시 사용. d_range 로 device 핸들 추출.
   * 값 범위: 유효 포인터. numElems/page_size 를 관리.
   * 동기화: 내부 get_cpu_offset/set_cpu_buffer 는 GPU 에서만 호출. */

  std::vector<range_t<TYPE> *> vr;
  /* [한국어] array_t 생성자에 전달되는 range 리스트. GIDS 는 현재 단일 range 만 사용.
   * 설정자: init_controllers 가 vr.push_back(nullptr) 후 vr[0]=h_range.
   * 읽는 자: new array_t<TYPE>(numElems, 0, vr, cudaDevice).
   * 값 범위: size == 1.
   * 동기화: 초기화 후 불변. */

  array_t<TYPE> *a;
  /* [한국어] GPU 커널이 보는 배열 뷰. 내부에 d_array_ptr (array_d_t<TYPE>*) 포함.
   * 설정자: init_controllers 가 new array_t<TYPE>(...) 로 생성.
   * 읽는 자: 모든 read_feature/set_window_buffering/store_tensor 커널이
   *         a->d_array_ptr 를 인자로 전달. get_array_ptr() 가 Python에 uint64_t로 노출.
   * 값 범위: 유효 포인터.
   * 동기화: 내부 bam_ptr 가 atomicCAS 로 페이지 슬롯 획득을 수행. */

  range_d_t<TYPE> *d_range;
  /* [한국어] h_range 의 device-side 뷰 (range_d_t<TYPE>*). 커널이 get_cpu_offset/set_cpu_buffer 호출 시 사용.
   * 설정자: init_controllers 가 (range_d_t<TYPE>*)h_range->d_range_ptr 로 캐스팅.
   * 읽는 자: read_feature_kernel_with_cpu_backing_memory, set_cpu_buffer_kernel.
   * 값 범위: GPU 메모리상의 구조체 포인터.
   * 동기화: 내부 필드 갱신은 set_cpu_buffer 경로에서 일회성, 이후 커널에선 읽기만. */
  //wb


  float kernel_time = 0;
  /* [한국어] read_feature* 실행 누적 시간(ms). 벤치마킹 용.
   * 설정자: 각 read_feature* 말미에 경과 시간 누적.
   * 읽는 자: print_stats 가 출력 후 0 으로 리셋.
   * 값 범위: >=0.
   * 동기화: 단일 Python 스레드 호출만 가정. */


  /*
   * [한국어]
   * init_controllers - BaM 페이지 캐시/range/array 를 생성해 이 스토어를 사용할 수 있게 초기화.
   *
   * @param GIDS_ctrl: 이미 init_GIDS_controllers 로 Controller 벡터가 준비된 구조체(값 복사).
   * @param ps:        페이지 크기(바이트). pageSize 와 dim 계산에 사용.
   * @param r_off:     SSD 시작 오프셋(바이트) = read_offset.
   * @param num_ele:   feature 텐서의 총 원소 수 = numElems.
   * @param cache_size:페이지 캐시 크기(MB). numPages = cache_size*1MB/ps.
   * @param num_ssd:   사용할 SSD 수 = n_ctrls.
   *
   * Python GIDS.DataLoader 설정 단계에서 한 번 호출된다. 실패 시 BaM 생성자 예외가
   * pybind11 을 거쳐 Python RuntimeError 로 승격.
   *
   * 호출 체인:
   *   Python BAM_Feature_Store_X.init_controllers → [이 함수]
   *     → new page_cache_t → new range_t<STRIPE> → new array_t → cudaMalloc(d_cpu_access)
   */
  void init_controllers(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t r_off, uint64_t num_ele, uint64_t cache_size,
                        uint64_t num_ssd);

  /*
   * [한국어]
   * read_feature - 단일 호모지니어스 배치의 노드 feature 를 GPU 로 가져온다.
   *
   * @param tensor_ptr: 결과를 쓸 출력 버퍼(GPU 주소, uint64 로 전달됨 — Python tensor.data_ptr()).
   * @param index_ptr:  노드 ID 배열(int64, GPU 주소). length == num_index.
   * @param num_index:  가져올 노드 개수.
   * @param dim:        노드 feature 차원(element 단위).
   * @param cache_dim:  페이지 내부 "행 크기"(원소 단위). 대부분 dim 과 같음.
   * @param key_off:    인덱스에 더해 줄 오프셋(heterograph 노드 타입별 기준 ID).
   *
   * cpu_buffer_flag 에 따라 read_feature_kernel 또는
   * read_feature_kernel_with_cpu_backing_memory 를 런칭. num_index 에서 grid 차원을 유도
   * (한 warp = 한 노드). 커널 전후 cudaDeviceSynchronize 로 시간 측정.
   *
   * 호출 체인:
   *   Python GIDS_DataLoader._fetch → [read_feature] → read_feature_kernel → bam_ptr.read()
   */
  void read_feature(uint64_t tensor_ptr, uint64_t index_ptr,int64_t num_index, int dim, int cache_dim, uint64_t key_off);

  /*
   * [한국어]
   * read_feature_hetero - 이종 그래프(heterograph) 에서 여러 node type 배치를 동시에 fetch.
   *
   * @param num_iter:        node type 개수.
   * @param i_ptr_list:      각 타입 출력 포인터 리스트.
   * @param i_index_ptr_list:각 타입 노드 ID 포인터.
   * @param num_index:       각 타입 노드 개수.
   * @param dim, cache_dim:  공통 차원.
   * @param key_off:         각 타입의 기본 key offset.
   *
   * 타입마다 별도 CUDA 스트림으로 발행해 오버랩 실행. 모든 스트림 sync 후 타이밍
   * 누적. 실행 컨텍스트: 호스트 유저스페이스, 커널은 num_iter 개 동시 런칭.
   *
   * 호출 체인:
   *   Python heterograph loader → [이 함수] → read_feature_kernel(_with_cpu_backing) × num_iter
   */
  void read_feature_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list, const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off);

  /*
   * [한국어]
   * read_feature_merged - hetero 와 유사하나 모든 타입이 공통 offset=0 을 쓰는 단순화 버전.
   *                      (MLP/homogenized 형태 벤치마크에 사용.)
   *
   * @param num_iter, i_ptr_list, i_index_ptr_list, num_index, dim, cache_dim:
   *        hetero 버전과 동일 의미. key_off 는 내부적으로 0 으로 고정.
   *
   * 호출 체인: Python → [이 함수] → 스트림 × num_iter read_feature_kernel.
   */
  void read_feature_merged(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list, const std::vector<uint64_t>&   num_index, int dim, int cache_dim);

  /*
   * [한국어]
   * read_feature_merged_hetero - merged 경로에 타입별 key_off 를 추가한 변형.
   *                              read_feature_hetero 와 거의 같은 기능이며 벤치마크 호환성 때문에 별도 유지.
   */
  void read_feature_merged_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list, const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off);

  /*
   * [한국어]
   * cpu_backing_buffer - Constant CPU Buffer 용 host-pinned 메모리 할당.
   *
   * @param dim: feature 차원(원소 단위).
   * @param len: 버퍼에 들어갈 노드 수.
   *
   * cudaHostAlloc(cudaHostAllocMapped) 로 host-pinned 영역을 확보하고
   * cudaHostGetDevicePointer 로 device-visible 주소를 확보, CPU_buffer 필드 채움.
   * 이후 cpu_buffer_flag=true 로 전환해 read_feature 가 hybrid 경로를 택하게 한다.
   */
  void cpu_backing_buffer(uint64_t dim, uint64_t len);

  /*
   * [한국어]
   * set_cpu_buffer - 주어진 노드 ID 리스트를 CPU 버퍼 슬롯에 매핑하고 실제 feature 를 채움.
   *
   * @param idx_buffer: 핫 노드 ID 배열(GPU 주소, uint64).
   * @param num:        배열 길이.
   *
   * 내부에서 두 개 커널 연속 실행: set_cpu_buffer_kernel(range 테이블 갱신) →
   * set_cpu_buffer_data_kernel(실제 SSD → CPU buffer 복사). seq_flag=false 로 전환.
   */
  void set_cpu_buffer(uint64_t idx_buffer, int num);

  /*
   * [한국어]
   * set_window_buffering - 향후 배치에 사용될 페이지들을 미리 fetch 해 page cache 에 앉힌다.
   *
   * @param id_idx:    페이지 ID 배열(GPU 주소, uint64).
   * @param num_pages: 프리패치할 페이지 수(grid 크기).
   * @param hash_off:  해시 오프셋(타입별 기준).
   *
   * set_window_buffering_kernel 런칭. 각 block 의 첫 스레드만 bam_ptr 를 통해 페이지 캐시
   * 에 해당 page 를 로드(=counter 설정)한다.
   */
  void set_window_buffering(uint64_t id_idx,  int64_t num_pages, int hash_off);

  /*
   * [한국어]
   * print_stats - 페이지 캐시/array/Controller 통계를 모두 출력하고 리셋한다.
   *               커널 누적 시간과 total_access 도 함께 리포트.
   */
  void print_stats();

  /*
   * [한국어]
   * print_stats_no_ctrl - Controller 통계를 제외하고 페이지 캐시 + array 만 출력/리셋.
   *                      운영 중 controller 상태 보존하고 캐시 효율만 보고 싶을 때 사용.
   */
  void print_stats_no_ctrl();


  /*
   * [한국어]
   * get_array_ptr - a->d_array_ptr (array_d_t<TYPE>*) 를 uint64_t 로 캐스팅해 Python 에 반환.
   *                 Python 측이 다른 커스텀 커널에 인자로 그대로 넘길 때 사용.
   */
  uint64_t get_array_ptr();

  /*
   * [한국어]
   * get_offset_array - set_offsets 로 저장한 3-튜플 포인터를 uint64_t 로 반환.
   */
  uint64_t get_offset_array();

  /*
   * [한국어]
   * set_offsets - 샘플러가 사용할 3-튜플 오프셋(in/index/data) 을 저장.
   *
   * @param in_off, index_off, data_off: 바이트 오프셋들.
   */
  void set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off);

  /*
   * [한국어]
   * store_tensor - 호스트/GPU 상의 feature 텐서를 SSD 로 기록한다.
   *
   * @param tensor_ptr: 입력 텐서 포인터(GPU 주소).
   * @param num:        기록할 노드 수.
   * @param offset:     SSD 상의 바이트 오프셋(이 노드들이 놓일 위치).
   *
   * write_feature_kernel2 를 <<<num, 128>>> 로 런칭해 각 블록이 한 노드를 기록.
   * 완료 후 flush_cache() 로 페이지 캐시의 더티 페이지를 강제 플러시한다.
   */
  void store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset);

  /*
   * [한국어]
   * read_tensor - 디버그용 순차 읽기. seq_read_kernel 을 <<<1,1>>> 로 실행.
   *
   * @param num:    읽을 원소 수.
   * @param offset: 시작 논리 인덱스.
   */
  void read_tensor( uint64_t num, uint64_t offset);

  /*
   * [한국어]
   * flush_cache - BaM 페이지 캐시의 flush_cache() 를 호출하고 디바이스 sync.
   *               store_tensor 이후 내구성(persistence) 보장을 위해 사용.
   */
  void flush_cache();

  /*
   * [한국어]
   * get_cpu_access_count - 최근 cudaMemcpy 로 가져온 cpu_access_count 를 반환.
   */
  unsigned int get_cpu_access_count();

  /*
   * [한국어]
   * flush_cpu_access_count - 호스트/디바이스 CPU 버퍼 히트 카운터를 0 으로 초기화.
   */
  void flush_cpu_access_count();

};

#endif
