/*
 * [한국어] GIDS 호스트 측 메인 구현 + pybind11 바인딩 (gids_nvme.cu)
 *
 * === 파일의 역할 ===
 * bam_nvme.h 에 선언된 GIDS_Controllers, BAM_Feature_Store<TYPE> 의 멤버 함수
 * 정의 전부와, PYBIND11_MODULE(BAM_Feature_Store, m) 으로 Python 에 노출할 API
 * 를 담는다. BaM 의 Controller/page_cache_t/range_t/array_t 를 직접 인스턴스화
 * 하고, GPU 커널(gids_kernel.cu)을 런칭한다. 파일은 단일 .cu 이지만 실제로 호스트
 * C++ + 디바이스 커널 호출이 섞여 있고, 내부에서 `#include "gids_kernel.cu"` 로
 * 커널 정의를 같은 번역 단위에 포함시킨다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Python 학습 스크립트 → GIDS_DGLDataLoader(GIDS.py) → pybind11 → [이 파일의 멤버]
 * → gids_kernel.cu (GPU) → BaM page_cache/nvm_queue → /dev/libnvm* → NVMe SSD.
 * 실행 컨텍스트는 호스트 유저스페이스 C++ 이며, 일부 멤버 변수는 device 메모리를
 * 참조한다(a->d_array_ptr 등).
 *
 * === 타 모듈과의 연결 ===
 * - 의존: bam_nvme.h, BaM 헤더 전체(ctrl.h, page_cache.h, queue.h, event.h), gids_kernel.cu,
 *         pybind11, libnvm.so(링크). C++ 표준 라이브러리.
 * - 상위: GIDS_Setup/GIDS/GIDS.py 가 `BAM_Feature_Store_float`/`_long`/`GIDS_Controllers`
 *         파이썬 클래스를 직접 사용.
 * - 공유 상태: std::vector<Controller*> ctrls 가 GIDS_Controllers 에서 생성되어
 *   BAM_Feature_Store 로 "값 복사" 되지만 실제로는 동일 포인터 공유 → Controller
 *   소유권이 느슨하게 공유된다. 소멸자를 정의하지 않아 프로세스 종료 시 OS 회수에 의존.
 *
 * === 주요 함수/구조체 요약 ===
 * - GIDS_Controllers::init_GIDS_controllers — /dev/libnvm* 선택 후 Controller 벡터 생성.
 * - BAM_Feature_Store<T>::init_controllers   — page_cache_t/range_t/array_t 인스턴스화(STRIPE).
 * - BAM_Feature_Store<T>::read_feature(_hetero/_merged/_merged_hetero) — feature fetch.
 * - BAM_Feature_Store<T>::store_tensor / read_tensor / flush_cache — SSD 쓰기/디버그 읽기.
 * - BAM_Feature_Store<T>::cpu_backing_buffer / set_cpu_buffer — Constant CPU Buffer 구성.
 * - BAM_Feature_Store<T>::set_window_buffering — 다음 배치 프리패치.
 * - PYBIND11_MODULE(...)                      — float/long 템플릿 특수화 별 Python 바인딩.
 */

/* [한국어] pybind11 코어 — py::class_, py::init 등 바인딩 DSL 을 제공. */
#include <pybind11/pybind11.h>

/* [한국어] 표준 정수 타입(uint64_t 등) 확보. BaM API 가 fixed-width 타입으로 정의됨. */
#include <cstdint>
/* [한국어] printf 계열(std::printf). 진행 로깅 용도. */
#include <cstdio>
/* [한국어] memcpy/memset(표준). 현재 파일은 사용 빈도 낮으나 헤더 일관성을 위해 포함. */
#include <cstring>
/* [한국어] ifstream/ofstream — 설정/디버그용 파일 입출력 확장 여지. */
#include <fstream>
/* [한국어] std::cout 디버그 출력. */
#include <iostream>
/* [한국어] std::runtime_error — BaM 쪽 예외와 호환, 바인딩 과정에서 사용. */
#include <stdexcept>
/* [한국어] std::string — 진단/설정 문자열 핸들링. */
#include <string>
/* [한국어] std::vector — ctrls/vr/ssd_list 등 핵심 컨테이너. */
#include <vector>


/* [한국어] C 계열 printf — BaM 쪽 헤더와 혼용될 때 호환성 보장. */
#include <stdio.h>
/* [한국어] (중복) vector — 일부 BaM 헤더 전 선언을 위해 한 번 더 보증. */
#include <vector>

/* [한국어] GIDS_Controllers/BAM_Feature_Store/GIDS_CPU_buffer 및 BaM 전체 헤더 체인. */
#include <bam_nvme.h>
/* [한국어] pybind11 의 std::vector / std::pair 자동 변환 지원. vector<uint64_t> 인자 바인딩에 필수. */
#include <pybind11/stl.h>
/* [한국어] GPU 커널 정의를 같은 TU 에 통합 — 템플릿 인스턴스화를 동일 .cu 에서 보장. */
#include "gids_kernel.cu"
//#include <bafs_ptr.h>
/* [한국어] (주석) 이전에는 bafs_ptr 를 포함했으나 BaM 병합 이후 불필요. */


/* [한국어] 고해상도 타이머 alias — read_feature 벤치마크 시간 측정에 사용. */
typedef std::chrono::high_resolution_clock Clock;

/*
 * [한국어]
 * GIDS_Controllers::init_GIDS_controllers - SSD 리스트에 해당하는 BaM Controller 객체들을 생성.
 *
 * @param num_ctrls: 사용할 SSD 개수(=ssd_list.size() 가정).
 * @param q_depth:   각 NVMe I/O 큐의 엔트리 수(queueDepth).
 * @param num_q:     Controller 당 I/O 큐 쌍(SQ/CQ) 개수(numQueues).
 * @param ssd_list:  ctrls_paths[] 에서 어떤 경로를 쓸지 지정하는 인덱스 배열.
 *
 * 실행 컨텍스트: 호스트 Python 메인 스레드 (GIDS 초기화 1회). Controller 생성자는
 * /dev/libnvm* 를 열고 admin queue 를 만든 뒤 GPU-resident SQ/CQ 를 할당한다.
 * 에러 시 BaM 내부 예외가 pybind11 을 통해 Python RuntimeError 로 승격.
 *
 * 호출 체인:
 *   Python GIDS_Controllers().init_GIDS_controllers
 *     → [이 함수] → BaM Controller(ctor) → libnvm ioctl → 커널 모듈 → NVMe admin queue.
 */
void GIDS_Controllers::init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q,
                          const std::vector<int>& ssd_list){

  n_ctrls = num_ctrls;      /* [한국어] 멤버에 저장 — 이후 루프 상한, Python 확인용. */
  queueDepth = q_depth;      /* [한국어] 모든 Controller 생성에 동일하게 전달될 큐 깊이. */
  numQueues = num_q;         /* [한국어] Controller 당 큐 쌍 개수. */

  for (size_t i = 0; i < n_ctrls; i++) {                      /* [한국어] 선택된 SSD 마다 Controller 1개 생성. */
 	printf("SSD index: %i\n", ssd_list[i]);                    /* [한국어] 디버그 — 어떤 /dev/libnvm* 를 여는지 확인. */
       	  ctrls.push_back(new Controller(ctrls_paths[ssd_list[i]], nvmNamespace, cudaDevice, queueDepth, numQueues));
       	  /* [한국어] BaM Controller 생성자는: (1) /dev/libnvm{idx} open,
       	   *          (2) admin queue 매핑, (3) Identify Controller/NS,
       	   *          (4) GPU-resident I/O 큐 queueDepth × numQueues 쌍 할당. */
  }
}


/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::cpu_backing_buffer - Constant CPU Buffer 용 host-pinned 메모리 할당.
 *
 * @param dim: 각 노드 feature 차원(원소 단위).
 * @param len: 버퍼에 저장할 노드 수.
 *
 * cudaHostAlloc(cudaHostAllocMapped) 로 host 가상주소를, cudaHostGetDevicePointer 로
 * GPU 가상주소를 동시에 확보한다. 이후 cpu_buffer_flag=true 로 전환되어
 * read_feature 가 hybrid 커널을 선택한다. 실행 컨텍스트는 호스트 초기화 단계.
 *
 * 호출 체인: Python BAM_Feature_Store_X.cpu_backing_buffer → [이 함수].
 */
template <typename TYPE>
void BAM_Feature_Store<TYPE>::cpu_backing_buffer(uint64_t dim, uint64_t len){
  TYPE* cpu_buffer_ptr;                                             /* [한국어] host 가상주소(pinned memory) 를 받을 임시 포인터. */
  TYPE* d_cpu_buffer_ptr;                                           /* [한국어] 동일 버퍼의 device 가상주소를 받을 임시 포인터. */

  cuda_err_chk(cudaHostAlloc((TYPE **)&cpu_buffer_ptr, sizeof(TYPE) * dim * len, cudaHostAllocMapped));
  /* [한국어] cudaHostAllocMapped: (a) host-pinned(페이지 잠금) 메모리 할당, (b) GPU 가 PCIe 로
   * 직접 접근 가능하도록 매핑까지 수행. 실패 시 cuda_err_chk 가 프로그램 중단. */
  cuda_err_chk(cudaHostGetDevicePointer((TYPE **)&d_cpu_buffer_ptr, (TYPE *)cpu_buffer_ptr, 0));
  /* [한국어] 위에서 잡은 호스트 포인터로부터 device-side 주소 획득. flag 인자는 0(예약). */

  CPU_buffer.cpu_buffer_dim = dim;                                  /* [한국어] 차원 저장 — 이후 확장/검증 용도. */
  CPU_buffer.cpu_buffer_len = len;                                  /* [한국어] 길이 저장 — 커널에서 hit 범위 판정. */
  CPU_buffer.cpu_buffer = cpu_buffer_ptr;                           /* [한국어] host 주소 저장 — Python 측이 초기 데이터 채울 때 사용 가능. */
  CPU_buffer.device_cpu_buffer = d_cpu_buffer_ptr;                  /* [한국어] device 주소 — read_feature 커널 인자로 전달. */
  cpu_buffer_flag = true;                                           /* [한국어] hybrid 경로 활성화 — 이후 read_feature 는 CPU-backed 커널 선택. */
}

/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::init_controllers - BaM page cache / range / array 초기화.
 *
 * @param GIDS_ctrl:  이미 Controller 벡터를 가진 GIDS_Controllers(값 복사).
 * @param ps:         페이지 크기(바이트) — pageSize 및 dim=ps/sizeof(TYPE) 계산에 사용.
 * @param read_off:   SSD 에서의 시작 오프셋(바이트).
 * @param cache_size: 페이지 캐시 크기(MB 단위).
 * @param num_ele:    feature 텐서의 총 원소 수.
 * @param num_ssd:    사용할 SSD 수(기본 1, GIDS_ctrl.ctrls.size() 와 일치해야 함).
 *
 * 주의: 헤더의 파라미터 순서(r_off, num_ele, cache_size)와 본 구현의 순서가
 *       (read_off, cache_size, num_ele) 로 다르다. Python 에서 호출 시 본 정의
 *       순서가 우선 — pybind11 바인딩(PYBIND11_MODULE)이 이 정의로 등록됨.
 *
 * 구성 단계:
 *   (1) 멤버 변수(numElems/read_offset/n_ctrls/pageSize/dim) 설정
 *   (2) ctrls 공유 복사
 *   (3) page_cache_t 생성(GPU-resident 페이지 캐시)
 *   (4) range_t<TYPE>(STRIPE) 생성 — 다중 SSD 페이지 스트라이핑
 *   (5) vr 에 단일 range 삽입, array_t<TYPE> 생성 — 커널용 뷰
 *   (6) d_cpu_access 통계 카운터 cudaMalloc+0 초기화
 *
 * 호출 체인: Python → [이 함수] → new page_cache_t / new range_t / new array_t.
 */
template <typename TYPE>
void BAM_Feature_Store<TYPE>::init_controllers(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, uint64_t cache_size, uint64_t num_ele, uint64_t num_ssd = 1) {

  numElems = num_ele;                       /* [한국어] 전체 feature 원소 수 저장 — range/array 생성 인자. */
  read_offset = read_off;                    /* [한국어] SSD 상 시작 오프셋. range 가 LBA 변환에 사용. */
  n_ctrls = num_ssd;                         /* [한국어] 스트라이핑 SSD 수. store_tensor 에서도 참조. */
  this -> pageSize = ps;                     /* [한국어] 페이지 크기(BaM 의 I/O 단위). */
  this -> dim = ps / sizeof(TYPE);           /* [한국어] 한 페이지당 feature 원소 수 = 페이지당 "행" 크기. */
  this -> total_access = 0;                  /* [한국어] 통계 초기화. */

  ctrls = GIDS_ctrl.ctrls;                    /* [한국어] Controller 포인터 벡터를 공유 복사(소유권은 GIDS_Controllers). */

  std::cout << "Ctrl sizes: " << ctrls.size() << std::endl;  /* [한국어] 확인 로그 — num_ssd 와 일치하는지 점검. */
  uint64_t page_size = pageSize;               /* [한국어] 지역 변수로 복사 — 가독성. */
  uint64_t n_pages = cache_size * 1024LL*1024/page_size;  /* [한국어] MB → 페이지 수 변환. LL 로 64비트 산술 강제. */
  this -> numPages = n_pages;                  /* [한국어] 멤버 갱신 — 페이지 캐시 크기. */

  std::cout << "n pages: " << (int)(this->numPages) <<std::endl;   /* [한국어] (int 캐스팅 주의: 2^31 초과 시 음수 출력 가능. 로깅 전용). */
  std::cout << "page size: " << (int)(this->pageSize) << std::endl;
  std::cout << "num elements: " << this->numElems << std::endl;

  this -> h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0],(uint64_t)64, ctrls);
  /* [한국어] BaM 페이지 캐시 생성. 인자: (page_size, n_pages, cudaDevice,
   *          첫 Controller 참조, queue_depth_per_cache_bucket=64, 전체 ctrls 벡터).
   *          내부에서 GPU memory n_pages*page_size 를 cudaMalloc 하고 페이지 슬롯
   *          관리 구조(dirty/valid 비트, LRU 등) 를 초기화한다. */
  page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
  /* [한국어] device-side page_cache 포인터. 여기선 사용되지 않지만 디버깅 용 습득. */
  uint64_t t_size = numElems * sizeof(TYPE);   /* [한국어] 전체 feature 텐서 바이트 크기. */

  this -> h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, (uint64_t)read_off,
                              (uint64_t)(t_size / page_size), (uint64_t)0,
                              (uint64_t)page_size, h_pc, cudaDevice,
			      //REPLICATE
			      STRIPE
			      );
  /* [한국어] range_t 생성: (start_idx=0, n_elems=numElems, ssd_start_off=read_off,
   *          n_pages = t_size/page_size, start_page=0, page_size, page_cache, cudaDev, STRIPE).
   *          STRIPE: 페이지 i 는 ctrls[i % n_ctrls] 의 SSD 에 배치 — 병렬 대역폭 극대화. */


  this -> d_range = (range_d_t<TYPE> *)h_range->d_range_ptr;
  /* [한국어] range 의 device-side 뷰 포인터 저장 — GPU 커널에서 get_cpu_offset 호출 시 사용. */

  this -> vr.push_back(nullptr);                /* [한국어] array_t 가 요구하는 vector<range_t*> 를 1개 슬롯으로 만들고 */
  this -> vr[0] = h_range;                      /* [한국어] 유일한 range 를 삽입. 단일 range 구성. */
  this -> a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);
  /* [한국어] BaM array_t 생성: 논리 원소 수 numElems, 시작 오프셋 0, range 리스트 vr.
   *          내부에 array_d_t<TYPE>* d_array_ptr 를 할당 — 모든 GPU 커널이 이 포인터로 read/write. */

  cudaMalloc(&d_cpu_access, sizeof(unsigned int));   /* [한국어] CPU 버퍼 히트 카운터(디바이스) 메모리 1개 확보. */
  cudaMemset(d_cpu_access, 0 , sizeof(unsigned));     /* [한국어] 0 으로 초기화 — 첫 배치 전 리셋 상태. */


  return;                                             /* [한국어] 초기화 완료. 에러는 내부 예외 경로로 전파. */
}







/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::set_window_buffering - 다음 배치용 페이지 프리패치 런처.
 *
 * @param id_idx:    페이지 ID 배열(GPU 주소, uint64_t*).
 * @param num_pages: 프리패치할 페이지 수(=grid 크기).
 * @param hash_off:  heterograph 타입 오프셋(기본 0).
 *
 * 커널은 block 당 첫 스레드만 쓰는 단순 루틴이지만 blockDim=32 로 워프 전체 내려보낸다
 * (나머지 31 스레드는 no-op). 실행 컨텍스트: 호스트 런처, GPU 커널 대기.
 *
 * 호출 체인: Python → [이 함수] → set_window_buffering_kernel → bam_ptr.set_window_buffer_counter.
 */
template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_window_buffering(uint64_t id_idx,  int64_t num_pages, int hash_off = 0){
	 uint64_t* idx_ptr = (uint64_t*) id_idx;                       /* [한국어] Python 에서 넘어온 GPU 주소(uint64)를 포인터로 재해석. */
	 uint64_t page_size = pageSize;                                 /* [한국어] 지역 복사 — 커널 인자로 전달. */
	 set_window_buffering_kernel<TYPE><<<num_pages, 32>>>(a->d_array_ptr,idx_ptr, page_size, hash_off);
	 /* [한국어] grid=num_pages, block=32(=1 warp). block 당 tid==0 만 실질 동작. */
	 cuda_err_chk(cudaDeviceSynchronize())                           /* [한국어] 프리패치 완료를 호스트에서 동기화. */
}


/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::print_stats_no_ctrl - Controller 제외, page cache + array 통계 출력.
 *
 * Controller 내부 카운터(예: SQ 발행 수)는 보존한 채 캐시 적중률만 보고 싶을 때 사용.
 * print_reset_stats() 는 "출력 후 0 리셋" 시맨틱스(이름 그대로).
 *
 * 호출 체인: Python → [이 함수] → page_cache_t::print_reset_stats / array_t::print_reset_stats.
 */
template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats_no_ctrl(){
  std::cout << "print stats: ";                /* [한국어] 로그 헤더. */
  this->h_pc->print_reset_stats();               /* [한국어] 페이지 캐시 hit/miss 통계 출력 + 리셋. */
  std::cout << std::endl;

  std::cout << "print array reset: ";
  this->a->print_reset_stats();                  /* [한국어] array_t 자체의 접근 통계. */
  std::cout << std::endl;
}


/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::print_stats - 모든 통계(page cache, array, Controller×n, 커널 시간) 출력/리셋.
 *
 * 호출 체인: Python → [이 함수] → page_cache/array/controllers 각각 print_reset_stats.
 */
template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats(){
  std::cout << "print stats: ";
  this->h_pc->print_reset_stats();               /* [한국어] 페이지 캐시 통계. */
  std::cout << std::endl;

  std::cout << "print array reset: ";
  this->a->print_reset_stats();                  /* [한국어] array 통계. */
  std::cout << std::endl;

  for(int i = 0; i < n_ctrls; i++){              /* [한국어] 각 SSD Controller 의 NVMe 카운터. */
 	std::cout << "print ctrl reset " << i << ": ";
  	(this->ctrls[i])->print_reset_stats();        /* [한국어] BaM Controller::print_reset_stats — SQE/CQE 수 등. */
  	std::cout << std::endl;

  }

  std::cout << "Kernel Time: \t " << this->kernel_time << std::endl;  /* [한국어] 누적 커널 시간(ms) 출력. */
  this->kernel_time = 0;                                               /* [한국어] 누적 시간 리셋. */
  std::cout << "Total Access: \t " << this->total_access << std::endl; /* [한국어] 총 요청 노드 수. */
  this->total_access = 0;                                              /* [한국어] 리셋. */
}







/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::read_feature - 단일 호모지니어스 배치 feature fetch 런처.
 *
 * @param i_ptr:      출력 텐서의 GPU 주소(uint64, 파이썬에서 tensor.data_ptr()).
 * @param i_index_ptr:노드 ID 배열(int64*, GPU) 의 uint64 주소.
 * @param num_index:  노드 수.
 * @param dim:        결과 차원.
 * @param cache_dim:  페이지 내부 행 크기(원소 단위).
 * @param key_off:    기준 ID 오프셋(기본 0).
 *
 * 커널 런칭 전후 cudaDeviceSynchronize 로 순수 커널 시간만 측정(kernel_time 누적).
 * cpu_buffer_flag 분기로 두 커널 중 하나를 선택. 마지막에 d_cpu_access → 호스트 복사.
 *
 * 호출 체인: Python GIDS_DGLDataLoader 배치 fetch → [이 함수] → read_feature_kernel(_with_cpu_backing).
 */
template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                     int64_t num_index, int dim, int cache_dim, uint64_t key_off = 0) {

  TYPE *tensor_ptr = (TYPE *)i_ptr;                  /* [한국어] Python 에서 받은 GPU 주소를 타입 있는 포인터로 재해석. */
  int64_t *index_ptr = (int64_t *)i_index_ptr;       /* [한국어] 동일. 노드 ID 배열 포인터. */

  uint64_t b_size = blkSize;                          /* [한국어] block 크기(기본 128). */
  uint64_t n_warp = b_size / 32;                      /* [한국어] block 당 warp 수. */
  uint64_t g_size = (num_index+n_warp - 1) / n_warp;  /* [한국어] grid 크기 = 필요한 warp 수를 블록 단위로 올림. */

  cuda_err_chk(cudaDeviceSynchronize());              /* [한국어] 이전 작업 완료 후 타이머 시작 — 정확한 측정. */
  auto t1 = Clock::now();                              /* [한국어] 시작 시각. */
  if(cpu_buffer_flag == false){                       /* [한국어] CPU 버퍼 미사용 → BaM 단독 경로. */
    read_feature_kernel<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                  index_ptr, dim, num_index, cache_dim, key_off);
    /* [한국어] read_feature_kernel 런칭: 각 warp = 한 노드. */
  }
  else{                                                /* [한국어] Constant CPU Buffer 활성. */
    read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size>>>(a->d_array_ptr, d_range, tensor_ptr,
                                                  index_ptr, dim, num_index, cache_dim, CPU_buffer, seq_flag,
                                                  d_cpu_access, key_off);
    /* [한국어] hybrid 커널 런칭 — range 의 get_cpu_offset 비트 + CPU_buffer 를 참조. */
  }
  cuda_err_chk(cudaDeviceSynchronize());               /* [한국어] 커널 완료 대기 — 정확한 종료 시각 얻음. */
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  /* [한국어] 이 배치에서의 CPU 히트 카운터를 호스트로 복사(통계 조회용). */
  auto t2 = Clock::now();                              /* [한국어] 종료 시각. */
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)                 /* [한국어] 마이크로초 차이. */
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)                 /* [한국어] (사용되지 않음; 레거시 변수). */
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)
  /* [한국어] us/1000 으로 소수점 포함 ms 산출. */

  kernel_time += ms_fractional;                         /* [한국어] 누적 커널 시간. */
  total_access += num_index;                            /* [한국어] 누적 접근 노드 수. */

  return;                                               /* [한국어] 정상 종료 — Python 측은 out_tensor_ptr 가 채워졌다고 가정. */
}

/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::read_feature_hetero - 이종 그래프 배치를 타입별 CUDA 스트림으로 병렬 fetch.
 *
 * @param num_iter:        node type 개수 (벡터 인자들의 길이와 동일).
 * @param i_ptr_list:      각 type 의 출력 텐서 포인터(uint64).
 * @param i_index_ptr_list:각 type 의 노드 ID 배열 포인터(uint64, int64*).
 * @param num_index:       type 별 노드 개수.
 * @param dim, cache_dim:  공통 차원.
 * @param key_off:         type 별 기준 ID 오프셋.
 *
 * num_iter 개 스트림을 만들어 커널을 비동기 발행 → 마지막에 모두 동기화.
 * 스트림 간 오버랩으로 여러 type 의 fetch 를 동시 수행. 실행 컨텍스트는 호스트 런처.
 *
 * 호출 체인: Python heterograph DataLoader → [이 함수] → read_feature_kernel × num_iter.
 */
template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off) {

  cudaStream_t streams[num_iter];                       /* [한국어] 스택에 스트림 배열 할당(VLA). */
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);                     /* [한국어] type 별 별도 CUDA 스트림 생성 — 실제 비동기 오버랩 가능. */
  }

  cuda_err_chk(cudaDeviceSynchronize());                 /* [한국어] 이전 작업 끝낸 뒤 타이머 시작. */
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){               /* [한국어] type 별로 커널 발행. */
    uint64_t i_ptr = i_ptr_list[i];                      /* [한국어] 출력 텐서 GPU 주소(uint64). */
    uint64_t    i_index_ptr =  i_index_ptr_list[i];     /* [한국어] 노드 ID 배열 GPU 주소. */
    TYPE *tensor_ptr = (TYPE *) i_ptr;                   /* [한국어] 타입 있는 포인터로 재해석. */
    int64_t *index_ptr = (int64_t *)i_index_ptr;         /* [한국어] 노드 ID 포인터. */

    uint64_t b_size = blkSize;                           /* [한국어] block size. */
    uint64_t n_warp = b_size / 32;                        /* [한국어] block 당 warp 수. */
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp; /* [한국어] 해당 type 에 필요한 grid size. */

    if(cpu_buffer_flag == false){                        /* [한국어] BaM 단독 경로. */
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, key_off[i]);
      /* [한국어] 스트림 i 에 비동기 런칭. */
    }
    else{                                                /* [한국어] hybrid 경로. */
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag,
                                                    d_cpu_access,  key_off[i]);
    }
    total_access += num_index[i];                        /* [한국어] 누적 통계 갱신. */
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);                    /* [한국어] 각 스트림 완료 대기. */
  }

  cuda_err_chk(cudaDeviceSynchronize());                  /* [한국어] 디바이스 전체 동기화(이중 안전). */
  cuda_err_chk(cudaDeviceSynchronize());                  /* [한국어] (중복이지만 원본 유지). */
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  /* [한국어] CPU 버퍼 히트 카운터 수거. */

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)                    /* [한국어] µs 단위 시간. */
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)                    /* [한국어] (사용 X, 레거시). */
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;

  kernel_time += ms_fractional;                            /* [한국어] 누적 커널 시간. */

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);                        /* [한국어] 스트림 리소스 해제. */
  }


  return;
}



/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::read_feature_merged - 모든 type 이 공통 key_off=0 을 쓰는 단순화 fetch.
 *                                                (벤치마크 호환용). cache_dim 기본값 1024.
 *
 * 매개변수 의미는 read_feature_hetero 와 동일하나 key_off 벡터 대신 0 을 사용.
 */
template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim=1024) {

  cudaStream_t streams[num_iter];                         /* [한국어] type 별 스트림 배열. */
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);                       /* [한국어] 스트림 생성. */
  }

  cuda_err_chk(cudaDeviceSynchronize());                   /* [한국어] 시작 전 동기화. */
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];                        /* [한국어] 출력 포인터. */
    uint64_t    i_index_ptr =  i_index_ptr_list[i];       /* [한국어] 인덱스 포인터. */
    TYPE *tensor_ptr = (TYPE *) i_ptr;                     /* [한국어] 타입 캐스트. */
    int64_t *index_ptr = (int64_t *)i_index_ptr;           /* [한국어] 인덱스 캐스트. */

    uint64_t b_size = blkSize;                             /* [한국어] block 크기. */
    uint64_t n_warp = b_size / 32;                          /* [한국어] block 당 warp 수. */
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;  /* [한국어] grid 크기. */


    if(cpu_buffer_flag == false){                          /* [한국어] BaM 단독. */
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, 0);
      /* [한국어] key_off=0 고정. */
    }
    else{                                                  /* [한국어] hybrid. */
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag,
                                                    d_cpu_access, 0);
    }
    total_access += num_index[i];                          /* [한국어] 통계. */
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);                      /* [한국어] 스트림 완료 대기. */
  }

  cuda_err_chk(cudaDeviceSynchronize());                    /* [한국어] 전체 동기. */
  cuda_err_chk(cudaDeviceSynchronize());                    /* [한국어] (원본 중복 유지). */
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  /* [한국어] CPU 히트 카운터 수거. */

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;

  kernel_time += ms_fractional;                              /* [한국어] 커널 시간 누적. */

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);                          /* [한국어] 스트림 해제. */
  }
  return;
}





/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::read_feature_merged_hetero - merged + hetero 결합 경로.
 *
 * read_feature_hetero 와 거의 동일하지만 cache_dim 이 외부 고정이며 벤치마크 스크립트
 * 호환성을 위해 분리. 매개변수 의미는 read_feature_hetero 참조.
 */
template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off) {

  cudaStream_t streams[num_iter];                          /* [한국어] type 별 스트림. */
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);                        /* [한국어] 생성. */
  }

  cuda_err_chk(cudaDeviceSynchronize());                    /* [한국어] 동기 시작. */
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];                         /* [한국어] 출력 주소. */
    uint64_t    i_index_ptr =  i_index_ptr_list[i];        /* [한국어] 인덱스 주소. */
    TYPE *tensor_ptr = (TYPE *) i_ptr;                      /* [한국어] 캐스트. */
    int64_t *index_ptr = (int64_t *)i_index_ptr;            /* [한국어] 캐스트. */

    uint64_t b_size = blkSize;                              /* [한국어] block 크기. */
    uint64_t n_warp = b_size / 32;                           /* [한국어] warp 수. */
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;    /* [한국어] grid 크기. */


    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, key_off[i]);
      /* [한국어] 타입별 key_off 적용 — hetero 경로. */
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag,
                                                    d_cpu_access, key_off[i]);
    }
    total_access += num_index[i];                            /* [한국어] 통계 누적. */
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);                   /* [한국어] 각 스트림 완료 대기. */
  }

  cuda_err_chk(cudaDeviceSynchronize());                 /* [한국어] 전체 동기. */
  cuda_err_chk(cudaDeviceSynchronize());                 /* [한국어] (중복 유지). */
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  /* [한국어] CPU 히트 카운터 수거. */

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;

  kernel_time += ms_fractional;                           /* [한국어] 커널 시간 누적. */

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);                       /* [한국어] 스트림 해제. */
  }
  return;
}







/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::store_tensor - GPU 상의 feature 텐서를 SSD 로 기록.
 *
 * @param tensor_ptr: 입력 텐서 포인터(GPU 주소, uint64).
 * @param num:        기록할 노드 수(=grid 크기).
 * @param offset:     SSD 상의 바이트 오프셋. 커널에는 sizeof(TYPE) 로 나눠 원소 단위로 전달.
 *
 * write_feature_kernel2 를 <<<num, 128>>> 로 발행: 한 블록이 한 노드를 담당하며
 * bam_ptr[] 대입을 통해 페이지를 dirty 로 만든다. 이후 flush_cache() 로 페이지 캐시의
 * 더티 페이지를 NVMe Write 로 플러시한다. 반드시 cudaDeviceSynchronize 로 완료 확인.
 *
 * 호출 체인: Python BAM_Feature_Store_X.store_tensor → [이 함수] → write_feature_kernel2
 *            → bam_ptr[]= → page_cache dirty → flush_cache → nvm_queue Write SQE → SSD.
 */
template <typename TYPE>
void  BAM_Feature_Store<TYPE>::store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset){


//__global__ void write_feature_kernel2(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr, uint64_t dim, uint32_t num_ctrls) {
	TYPE* t_ptr = (TYPE*) tensor_ptr;                      /* [한국어] 입력 GPU 포인터 캐스트. */
	page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc -> d_pc_ptr);  /* [한국어] device-side page_cache 핸들. */
	size_t b_size = 128;                                    /* [한국어] block 크기 = 128 스레드(4 warp). */
	printf("num of writing node data: %llu dim: %llu\n", num, dim);  /* [한국어] 진행 로그. */
	write_feature_kernel2<TYPE><<<num, b_size>>>(h_pc->pdt.d_ctrls, d_pc, a->d_array_ptr, t_ptr, dim,  n_ctrls, offset/sizeof(TYPE));
	/* [한국어] grid=num(노드 하나당 한 블록), 블록 내 스레드가 dim 분할.
	 *          offset/sizeof(TYPE) 로 바이트 → 원소 인덱스로 변환해 전달. */
	cuda_err_chk(cudaDeviceSynchronize());                    /* [한국어] 커널 완료 대기. */
  	h_pc->flush_cache();                                       /* [한국어] dirty 페이지 전부 SSD 로 플러시 — 내구성 확보. */
   	cuda_err_chk(cudaDeviceSynchronize());                     /* [한국어] 플러시 완료 대기. */
/*
  uint64_t s_offset = 0; 
  
  uint64_t total_cache_size = (pageSize * numPages);
  uint64_t total_tensor_size = (sizeof(TYPE) * num);
  uint64_t num_pages = total_tensor_size / pageSize;

  uint32_t n_tsteps = ceil((float)(total_tensor_size)/(float)total_cache_size);  
  printf("total iter: %llu\n", (unsigned long long) n_tsteps);
  TYPE* t_ptr = (TYPE*) tensor_ptr;
  
  page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc -> d_pc_ptr);
  size_t b_size = 128;
  size_t g_size = (((total_tensor_size + pageSize -1) / pageSize)  + b_size - 1)/b_size;

  for (uint32_t cstep =0; cstep < n_tsteps; cstep++) {
    uint64_t cpysize = std::min(total_cache_size, (total_tensor_size-s_offset));


   // printf("first ele:%f\n", t_ptr[0]);
    cuda_err_chk(cudaMemcpy(h_pc->pdt.base_addr, t_ptr+s_offset+offset, cpysize, cudaMemcpyHostToDevice));
    printf("g size: %i num: %llu\n", g_size, num);
    write_feature_kernel<TYPE><<<100, b_size>>>(h_pc->pdt.d_ctrls, d_pc, a->d_array_ptr, t_ptr, num_pages, pageSize, offset, s_offset, n_ctrls);
    cuda_err_chk(cudaDeviceSynchronize());
    
  // printf("CALLLING FLUSH\n");
  // h_pc->flush_cache();
    //cuda_err_chk(cudaDeviceSynchronize());
    s_offset = s_offset + cpysize; 

  }
*/
}


/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::flush_cache - BaM 페이지 캐시 전체 플러시 + 디바이스 동기.
 *
 * store_tensor 이후 또는 주기적 내구성 포인트에서 호출. page cache 내 dirty page 를
 * 모두 NVMe Write 로 반영한다.
 */
template <typename TYPE>
void  BAM_Feature_Store<TYPE>::flush_cache(){
  h_pc->flush_cache();                       /* [한국어] BaM 의 dirty 페이지 플러시 — 내부에서 write_data 반복. */
  cuda_err_chk(cudaDeviceSynchronize());      /* [한국어] 완료 대기. */
}



/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::set_cpu_buffer - 핫 노드 리스트를 CPU 버퍼에 올리고 매핑 비트 기록.
 *
 * @param idx_buffer: 핫 노드 ID 배열(GPU 주소, uint64).
 * @param num:        노드 수.
 *
 * 두 단계: (1) set_cpu_buffer_kernel 로 range 의 매핑 비트 기록,
 *         (2) set_cpu_buffer_data_kernel 로 실제 feature 데이터 SSD→CPU 복사.
 * 마지막에 seq_flag=false 로 전환해 희소 매핑 모드 활성화.
 *
 * 호출 체인: Python → [이 함수] → set_cpu_buffer_kernel / set_cpu_buffer_data_kernel.
 */
template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_cpu_buffer(uint64_t idx_buffer, int num){

  int bsize = 1024;                                    /* [한국어] 매핑 커널의 block 크기(1D). */
  int grid = (num + bsize - 1) / bsize;                /* [한국어] 올림 grid 계산. */
  uint64_t* idx_ptr = (uint64_t* ) idx_buffer;         /* [한국어] GPU 포인터 캐스트. */
  set_cpu_buffer_kernel<TYPE><<<grid,bsize>>>(d_range, idx_ptr, num, pageSize);
  /* [한국어] 각 핫 노드 → CPU 슬롯 매핑 기록 (set_cpu_buffer(row, slot) 호출). */
  cuda_err_chk(cudaDeviceSynchronize());

  set_cpu_buffer_data_kernel<TYPE><<<num,32>>>(a->d_array_ptr, CPU_buffer.device_cpu_buffer, idx_ptr, dim, num);
  /* [한국어] 각 핫 노드 feature 를 SSD → CPU_buffer 로 복사. block=32(1 warp). */
  cuda_err_chk(cudaDeviceSynchronize());

  seq_flag = false;                                     /* [한국어] 선형 매핑(row_index 기반) → 희소 매핑(비트 기반)으로 전환. */


}



/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::set_offsets - 샘플러와 공유할 3-튜플(in/index/data) 오프셋 저장.
 *
 * @param in_off, index_off, data_off: 바이트 오프셋.
 */
template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off){

 offset_array = new uint64_t[3];                         /* [한국어] 3 원소 heap 배열 할당. 해제는 프로세스 종료 시(소멸자 없음). */
    printf("set offset: in_off: %llu index_off: %llu data_off: %llu offset_ptr:%llu\n", in_off, index_off, data_off, (uint64_t) offset_array);
    /* [한국어] 디버그 로그 — Python 측이 offset_ptr 를 어떤 값으로 받을지 확인. */

  offset_array[0] = (in_off);                             /* [한국어] 입력 오프셋. */
  offset_array[1] = (index_off);                           /* [한국어] 인덱스 오프셋. */
  offset_array[2] = (data_off);                            /* [한국어] 데이터 오프셋. */

}


/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::get_offset_array - offset_array 의 주소를 uint64 로 Python 에 전달.
 *
 * Python 측이 cffi/pytorch tensor 로 해당 주소를 재해석해 사용.
 */
template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_offset_array(){
  return ((uint64_t) offset_array);                        /* [한국어] 포인터 → uint64 형변환. */
}

/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::get_array_ptr - array_d_t<TYPE>* 를 uint64 로 Python 에 노출.
 *                                          사용자 정의 커널에서 직접 인자로 쓸 때 활용.
 */
template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_array_ptr(){
	return ((uint64_t) (a->d_array_ptr));                 /* [한국어] device array 포인터를 uint64 로 반환. */
}


/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::read_tensor - 디버그용: 단일 스레드 순차 읽기 후 printf 출력.
 *
 * @param num:    읽을 원소 수.
 * @param offset: 시작 원소 인덱스(페이지 내부 단위가 아니라 논리 인덱스).
 */
template <typename TYPE>
void  BAM_Feature_Store<TYPE>::read_tensor(uint64_t num, uint64_t offset){
  printf("offset:%llu\n", (unsigned long long) offset);     /* [한국어] 오프셋 로그. */
  seq_read_kernel<TYPE><<<1, 1>>>(a->d_array_ptr, num, offset);  /* [한국어] 단일 블록/스레드로 순차 스캔. */
  cuda_err_chk(cudaDeviceSynchronize());                      /* [한국어] 완료 대기. */

}


/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::get_cpu_access_count - 호스트 쪽에 캐싱된 CPU 히트 카운터 반환.
 *
 * 마지막 read_feature* 호출 끝에서 cudaMemcpy 로 갱신된 값.
 */
template <typename TYPE>
unsigned int BAM_Feature_Store<TYPE>::get_cpu_access_count(){
	return cpu_access_count;                                   /* [한국어] 호스트 사본 반환. */
}

/*
 * [한국어]
 * BAM_Feature_Store<TYPE>::flush_cpu_access_count - 호스트/디바이스 CPU 히트 카운터 모두 0 으로 리셋.
 *
 * 배치 단위 또는 epoch 단위로 호출해 누적 통계를 초기화.
 */
template <typename TYPE>
void BAM_Feature_Store<TYPE>::flush_cpu_access_count(){
	cpu_access_count = 0;                                      /* [한국어] 호스트 사본 리셋. */
  cudaMemset(d_cpu_access, 0 , sizeof(unsigned));              /* [한국어] 디바이스 카운터 리셋 — 다음 커널부터 0 에서 시작. */
}

/*
 * [한국어]
 * create_BAM_Feature_Store<T> - 자유 함수 팩토리. 현재 pybind11 에 노출되지 않음(py::init<> 사용).
 *                               템플릿 디스패치 예시로 보존된 헬퍼.
 *
 * @return: 값 타입 BAM_Feature_Store<T> (복사 비용 주의).
 */
template <typename T>
BAM_Feature_Store<T> create_BAM_Feature_Store() {
    return BAM_Feature_Store<T>();                            /* [한국어] 기본 생성자로 빈 스토어 반환(추후 init_controllers 필요). */
}



/*
 * [한국어]
 * PYBIND11_MODULE(BAM_Feature_Store, m) - Python 모듈 진입점(확장 로드 시 자동 실행).
 *
 * 이 블록이 만들어내는 Python 심볼:
 *   - BAM_Feature_Store_float : C++ 의 BAM_Feature_Store<float>   에 대응 (GNN feature 경로).
 *   - BAM_Feature_Store_long  : C++ 의 BAM_Feature_Store<int64_t> 에 대응 (라벨/정수 feature).
 *   - GIDS_Controllers         : C++ 의 GIDS_Controllers 구조체.
 *
 * 각 py::class_ 블록은 .def("name", &Cls::method) 로 멤버 함수를 Python 메서드로 노출한다.
 * 타입 디스패치는 Python 에서 "어느 클래스를 사용할지" 로 이루어진다 —
 * GIDS_Setup/GIDS/GIDS.py 가 feature dtype 에 따라 _float / _long 을 선택.
 */
PYBIND11_MODULE(BAM_Feature_Store, m) {
  m.doc() = "Python bindings for an example library";    /* [한국어] 모듈 독스트링(Python help() 용). */

  namespace py = pybind11;                                /* [한국어] 이 스코프에서 pybind11 네임스페이스 단축. */

  //py::class_<BAM_Feature_Store<>, std::unique_ptr<BAM_Feature_Store<float>, py::nodelete>>(m, "BAM_Feature_Store")
  /* [한국어] (주석) 이전에는 py::nodelete 로 소유권을 떼어냈지만 현재는 기본 소유. */
    py::class_<BAM_Feature_Store<float>>(m, "BAM_Feature_Store_float")
      /* [한국어] 템플릿 특수화 <float> → 파이썬 클래스 "BAM_Feature_Store_float" 로 노출. */
      .def(py::init<>())                                                              /* [한국어] 기본 생성자 → Python BAM_Feature_Store_float(). */
      .def("init_controllers", &BAM_Feature_Store<float>::init_controllers)           /* [한국어] Python bfs.init_controllers(...) — BaM 초기화. */
      .def("read_feature", &BAM_Feature_Store<float>::read_feature)                   /* [한국어] Python bfs.read_feature(...) — 단일 배치 fetch. */
      .def("read_feature_hetero", &BAM_Feature_Store<float>::read_feature_hetero)     /* [한국어] hetero fetch. */

      .def("read_feature_merged_hetero", &BAM_Feature_Store<float>::read_feature_merged_hetero)  /* [한국어] merged+hetero. */
      .def("read_feature_merged", &BAM_Feature_Store<float>::read_feature_merged)                /* [한국어] merged. */
      .def("set_window_buffering", &BAM_Feature_Store<float>::set_window_buffering)              /* [한국어] window buffering 프리패치. */
      .def("cpu_backing_buffer", &BAM_Feature_Store<float>::cpu_backing_buffer)                  /* [한국어] Constant CPU Buffer 할당. */
      .def("set_cpu_buffer", &BAM_Feature_Store<float>::set_cpu_buffer)                          /* [한국어] 핫 노드 CPU 버퍼 매핑. */

      .def("flush_cache", &BAM_Feature_Store<float>::flush_cache)                               /* [한국어] page cache flush → SSD. */
      .def("store_tensor",  &BAM_Feature_Store<float>::store_tensor)                            /* [한국어] 텐서 → SSD 기록. */
      .def("read_tensor",  &BAM_Feature_Store<float>::read_tensor)                              /* [한국어] 디버그 순차 읽기. */

      .def("get_array_ptr", &BAM_Feature_Store<float>::get_array_ptr)                           /* [한국어] array_d_t 포인터 노출. */
      .def("get_offset_array", &BAM_Feature_Store<float>::get_offset_array)                     /* [한국어] offset_array 포인터 노출. */
      .def("set_offsets", &BAM_Feature_Store<float>::set_offsets)                               /* [한국어] 3-튜플 오프셋 설정. */
      .def("get_cpu_access_count", &BAM_Feature_Store<float>::get_cpu_access_count)             /* [한국어] CPU 히트 카운터 조회. */
      .def("flush_cpu_access_count", &BAM_Feature_Store<float>::flush_cpu_access_count)         /* [한국어] 히트 카운터 리셋. */

      .def("print_stats", &BAM_Feature_Store<float>::print_stats);                              /* [한국어] 통계 출력+리셋 (세미콜론 = class_ 체인 종료). */



    py::class_<BAM_Feature_Store<int64_t>>(m, "BAM_Feature_Store_long")
      /* [한국어] 템플릿 특수화 <int64_t> → 파이썬 클래스 "BAM_Feature_Store_long" 로 노출. 라벨/정수 feature 용. */
      .def(py::init<>())                                                                   /* [한국어] 기본 생성자. */
      .def("init_controllers", &BAM_Feature_Store<int64_t>::init_controllers)              /* [한국어] 초기화. */
      .def("read_feature", &BAM_Feature_Store<int64_t>::read_feature)                      /* [한국어] 단일 fetch. */
      .def("read_feature_hetero", &BAM_Feature_Store<int64_t>::read_feature_hetero)        /* [한국어] hetero fetch. */

      .def("read_feature_merged", &BAM_Feature_Store<int64_t>::read_feature_merged)                /* [한국어] merged. */
      .def("read_feature_merged_hetero", &BAM_Feature_Store<int64_t>::read_feature_merged_hetero)  /* [한국어] merged+hetero. */


      .def("set_window_buffering", &BAM_Feature_Store<int64_t>::set_window_buffering)              /* [한국어] 프리패치. */
      .def("cpu_backing_buffer", &BAM_Feature_Store<int64_t>::cpu_backing_buffer)                  /* [한국어] CPU 버퍼 할당. */
      .def("set_cpu_buffer", &BAM_Feature_Store<int64_t>::set_cpu_buffer)                          /* [한국어] 핫 노드 매핑. */

      .def("flush_cache", &BAM_Feature_Store<int64_t>::flush_cache)                               /* [한국어] 캐시 플러시. */
      .def("store_tensor",  &BAM_Feature_Store<int64_t>::store_tensor)                            /* [한국어] 쓰기. */
      .def("read_tensor",  &BAM_Feature_Store<int64_t>::read_tensor)                              /* [한국어] 디버그 읽기. */

      .def("get_array_ptr", &BAM_Feature_Store<int64_t>::get_array_ptr)                           /* [한국어] array 포인터. */
      .def("get_offset_array", &BAM_Feature_Store<int64_t>::get_offset_array)                     /* [한국어] offset 포인터. */
      .def("set_offsets", &BAM_Feature_Store<int64_t>::set_offsets)                               /* [한국어] 오프셋 설정. */
      .def("get_cpu_access_count", &BAM_Feature_Store<int64_t>::get_cpu_access_count)             /* [한국어] 히트 조회. */
      .def("flush_cpu_access_count", &BAM_Feature_Store<int64_t>::flush_cpu_access_count)         /* [한국어] 히트 리셋. */


      .def("print_stats", &BAM_Feature_Store<int64_t>::print_stats);                              /* [한국어] 통계 출력. */




      py::class_<GIDS_Controllers>(m, "GIDS_Controllers")
      /* [한국어] BaM Controller 묶음을 Python 에 노출. GIDS.py 가 init 단계에서 사용. */
      .def(py::init<>())                                                                           /* [한국어] 기본 생성자. */
      .def("init_GIDS_controllers", &GIDS_Controllers::init_GIDS_controllers);                     /* [한국어] SSD 리스트로 Controller 벡터 생성. */

}

//gids


