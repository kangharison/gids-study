/*
 * [한국어] GIDS 의 모든 GPU 디바이스 커널 모음 (gids_kernel.cu)
 *
 * === 파일의 역할 ===
 * GIDS 가 SSD 상의 feature 를 GPU 로 직접 가져오기 위해 사용하는 CUDA __global__
 * 커널들을 담는다. 모든 커널은 호스트 측 BAM_Feature_Store 의 public 메서드
 * (read_feature, store_tensor 등)가 런칭하며, 내부에서 BaM 의 device-side accessor
 * `bam_ptr<T>` 를 통해 페이지 캐시 hit/miss 처리를 수행한다. 이 파일은 별도
 * 타깃이 아니라 gids_nvme.cu 에 `#include "gids_kernel.cu"` 로 번들된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Python(mini-batch node IDs) → GIDS_DGLDataLoader → BAM_Feature_Store::read_feature*
 * → [본 파일 커널] → bam_ptr.read() → array_d_t<T>::seq_read →
 * page_cache_d_t::acquire_page → (miss) nvm_queue.sq_enqueue → NVMe SSD.
 * 실행 컨텍스트: 모든 함수가 GPU 디바이스 코드(__global__). block/warp 가 각자
 * 하나의 "node index" 를 담당하는 것이 본 프로젝트의 공통 매핑이다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: BaM 의 array_d_t / range_d_t / page_cache_d_t / Controller (ctrl.h, page_cache.h).
 *         gids_module/include/bam_nvme.h 의 GIDS_CPU_buffer<T>.
 * - 상위 호출자: gids_nvme.cu 의 모든 BAM_Feature_Store<TYPE> 메서드.
 * - 데이터 흐름: index_ptr(int64, GPU) + dr(array_d_t<T>*) → bam_ptr 가 페이지 fetch →
 *   out_tensor_ptr 로 복사. CPU 버퍼 hit 면 SSD 우회.
 *
 * === 주요 함수/구조체 요약 ===
 * - read_feature_kernel                    : BaM 단독 경로. warp 단위 노드 ID → bam_ptr.read.
 * - read_feature_kernel_with_cpu_backing_memory : cpu_off 비트로 CPU/SSD 경로 동적 분기.
 * - set_cpu_buffer_kernel                  : range 테이블에 CPU 버퍼 매핑 비트 기록.
 * - set_cpu_buffer_data_kernel             : 선택된 노드의 feature 를 SSD → CPU 버퍼로 복사.
 * - set_window_buffering_kernel            : 차기 배치용 페이지 프리패치.
 * - read_kernel / seq_read_kernel          : 디버그용 순차 읽기 프린터.
 * - write_feature_kernel / _kernel2        : feature 쓰기 경로(SSD 저장).
 */


/*
 * [한국어]
 * read_feature_kernel - mini-batch 의 node ID 들을 받아 각 warp 가 한 노드의
 *                      feature 차원 전체를 BaM 페이지 캐시를 통해 GPU 로 가져온다.
 *
 * @param dr:              BaM array device 핸들(array_d_t<T>*). bam_ptr 생성자의 입력.
 * @param out_tensor_ptr:  결과 쓸 GPU 버퍼(길이 num_idx*dim).
 * @param index_ptr:       노드 ID 배열(int64, GPU). length == num_idx.
 * @param dim:             출력 per-node 차원.
 * @param num_idx:         처리할 노드 수.
 * @param cache_dim:       페이지 내 per-row 원소 수(통상 dim 과 동일).
 * @param key_off:         각 ID 에 더할 오프셋(heterograph 베이스 ID).
 *
 * block ↔ warp ↔ node 매핑:
 *   idx_idx = blockIdx.x * (blockDim.x/32) + (threadIdx.x/32)  → 이 warp 가 담당할 node.
 *   warp 내 32 스레드가 tid=0..31 로 dim 차원을 stride=32 로 분할 처리.
 *
 * bam_ptr.read(idx) 는 다음으로 확장된다:
 *   array_d_t<T>::seq_read(idx) → page_cache_d_t::acquire_page(page_id)
 *     → hit 면 페이지 내 오프셋 로드 / miss 면 nvm_queue.sq_enqueue + cq_poll → 재시도.
 *
 * 호출 체인:
 *   BAM_Feature_Store::read_feature → [이 커널] → bam_ptr.read → BaM 페이지 캐시 → NVMe.
 */
template <typename T = float>
__global__ void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off) {

  uint64_t bid = blockIdx.x;                  /* [한국어] 이 블록의 그리드 인덱스. 블록당 여러 warp 로 여러 노드를 처리. */
  int num_warps = blockDim.x / 32;            /* [한국어] 블록 내 warp 수 = blockDim.x / 32. GIDS 기본 blockDim=128 → 4 warp. */
  int warp_id = threadIdx.x / 32;             /* [한국어] 블록 내부 warp 번호(0..num_warps-1). */
  int idx_idx = bid * num_warps + warp_id;    /* [한국어] 전역 warp 번호 = 이 warp 가 담당할 node 의 배치 내 인덱스. */
  if (idx_idx < num_idx) {                    /* [한국어] grid 를 올림으로 잡아 overshoot 가 생기므로 유효 warp 만 통과. */
 	   bam_ptr<T> ptr(dr);                    /* [한국어] BaM device accessor 생성. 생성자 내부에서 page_cache 핸들 확보(저비용). */

        uint64_t row_index = index_ptr[idx_idx] + key_off;  /* [한국어] 노드 ID 로드 후 타입 베이스 오프셋 가산 — 이종 그래프 대응. */
      	uint64_t tid = threadIdx.x % 32;                    /* [한국어] warp 내 레인 번호(0..31). feature 차원 스트라이드로 사용. */


    for (; tid < dim; tid += 32) {                          /* [한국어] warp 협력: 32 레인이 dim 전체를 stride 32 로 분할 처리. */
	   // T temp = ptr[(row_index) * cache_dim + tid];
     const size_t idx = (row_index) * cache_dim + tid;      /* [한국어] 논리 원소 인덱스. (node * cache_dim) + 차원 레인. */
     //out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = ptr[idx];
      out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = ptr.read(idx);
      /* [한국어] ptr.read(idx): BaM page cache hit 면 캐시에서, miss 면 NVMe SQE 발행 후 완료 대기.
       *                        읽은 T 원소를 출력 텐서의 (warp_idx*dim + tid) 위치에 기록. */

    }
  }
}

/*
 * [한국어]
 * read_feature_kernel_with_cpu_backing_memory - Constant CPU Buffer(hybrid) 경로 커널.
 *
 * @param dr:          BaM array device 핸들.
 * @param range:       range_d_t<T>*. get_cpu_offset(row) 로 node → CPU 버퍼 매핑 비트 조회.
 * @param out_tensor_ptr, index_ptr, dim, num_idx, cache_dim, key_off: read_feature_kernel 과 동일.
 * @param CPU_buffer:  GIDS_CPU_buffer<T>. device_cpu_buffer 가 host-pinned 공유 메모리 GPU 주소.
 * @param cpu_seq:     true=앞 K 노드를 CPU 버퍼에 선형 저장한 경우. false=희소 매핑(비트).
 * @param d_cpu_access:CPU 버퍼 히트 카운터(통계). atomicAdd 로 갱신.
 *
 * 각 warp 는 한 노드를 담당한다. cpu_seq 에 따라 두 가지 hit 판정:
 *   (a) cpu_seq=true : row_index < cpu_buffer_len 이면 hit.
 *   (b) cpu_seq=false: cpu_off & 1 == 1 이면 hit, 상위 31비트가 CPU 버퍼 슬롯 인덱스.
 * hit 시 host-pinned 버퍼에서 읽고 SSD 접근 회피. miss 시 ptr.read() 로 BaM 경로.
 *
 * atomicAdd(d_cpu_access, 1): warp 대표(tid==0) 하나만 1 씩 증가시켜 CPU hit 수 집계.
 *   통계 목적이므로 순서 보장 불필요, GPU atomic 으로 경합 해소.
 *
 * 호출 체인:
 *   BAM_Feature_Store::read_feature (cpu_buffer_flag=true) → [이 커널]
 *     → (hit) host-pinned 메모리 직접 load / (miss) bam_ptr.read → NVMe.
 */
template <typename T = float>
__global__ void read_feature_kernel_with_cpu_backing_memory(array_d_t<T> *dr, range_d_t<T> *range, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, GIDS_CPU_buffer<T> CPU_buffer, bool cpu_seq, unsigned int* d_cpu_access, uint64_t key_off) {

  uint64_t bid = blockIdx.x;                   /* [한국어] block 인덱스 — 블록 내부 warp 들이 여러 노드를 나눠 담당. */

  int num_warps = blockDim.x / 32;             /* [한국어] 블록 내 warp 개수. */
  int warp_id = threadIdx.x / 32;              /* [한국어] 블록 내 warp 번호. */
  int idx_idx = bid * num_warps + warp_id;     /* [한국어] 전역 warp 인덱스 = 담당 node 의 배치 인덱스. */
  if (idx_idx < num_idx) {                     /* [한국어] 오버슈팅 warp 차단. */
 	    bam_ptr<T> ptr(dr);                   /* [한국어] BaM accessor 준비(miss 시 사용). */

      uint64_t row_index = index_ptr[idx_idx] + key_off;   /* [한국어] 노드 ID + 타입 기준 오프셋. */
      uint64_t tid = threadIdx.x % 32;                     /* [한국어] warp 레인 인덱스. */

      uint32_t cpu_off = range -> get_cpu_offset(row_index);
      /* [한국어] range_d_t 의 GPU 메서드 호출. 반환값 비트0 = CPU 버퍼 유무,
       *          비트1..31 = CPU 버퍼 내 슬롯 인덱스. set_cpu_buffer_kernel 이 설정함. */


      if(cpu_seq){                             /* [한국어] 선형 매핑 모드: row_index 가 곧 CPU 슬롯. */
        if(row_index < CPU_buffer.cpu_buffer_len){ /* [한국어] 범위 내 → CPU hit. */
          if(tid == 0)                         /* [한국어] warp 당 1회만 통계 증가(중복 집계 방지). */
            atomicAdd(d_cpu_access, 1);         /* [한국어] 통계 용도 atomic. 순서/정합성 요구 없음. */
          for (; tid < dim; tid += 32) {        /* [한국어] warp 협력으로 dim 전체 복사. */
            T temp = CPU_buffer.device_cpu_buffer[(row_index) * cache_dim + tid];
            /* [한국어] host-pinned 메모리를 PCIe 로 직접 읽음. SSD 접근 없음. */
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            /* [한국어] 결과를 GPU 출력 텐서에 저장. */
            }
        }

        else{                                   /* [한국어] CPU 버퍼 범위 밖 → SSD fetch 경로. */
        for (; tid < dim; tid += 32) {
          //T temp = ptr[(row_index) * cache_dim + tid];
          T temp = ptr.read((row_index) * cache_dim + tid);
          /* [한국어] bam_ptr.read → BaM page cache → miss 시 nvm_queue.sq_enqueue. */
          out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
      }
      }
      else{                                     /* [한국어] 희소 매핑 모드: get_cpu_offset 비트로 판정. */
        if((cpu_off & 0x1) == 1){                /* [한국어] 비트0=1 이면 CPU hit. */
          if(tid == 0)                          /* [한국어] warp 대표만 카운트 증가. */
            atomicAdd(d_cpu_access, 1);          /* [한국어] 통계 전용 atomic. */

            for (; tid < dim; tid += 32) {
              T temp = CPU_buffer.device_cpu_buffer[(cpu_off >> 1) * cache_dim + tid];
              /* [한국어] (cpu_off>>1) = CPU 버퍼 슬롯 번호. 해당 슬롯에서 dim 분량을 읽음. */
              out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            }
        }

        else{                                   /* [한국어] 비트0=0 → SSD 경로. */
          for (; tid < dim; tid += 32) {
            //T temp = ptr[(row_index) * cache_dim + tid];
            T temp = ptr.read((row_index) * cache_dim + tid);
            /* [한국어] BaM 페이지 캐시 경유 SSD 읽기. */
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
          }
        }
      }
  }
}


/*
 * [한국어]
 * set_cpu_buffer_kernel - 핫 노드 ID 배열을 받아 range_d_t 의 매핑 테이블에
 *                        "이 node 는 CPU 버퍼 슬롯 idx 에 있다" 를 기록.
 *
 * @param d_range: range_d_t<T>*. set_cpu_buffer(row, slot) 로 비트 기록.
 * @param idx_ptr: 핫 노드 ID 배열(GPU).
 * @param num:     배열 길이.
 * @param pageSize:페이지 크기(현재 함수에서는 사용되지 않음, 향후 확장용).
 *
 * block/thread 매핑: idx = tid + bid*blockDim → 배열 선형 인덱스.
 * 호출 체인: BAM_Feature_Store::set_cpu_buffer → [이 커널] → range_d_t::set_cpu_buffer.
 */
template <typename T = float>
__global__ void set_cpu_buffer_kernel(range_d_t<T> *d_range, uint64_t* idx_ptr, int num, uint32_t pageSize) {

  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;   /* [한국어] 전역 스레드 = 담당할 핫 노드 인덱스. */
  if(idx <  num){                                          /* [한국어] 오버슈팅 차단. */
    d_range -> set_cpu_buffer(idx_ptr[idx], idx );         /* [한국어] (row=idx_ptr[idx], slot=idx) 를 range 매핑에 기록. */
  }

}

/*
 * [한국어]
 * set_cpu_buffer_data_kernel - 각 핫 노드의 실제 feature 데이터를 SSD→CPU 버퍼로 복사.
 *
 * @param dr:         BaM array device 핸들. SSD fetch 경로 제공.
 * @param CPU_buffer: host-pinned 영역의 device 포인터(GIDS_CPU_buffer::device_cpu_buffer).
 * @param idx_ptr:    핫 노드 ID 배열.
 * @param dim:        노드 당 원소 수.
 * @param num:        노드 수.
 *
 * 블록 ↔ 노드 매핑: blockIdx.x = 핫 노드 슬롯 인덱스. 블록 내 스레드가 dim 차원을 분할.
 * ptr[i] 는 BaM 의 read 와 동치(bam_ptr 의 subscript). page_cache miss 시 NVMe fetch.
 *
 * 호출 체인: BAM_Feature_Store::set_cpu_buffer → [이 커널] → bam_ptr[] → page_cache → NVMe.
 */
template <typename T = float>
__global__ void set_cpu_buffer_data_kernel(array_d_t<T> *dr, T* CPU_buffer, uint64_t* idx_ptr, uint64_t dim, int num) {
	uint64_t bid = blockIdx.x;               /* [한국어] 이 블록이 담당할 "핫 노드 슬롯" 번호. */
	bam_ptr<T> ptr(dr);                       /* [한국어] BaM accessor — SSD feature 를 읽어 올 수단. */
	if(bid <  num){                            /* [한국어] 유효 범위 검사. */
		uint64_t idx = idx_ptr[bid];          /* [한국어] 실제 node ID (SSD 상 논리 행). */

		for(uint64_t i  = threadIdx.x; i < dim; i += blockDim.x){   /* [한국어] 블록 스레드가 차원 분할 처리. */
			CPU_buffer[bid * dim + i] = ptr[idx * dim + i];         /* [한국어] SSD→CPU 영속 사본 기록. */
		}
	}

}


/*
 * [한국어]
 * set_window_buffering_kernel - 다음 배치에 필요한 페이지를 미리 BaM 페이지 캐시에 앉힌다.
 *
 * @param dr:        BaM array device 핸들.
 * @param index_ptr: 프리패치할 페이지 인덱스 배열.
 * @param page_size: 한 페이지의 바이트 수.
 * @param hash_off:  heterograph 에서 노드 타입별 해시 기준점.
 *
 * block ↔ 페이지 매핑: blockIdx.x = 프리패치 대상 페이지 번호. 각 block 의 첫 스레드
 * 만이 bam_ptr.set_window_buffer_counter 를 호출해 해당 페이지의 카운터를 증가시킨다.
 * 카운터 증가는 BaM 페이지 캐시가 "이 페이지를 확보(필요시 SSD fetch)" 하도록 유도한다.
 *
 * 호출 체인: BAM_Feature_Store::set_window_buffering → [이 커널] → bam_ptr.set_window_buffer_counter.
 */
template <typename T = float>
__global__
void set_window_buffering_kernel(array_d_t<T>* dr, uint64_t *index_ptr, uint64_t page_size, int hash_off){
	bam_ptr<T> ptr(dr);                           /* [한국어] BaM accessor 준비. */
	if(threadIdx.x == 0){                         /* [한국어] 페이지 당 1개 스레드로 중복 트리거 방지. */
		uint64_t page_idx = index_ptr[blockIdx.x] + hash_off;  /* [한국어] 프리패치 대상 page ID + 해시 오프셋. */
		ptr.set_window_buffer_counter(page_idx * page_size/sizeof(T), 1);
		/* [한국어] BaM 의 페이지 캐시에 '이 원소가 속한 페이지를 미리 fetch/pin' 요청.
		 *          page_idx * (page_size/sizeof(T)) = 페이지의 첫 원소 인덱스. */
	}
}

/*
 * [한국어]
 * read_kernel - 디버그용 단일 스레드 순차 읽기 + printf 덤프(정수형 캐스팅).
 *
 * @param dr:     BaM array device 핸들.
 * @param num:    읽을 원소 수.
 * @param offset: 시작 원소 인덱스.
 *
 * 단일 스레드(thread (0,0))만이 for-loop 로 num 개 원소를 순차 읽어 printf 로 출력.
 * 성능 목적이 아니라 BaM 경로/데이터 샘플 확인용. read_tensor 에서는 사용되지 않고
 * seq_read_kernel 이 실제 호출됨(본 함수는 대안 변형).
 */
template <typename T = float>
__global__ void read_kernel(array_d_t<T> *dr,
                                    uint64_t num, uint64_t offset) {
      bam_ptr<T> ptr(dr);                                       /* [한국어] BaM accessor. */
     if(threadIdx.x == 0 && blockIdx.x == 0){                    /* [한국어] 오직 첫 스레드만 실행하여 중복 출력 방지. */
        for(uint64_t i = 0; i < num; i++){                        /* [한국어] num 개 원소 순차 접근. */
              if(i == 0) printf("idx: %llu type size:%i \n", offset,  (int) sizeof(T));
              /* [한국어] 첫 이터레이션에만 시작 오프셋과 원소 크기 로깅. */
             // T temp = ptr[i + offset];
              printf("read data: %llu\n",  (unsigned long long) ptr[i + offset]);
              /* [한국어] bam_ptr[] 가 BaM 경로로 원소를 읽어 반환, 정수형 형식으로 출력. */
             // printf("float read data: %f\n", temp);

        }
     }
}


/*
 * [한국어]
 * seq_read_kernel - 디버그용 순차 읽기(float 출력 버전). BAM_Feature_Store::read_tensor 가 호출.
 *
 * @param dr:     BaM array device 핸들.
 * @param num:    읽을 원소 수.
 * @param offset: 시작 원소 인덱스.
 *
 * 단일 스레드가 num 개 원소를 BaM 로 읽어 float 로 캐스팅 출력. bam_ptr[] 접근마다
 * 페이지 캐시 검색 및 필요 시 NVMe fetch 가 발생한다. 실제 학습 경로에서는 사용되지
 * 않고 데이터 검증 전용.
 */
template <typename T = float>
__global__ void seq_read_kernel(array_d_t<T> *dr,
                                    uint64_t num, uint64_t offset) {
    bam_ptr<T> ptr(dr);                                         /* [한국어] BaM accessor. */
     if(threadIdx.x == 0 && blockIdx.x == 0){                    /* [한국어] 단일 스레드 실행. */
        for(uint64_t i = 0; i < num; i++){                        /* [한국어] 순차 읽기. */
             // if(i == 0) printf("idx: %llu type size:%i \n", offset,  (int) sizeof(T));
              T temp = ptr[i + offset];                           /* [한국어] (현재는 사용되지 않으나) 읽기 동기 확보. */
              //printf("read data: %llu\n",  (unsigned long long) ptr[i + offset]);
              printf("read data: %f\n",  (float) ptr[i + offset]);
              /* [한국어] 두 번째 ptr[] 는 같은 페이지를 다시 읽음(캐시 hit 기대). float 로 출력. */
             // printf("float read data: %f\n", temp);

        }
     }
}


/*
 * [한국어]
 * write_feature_kernel - (레거시) 페이지 단위로 SSD 스트라이핑 쓰기 경로.
 *
 * @param ctrls:      BaM Controller 포인터 배열(GPU).
 * @param pc:         page_cache_d_t*. write_data 의 출발 캐시.
 * @param dr:         BaM array device 핸들(현 함수에서는 직접 사용 안 함).
 * @param in_tensor_ptr: 입력 feature 포인터(참고용, write_data 에서는 pc 내부를 쓴다).
 * @param num:        기록할 페이지 수.
 * @param page_size:  페이지 바이트.
 * @param o_offset:   최상위 오프셋(사용자 설정).
 * @param s_offset:   스텝(반복) 오프셋 — store_tensor 의 옛 반복 루프에서 사용.
 * @param num_ctrls:  스트라이핑 SSD 개수.
 *
 * tid 를 SSD/큐/페이지로 분해해 각 SSD 의 특정 큐에 블록 write 를 발행. 현재
 * store_tensor 는 대신 write_feature_kernel2 를 사용하므로 본 함수는 주석된 레거시
 * 경로(혹은 향후 확장용)이다. block_size_log 는 NVMe LBA 크기의 log2.
 *
 * 호출 체인(과거): store_tensor (주석 처리된 루프) → [이 커널] → write_data → nvm_queue.
 */
template <typename T = float>
__global__ void write_feature_kernel(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr,
                                    uint64_t num, uint64_t page_size,  uint64_t o_offset,  uint64_t s_offset, uint32_t num_ctrls) {

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;    /* [한국어] 전역 스레드 = 페이지 × SSD 조합 인덱스. */
    uint32_t ctrl = (tid) % (num_ctrls);                      /* [한국어] 이 스레드가 쓸 Controller 선택(스트라이핑). */
    uint64_t pc_idx = tid / num_ctrls;                        /* [한국어] 해당 SSD 내에서 몇 번째 페이지인지. */

    uint32_t queue = (tid) % (ctrls[ctrl]->n_qps);            /* [한국어] 해당 Controller 의 I/O 큐 선택(경합 분산). */

    if(tid < num){                                            /* [한국어] 오버슈팅 방지. */
    	uint64_t start_block = ((o_offset+s_offset + pc_idx*page_size)) >> ctrls[ctrl]->d_qps[queue].block_size_log ;
    	/* [한국어] 바이트 오프셋을 해당 큐의 LBA 단위로 변환(>>log2). */

    	uint64_t n_blocks = page_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
    	/* [한국어] 페이지 크기를 LBA 개수로 변환. */
    	write_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
    	/* [한국어] BaM write_data: 해당 큐 쌍에 NVMe Write SQE 제출 + CQE 폴링. */
    }
}

/*
 * [한국어]
 * write_feature_kernel2 - store_tensor 가 실제로 사용하는 쓰기 경로.
 *
 * @param ctrls:         (미사용, 시그니처 호환용) Controller 배열.
 * @param pc:            (미사용) page_cache_d_t*.
 * @param dr:            BaM array device 핸들.
 * @param in_tensor_ptr: 기록할 원본 feature 버퍼(GPU 주소).
 * @param dim:           노드 당 원소 수.
 * @param num_ctrls:     (미사용).
 * @param offset:        원본 버퍼 내 시작 원소 오프셋(store_tensor 에서 offset/sizeof(TYPE)).
 *
 * block ↔ 노드 매핑: blockIdx.x = 기록할 노드 index. 블록 내 스레드가 dim 을 분할.
 * bam_ptr 의 write subscript( ptr[i] = value )가 page_cache 를 dirty 로 표시하고
 * flush_cache 시점에 NVMe Write 로 반영된다.
 *
 * 호출 체인: BAM_Feature_Store::store_tensor → [이 커널] → bam_ptr[]= → page_cache dirty
 *            → flush_cache → nvm_queue Write SQE.
 */
template <typename T = float>
__global__ void write_feature_kernel2(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr, uint64_t dim, uint32_t num_ctrls, uint64_t offset) {


	bam_ptr<T> ptr(dr);                                  /* [한국어] BaM accessor (쓰기 경로 포함). */
	uint64_t row_index = blockIdx.x;                     /* [한국어] 이 블록이 담당할 노드 번호. */

	for(int i = threadIdx.x; i < dim; i += blockDim.x){   /* [한국어] 블록 스레드 협력으로 dim 차원 분할. */
		ptr[(row_index) * dim + i] = in_tensor_ptr[(row_index) * dim + i + offset];
		/* [한국어] 입력 텐서 → BaM array(페이지 캐시 내부) 쓰기. 페이지가 dirty 로 마킹됨. */
	}
}




