/*
 * [한국어] 호스트 메모리 락 유틸리티 (lock_mem.cpp)
 *
 * === 파일의 역할 ===
 * 512 GiB 규모의 호스트 메모리를 malloc으로 확보한 뒤 mlockall(MCL_CURRENT)
 * 로 swap out을 방지한 채로 보유해, GIDS/BaM 실험 시 커널이 page-cache
 * 압박을 이유로 무관한 유저 프로세스 페이지를 swap 하지 않도록 "메모리
 * 공간을 묶어두는" 보조 도구다. 표준 입력을 대기하여 운용자가 ctrl/enter
 * 로 종료 타이밍을 잡는다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * evaluation 에서 BaM 단독/GIDS full 벤치를 돌릴 때, 별도 터미널에서 이
 * 바이너리를 띄워 호스트 메모리 압박을 인위적으로 만들어 페이지 캐시/UVA
 * 경합 영향을 배제한다. NVMe I/O 경로 자체와는 무관하며, 운용 편의 도구다.
 *
 * === 타 모듈과의 연결 ===
 * 외부 라이브러리 의존 없음(glibc only). BaM/DGL/PyTorch와 직접 링크되지
 * 않고, 단지 동일 호스트에 공존하면서 OS 메모리 통계에 영향을 준다.
 * mlockall은 RLIMIT_MEMLOCK 한도와 CAP_IPC_LOCK 권한을 필요로 한다.
 *
 * === 주요 함수/구조체 요약 ===
 *   main(): 512 GiB 할당 → mmap(size 미설정 버그성 호출) → mlockall →
 *           stdin 대기 → munlockall. 학습 전후 수동으로 실행.
 *
 * 주의: 원본 코드에는 `size` 변수가 미초기화 상태로 mmap에 전달되고 `fd`
 * 역시 초기화 없이 쓰이는 잠재적 문제가 있으나, 본 작업 규칙(코드 수정
 * 금지)에 따라 주석으로만 명시하고 그대로 둔다.
 */

/* This contains the mmap calls. */
#include <sys/mman.h>  /* [한국어] mmap/mlockall/munlockall 선언. POSIX 가상메모리 API. */
/* These are for error printing. */
#include <errno.h>    /* [한국어] errno 전역 변수 - 실패 원인 코드 확인용. */
#include <string.h>   /* [한국어] strerror 등 C 문자열/메모리 함수. */
#include <stdarg.h>   /* [한국어] 가변 인자 매크로 (본 파일에선 직접 사용 X). */
/* This is for open. */
#include <fcntl.h>    /* [한국어] open()/O_* 플래그. mmap 대상 fd 준비용으로 include. */
#include <stdio.h>    /* [한국어] printf 계열. */
#include <stdlib.h>   /* [한국어] malloc/free/exit. 512GiB 예약에 malloc 사용. */
#include <unistd.h>   /* [한국어] close/read/write/sysconf. */
#include <sys/mman.h> /* [한국어] 중복 include — 원본 유지. mlockall/munlockall 선언. */
#include <iostream>   /* [한국어] std::cout / std::cin - 상태 출력·대기. */


/*
 * [한국어]
 * main - 512 GiB 호스트 메모리를 malloc 후 mlockall로 고정한 채 사용자 입력
 *        을 대기하는 단일-진입점 함수.
 *
 * @return: 정상 종료 시 0. 현재 원본은 명시적 return 없음 (C++에서 main은 암묵적 0).
 *
 * 실행 컨텍스트: 유저스페이스 단일 스레드. 실행 중 mlockall 에 의해 프로세스 전
 *    가상주소가 swap out 방지 상태가 된다. 호출자는 사람(셸).
 * 호출 체인: shell → main() → malloc → mmap → mlockall → std::cin → munlockall.
 * 에러 경로: mlockall 실패 시 ret < 0 가 출력되며, 이후 munlockall 이 그대로
 *    호출되어도 no-op 에 가까운 처리가 된다. 근본 원인은 RLIMIT_MEMLOCK 부족.
 */
int main() {
        int fd;                 /* [한국어] 파일 디스크립터 — 선언만 있고 초기화/open 없음.
                                 *          원본의 잠재적 버그. mmap 에 미초기화 값이 전달됨. */
        struct stat s;          /* [한국어] stat 결과 저장용. 실제로는 사용되지 않음(dead var). */
        size_t size;            /* [한국어] mmap 길이 — 초기화 없이 mmap에 전달. 의도된 동작 아님. */

        const char* mapped;     /* [한국어] mmap 반환 포인터 저장용. */
	char c;                 /* [한국어] std::cin >> c 로 종료 시그널(임의 키 입력)을 받기 위한 버퍼. */

	//Locking 512GB CPU memory
        /* [한국어] 512 GiB = 512 × 1024 MiB × 1024 KiB × 1024 B 를 LL로 계산.
         *          32-bit 오버플로 방지용 LL 접미사 필수. */
        size_t mem_lock = (512LL ) * 1024LL * 1024 * 1024;
        /* [한국어] 512 GiB 가상 메모리 예약. 실제 물리 페이지 할당은 이후 touch 시점
         *          또는 mlockall 호출에 의해 강제된다. malloc 실패 시 mem_p=NULL. */
        char* mem_p = (char*) malloc(mem_lock);

        /* [한국어] mmap 호출 — 파라미터가 미초기화(size, fd)라 실효성이 모호하며,
         *          원본 코드의 남은 실험 흔적으로 보인다. PROT_READ|PROT_WRITE,
         *          MAP_SHARED 로 매핑 시도. 실제 락 동작은 아래 mlockall 담당. */
        mapped = (char*) mmap (0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        //int ret = mlock(mem_p, mem_lock);
        /* [한국어] mlockall(MCL_CURRENT) — 현재 프로세스가 매핑한 모든 페이지를
         *          물리메모리에 고정해 swap 대상에서 제외. 미래 매핑 페이지까지
         *          포함하려면 MCL_FUTURE 를 OR 해야 하나 여기선 CURRENT 만. */
        int ret = mlockall(MCL_CURRENT);
        std::cout << "ret: " << ret << std::endl;   /* [한국어] 성공 0, 실패 시 -1. */
        std::cout  << "Done locking! ";              /* [한국어] 사용자에게 락 완료 알림. */
        std::cin >> c;                                /* [한국어] 아무 키 입력 전까지 대기 — 락 해제 타이밍 제어. */

        munlockall();//(mapped, size);
        /* [한국어] munlockall — 위에서 건 모든 페이지 락을 해제. 프로세스 종료 시에도
         *          자동 해제되지만 명시적 호출로 의도 표현. 원본 주석은 mlock 시절 흔적. */
}

