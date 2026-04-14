/*
 * [한국어] cmake-cpp-pybind11 템플릿의 예제 클래스 선언 (example.h)
 *
 * === 파일의 역할 ===
 * pybind11 + CMake 템플릿이 동작하는지 확인하기 위한 최소 C++ 클래스 하나를 선언한다.
 * 실제 GIDS 런타임(특성 fetch, BaM page cache)과는 무관한 데모 자산이며, 빌드
 * 스켈레톤 호환성 유지를 위해 보존된다. double 스칼라 하나를 감싸고 += 연산과
 * print() 출력만 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GIDS 데이터 로딩/학습 경로(gids_nvme.cu → gids_kernel.cu → BaM)에는 전혀
 * 참여하지 않는다. gids_module/example/ 하위의 pybind11 확장(example.so)이
 * 이 클래스를 파이썬에 노출하기 위해서만 include 된다. 실행 컨텍스트는
 * 호스트 유저스페이스 C++.
 *
 * === 타 모듈과의 연결 ===
 * 의존 없음(표준 라이브러리조차 include 하지 않음 — 구현부에서 iostream 사용 가정).
 * 상위로는 example/ pybind11 바인딩 TU가 이 헤더를 사용한다.
 *
 * === 주요 함수/구조체 요약 ===
 * - Example(double): 스칼라 값으로 인스턴스 초기화.
 * - operator+=(const Example&): 내부 _a 에 피연산자 _a 를 더함(in-place).
 * - print(): _a 를 표준 출력으로 찍음(정의는 별도 .cpp/.cu).
 */

class Example
{
public:
  /*
   * [한국어]
   * Example(double) - 스칼라 한 개로 인스턴스를 초기화하는 생성자.
   *
   * @param a: 내부 상태 _a 에 복사될 실수 값. 호출자(Python 쪽 py::init<double>)
   *           가 Py-float 를 그대로 전달.
   *
   * 초기화 리스트로 _a(a) 를 수행 — 사본 1회만 발생. 재진입/스레드 안전성은
   * 고려 대상 아님(단일 객체 단일 스레드에서만 사용되는 데모 코드).
   *
   * 호출 체인:
   *   example.Example(1.0) → pybind11::init<double> → [이 생성자]
   */
  Example( double a )
    : _a( a)                /* [한국어] 멤버 이니셜라이저로 private 필드 _a 를 인자로 복사. */
  {
    /* [한국어] 본체는 비어 있음 — 모든 초기화는 이니셜라이저 리스트에서 완료. */
  }

  /*
   * [한국어]
   * operator+= - 다른 Example 의 내부 값을 자신에 합산한다 (in-place 덧셈).
   *
   * @param other: 우변 Example. const 참조 — 수정 불가, 복사 비용 없음.
   * @return: *this 에 대한 비-const 참조. 체이닝(a += b += c) 가능하도록 표준 관례 준수.
   *
   * 동작: _a += other._a (double 합). 캡슐화 위반 없이 같은 클래스이므로 private
   * 필드에 직접 접근 가능. 실행 컨텍스트는 단일 스레드 데모 용도.
   *
   * 호출 체인:
   *   파이썬 `x += y` (pybind11 self += other 바인딩) → [이 연산자]
   */
  Example& operator+=( const Example& other )
  {
    _a += other._a;           /* [한국어] 두 Example 의 내부 double 을 합산해 좌변 상태를 갱신. */
    return *this;             /* [한국어] 연산자 체이닝을 위해 자신의 참조 반환. */
  }

  /*
   * [한국어]
   * print - 내부 _a 값을 표준 출력으로 내보낸다.
   *
   * @return: 없음(void).
   *
   * 선언만 존재 — 정의는 별도의 TU(보통 example.cpp)에 있다. cout/printf 중
   * 어느 쪽을 쓰는지는 구현에서 정의됨. 단일 스레드 호출만 가정.
   */
  void print();
private:
  double _a;
  /* [한국어] 이 Example 인스턴스가 감싸는 실수 상태.
   * 설정자: 생성자와 operator+= 만 수정.
   * 읽는 자: print() 가 출력 용도로 읽음.
   * 값 범위: IEEE 754 double 전 범위. NaN/Inf 체크 없음.
   * 동기화: 데모용 싱글스레드 객체 — 별도 락 없음. */
};


