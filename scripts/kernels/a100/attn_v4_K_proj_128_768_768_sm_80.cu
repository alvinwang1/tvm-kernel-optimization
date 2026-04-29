#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

struct __align__(8) half4 {
  __half x, y, z, w;
  __host__ __device__ half4() : x(__half(0)), y(__half(0)), z(__half(0)), w(__half(0)) {}
  __host__ __device__ half4(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}

};
__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) {
    return half4(x, y, z, w);
}
#include <mma.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_reindex_shared_dyn_wmma_accumulator[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_reindex_shared_dyn_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_reindex_shared_dyn_wmma_matrix_b[4];
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 4) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 4) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 8) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 8) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 800)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 3072));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 16) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 16) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 20) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 20) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 1600)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 6144));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 28) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 28) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 32) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 32) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 2400)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 9216));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 40) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 40) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 44) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 44) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 3200)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 12288));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 52) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 52) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 56) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 56) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 4000)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 15360));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 64) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 64) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 68) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 68) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 4800)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 18432));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 76) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 76) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 80) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 80) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 5600)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 21504));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 88) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 88) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 92) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + ((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 92) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 6400)) = *(half4*)(B + (((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 7040)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 12288));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 7680)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 24576));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 8320)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 36864));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 8960)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 49152));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 9600)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 61440));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 10240)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 73728));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 10880)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 86016));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 11520)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 98304));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 12160)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 110592));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 12800)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 122880));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 13440)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 135168));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 3200)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 16)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 32)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 48)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 6400)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 7040)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 7680)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 8320)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 64)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 80)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 96)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 112)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 8960)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 9600)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 10240)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 10880)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 128)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 144)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 160)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 176)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 11520)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 12160)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 12800)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 13440)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2))) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 4) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 4) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 8) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 8) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 800)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 3264));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 16) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 16) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 20) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 20) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 1600)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 6336));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 28) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 28) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 32) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 32) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 2400)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 9408));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 40) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 40) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 44) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 44) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 3200)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 12480));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 52) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 52) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 56) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 56) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 4000)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 15552));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 64) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 64) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 68) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 68) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 4800)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 18624));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 76) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 76) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 80) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 80) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 5600)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 21696));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 88) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 88) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 192));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 92) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 92) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 192));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 6400)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 147456));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 7040)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 159744));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 7680)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 172032));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 8320)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 184320));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 8960)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 196608));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 9600)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 208896));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 10240)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 221184));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 10880)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 233472));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 11520)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 245760));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 12160)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 258048));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 12800)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 270336));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 13440)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 282624));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 3200)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 16)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 32)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 48)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 6400)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 7040)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 7680)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 8320)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 64)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 80)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 96)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 112)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 8960)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 9600)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 10240)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 10880)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 128)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 144)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 160)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 176)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 11520)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 12160)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 12800)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 13440)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2))) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 4) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 4) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 8) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 8) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 800)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 3456));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 16) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 16) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 20) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 20) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 1600)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 6528));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 28) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 28) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 32) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 32) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 2400)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 9600));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 40) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 40) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 44) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 44) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 3200)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 12672));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 52) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 52) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 56) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 56) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 4000)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 15744));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 64) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 64) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 68) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 68) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 4800)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 18816));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 76) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 76) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 80) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 80) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 5600)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 21888));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 88) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 88) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 384));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 92) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 92) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 384));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 6400)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 294912));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 7040)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 307200));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 7680)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 319488));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 8320)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 331776));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 8960)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 344064));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 9600)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 356352));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 10240)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 368640));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 10880)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 380928));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 11520)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 393216));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 12160)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 405504));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 12800)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 417792));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 13440)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 430080));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 3200)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 16)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 32)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 48)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 6400)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 7040)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 7680)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 8320)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 64)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 80)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 96)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 112)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 8960)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 9600)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 10240)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 10880)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 128)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 144)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 160)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 176)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 11520)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 12160)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 12800)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 13440)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2))) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 4) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 4) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 8) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 8) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 800)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 3648));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 16) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 16) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 20) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 20) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 1600)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 6720));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 28) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 28) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 32) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 32) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 2400)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 9792));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 40) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 40) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 44) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 44) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 3200)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 12864));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 52) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 52) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 56) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 56) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 4000)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 15936));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 64) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 64) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 68) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 68) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 4800)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 19008));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 76) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 76) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 80) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 80) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((((int)threadIdx.y) / 3) * 200) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 5600)) = *(half2*)(A + ((((((((int)blockIdx.x) / 3) * 24576) + ((((int)threadIdx.y) / 3) * 768)) + ((((int)threadIdx.y) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 22080));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 88) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 88) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 64) % 192)) + 576));
  *(half2*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) + 92) / 3) * 200) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192))) = *(half2*)(A + (((((((int)blockIdx.x) / 3) * 24576) + (((((int)threadIdx.y) + 92) / 3) * 768)) + ((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128) % 192)) + 576));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 6400)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 442368));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 7040)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 454656));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 7680)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 466944));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 8320)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 479232));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 8960)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 491520));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 9600)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 503808));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 10240)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 516096));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 10880)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 528384));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 11520)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 540672));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 12160)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 552960));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 12800)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 565248));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)) + 13440)) = *(half4*)(B + ((((((((int)threadIdx.y) * 3072) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 577536));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 3200)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 16)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 32)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 48)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 6400)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 7040)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 7680)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 8320)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 64)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 80)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 96)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 112)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 8960)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 9600)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 10240)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 10880)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 128)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 144)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 160)])), 200);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 3200) + 176)])), 200);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 11520)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 12160)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 12800)])), 40);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 16) + 13440)])), 40);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 256)])), C_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(C + (((((((((int)blockIdx.x) / 3) * 24576) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 2))) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(C + ((((((((((int)blockIdx.x) / 3) * 24576) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 16)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(C + ((((((((((int)blockIdx.x) / 3) * 24576) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 12288)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(C + ((((((((((int)blockIdx.x) / 3) * 24576) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + (((int)blockIdx.y) * 96)) + ((((int)blockIdx.x) % 3) * 32)) + ((((int)threadIdx.x) & 7) * 2)) + 12304)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
}

