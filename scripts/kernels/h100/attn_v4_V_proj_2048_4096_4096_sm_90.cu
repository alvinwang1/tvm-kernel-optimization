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
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C);
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_reindex_shared_dyn_wmma_accumulator[32];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_reindex_shared_dyn_wmma_matrix_a[32];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_reindex_shared_dyn_wmma_matrix_b[16];
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[2], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[3], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[4], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[5], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[6], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[7], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[8], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[9], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[10], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[11], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[12], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[13], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[14], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[15], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[16], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[17], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[18], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[19], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[20], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[21], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[22], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[23], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[24], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[25], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[26], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[27], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[28], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[29], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[30], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[31], 0.000000e+00f);
  for (int ax2_0_0 = 0; ax2_0_0 < 64; ++ax2_0_0) {
    __syncthreads();
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 8704)) = *(half4*)(A + (((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 8992)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 16384));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 9280)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 32768));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 9568)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 49152));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 9856)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 65536));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 10144)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 81920));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 10432)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 98304));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 10720)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 114688));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 11008)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 131072));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 11296)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 147456));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 11584)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 163840));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 11872)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 180224));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 12160)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 196608));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 12448)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 212992));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 12736)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 229376));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 13024)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 245760));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 13312)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 262144));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 13600)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 278528));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 13888)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 294912));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 14176)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 311296));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 14464)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 327680));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 14752)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 344064));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 15040)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 360448));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 15328)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 376832));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 15616)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 393216));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 15904)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 409600));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 16192)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 425984));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 16480)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 442368));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 16768)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 458752));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 17056)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 475136));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 17344)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 491520));
    *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 17632)) = *(half4*)(A + ((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 507904));
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(B + ((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 544)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 16384));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 1088)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 1632)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 49152));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 2176)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 2720)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 81920));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 3264)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 98304));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 3808)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 114688));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 4352)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 4896)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 147456));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 5440)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 163840));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 5984)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 180224));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 6528)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 196608));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 7072)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 212992));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 7616)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 229376));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 8160)) = *(uint4*)(B + (((((((ax2_0_0 * 262144) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 245760));
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[8704])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[8720])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[8736])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[8752])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[4], (&(((half*)buf_dyn_shmem)[9856])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[5], (&(((half*)buf_dyn_shmem)[9872])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[6], (&(((half*)buf_dyn_shmem)[9888])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[7], (&(((half*)buf_dyn_shmem)[9904])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[8], (&(((half*)buf_dyn_shmem)[11008])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[9], (&(((half*)buf_dyn_shmem)[11024])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[10], (&(((half*)buf_dyn_shmem)[11040])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[11], (&(((half*)buf_dyn_shmem)[11056])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[12], (&(((half*)buf_dyn_shmem)[12160])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[13], (&(((half*)buf_dyn_shmem)[12176])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[14], (&(((half*)buf_dyn_shmem)[12192])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[15], (&(((half*)buf_dyn_shmem)[12208])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[16], (&(((half*)buf_dyn_shmem)[13312])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[17], (&(((half*)buf_dyn_shmem)[13328])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[18], (&(((half*)buf_dyn_shmem)[13344])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[19], (&(((half*)buf_dyn_shmem)[13360])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[20], (&(((half*)buf_dyn_shmem)[14464])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[21], (&(((half*)buf_dyn_shmem)[14480])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[22], (&(((half*)buf_dyn_shmem)[14496])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[23], (&(((half*)buf_dyn_shmem)[14512])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[24], (&(((half*)buf_dyn_shmem)[15616])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[25], (&(((half*)buf_dyn_shmem)[15632])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[26], (&(((half*)buf_dyn_shmem)[15648])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[27], (&(((half*)buf_dyn_shmem)[15664])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[28], (&(((half*)buf_dyn_shmem)[16768])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[29], (&(((half*)buf_dyn_shmem)[16784])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[30], (&(((half*)buf_dyn_shmem)[16800])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[31], (&(((half*)buf_dyn_shmem)[16816])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 64)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 16)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 32)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 48)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 2176)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 2192)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 2208)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 2224)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[8], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 4352)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[9], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 4368)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[10], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 4384)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[11], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 4400)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[12], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 6528)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[13], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 6544)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[14], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 6560)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[15], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 6576)])), 136);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[16], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[17], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[18], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[19], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[16], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[17], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[18], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[19], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[16], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[17], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[18], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[19], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[16], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[17], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[18], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[19], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[20], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[21], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[22], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[23], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[20], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[21], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[22], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[23], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[20], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[21], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[22], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[23], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[20], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[21], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[22], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[23], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[24], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[25], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[26], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[27], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[24], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[25], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[26], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[27], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[24], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[25], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[26], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[27], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[24], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[25], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[26], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[27], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[28], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[29], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[30], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[31], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[28], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[29], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[30], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[31], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[28], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[29], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[30], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[31], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[28], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[31]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[29], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[31]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[30], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[31]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[31], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[31]);
  }
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[1], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[2], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[3], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  C[((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15))] = ((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 16384)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 64)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 32768)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 128)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 49152)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 192)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 16)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 256)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 16400)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 320)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 32784)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 384)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 49168)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 448)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 32)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 512)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 16416)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 576)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 32800)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 640)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 49184)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 704)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 48)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 768)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 16432)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 832)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 32816)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 896)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 49200)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 960)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 64)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1024)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 16448)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1088)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 32832)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1152)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 49216)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1216)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 80)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1280)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 16464)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1344)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 32848)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1408)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 49232)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1472)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 96)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1536)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 16480)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1600)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 32864)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1664)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 49248)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1728)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 112)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1792)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 16496)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1856)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 32880)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1920)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 49264)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1984)];
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[4], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[5], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[6], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[7], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 65536)] = ((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 81920)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 64)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 98304)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 128)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 114688)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 192)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 65552)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 256)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 81936)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 320)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 98320)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 384)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 114704)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 448)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 65568)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 512)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 81952)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 576)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 98336)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 640)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 114720)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 704)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 65584)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 768)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 81968)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 832)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 98352)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 896)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 114736)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 960)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 65600)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1024)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 81984)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1088)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 98368)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1152)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 114752)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1216)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 65616)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1280)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 82000)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1344)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 98384)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1408)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 114768)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1472)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 65632)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1536)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 82016)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1600)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 98400)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1664)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 114784)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1728)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 65648)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1792)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 82032)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1856)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 98416)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1920)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 114800)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1984)];
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[8], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[9], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[10], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[11], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 131072)] = ((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 147456)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 64)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 163840)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 128)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 180224)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 192)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 131088)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 256)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 147472)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 320)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 163856)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 384)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 180240)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 448)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 131104)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 512)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 147488)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 576)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 163872)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 640)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 180256)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 704)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 131120)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 768)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 147504)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 832)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 163888)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 896)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 180272)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 960)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 131136)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1024)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 147520)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1088)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 163904)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1152)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 180288)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1216)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 131152)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1280)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 147536)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1344)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 163920)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1408)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 180304)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1472)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 131168)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1536)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 147552)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1600)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 163936)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1664)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 180320)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1728)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 131184)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1792)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 147568)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1856)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 163952)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1920)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 180336)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1984)];
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[12], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[13], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[14], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[15], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 196608)] = ((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 212992)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 64)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 229376)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 128)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 245760)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 192)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 196624)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 256)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 213008)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 320)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 229392)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 384)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 245776)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 448)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 196640)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 512)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 213024)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 576)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 229408)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 640)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 245792)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 704)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 196656)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 768)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 213040)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 832)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 229424)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 896)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 245808)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 960)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 196672)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1024)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 213056)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1088)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 229440)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1152)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 245824)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1216)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 196688)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1280)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 213072)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1344)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 229456)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1408)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 245840)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1472)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 196704)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1536)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 213088)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1600)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 229472)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1664)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 245856)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1728)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 196720)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1792)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 213104)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1856)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 229488)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1920)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 245872)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1984)];
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[16], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[17], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[18], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[19], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 262144)] = ((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 278528)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 64)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 294912)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 128)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 311296)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 192)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 262160)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 256)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 278544)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 320)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 294928)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 384)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 311312)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 448)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 262176)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 512)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 278560)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 576)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 294944)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 640)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 311328)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 704)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 262192)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 768)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 278576)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 832)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 294960)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 896)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 311344)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 960)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 262208)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1024)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 278592)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1088)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 294976)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1152)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 311360)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1216)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 262224)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1280)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 278608)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1344)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 294992)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1408)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 311376)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1472)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 262240)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1536)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 278624)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1600)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 295008)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1664)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 311392)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1728)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 262256)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1792)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 278640)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1856)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 295024)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1920)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 311408)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1984)];
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[20], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[21], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[22], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[23], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 327680)] = ((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 344064)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 64)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 360448)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 128)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 376832)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 192)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 327696)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 256)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 344080)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 320)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 360464)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 384)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 376848)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 448)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 327712)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 512)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 344096)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 576)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 360480)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 640)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 376864)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 704)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 327728)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 768)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 344112)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 832)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 360496)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 896)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 376880)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 960)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 327744)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1024)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 344128)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1088)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 360512)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1152)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 376896)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1216)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 327760)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1280)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 344144)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1344)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 360528)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1408)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 376912)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1472)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 327776)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1536)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 344160)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1600)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 360544)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1664)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 376928)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1728)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 327792)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1792)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 344176)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1856)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 360560)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1920)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 376944)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1984)];
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[24], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[25], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[26], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[27], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 393216)] = ((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 409600)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 64)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 425984)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 128)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 442368)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 192)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 393232)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 256)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 409616)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 320)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 426000)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 384)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 442384)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 448)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 393248)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 512)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 409632)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 576)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 426016)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 640)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 442400)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 704)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 393264)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 768)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 409648)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 832)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 426032)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 896)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 442416)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 960)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 393280)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1024)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 409664)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1088)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 426048)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1152)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 442432)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1216)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 393296)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1280)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 409680)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1344)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 426064)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1408)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 442448)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1472)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 393312)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1536)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 409696)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1600)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 426080)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1664)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 442464)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1728)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 393328)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1792)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 409712)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1856)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 426096)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1920)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 442480)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1984)];
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[28], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[29], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[30], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[31], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 458752)] = ((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 32) + ((int)threadIdx.x))];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 475136)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 64)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 491520)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 128)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 507904)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 192)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 458768)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 256)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 475152)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 320)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 491536)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 384)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 507920)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 448)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 458784)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 512)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 475168)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 576)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 491552)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 640)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 507936)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 704)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 458800)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 768)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 475184)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 832)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 491568)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 896)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 507952)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 960)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 458816)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1024)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 475200)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1088)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 491584)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1152)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 507968)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1216)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 458832)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1280)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 475216)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1344)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 491600)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1408)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 507984)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1472)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 458848)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1536)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 475232)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1600)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 491616)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1664)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 508000)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1728)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 458864)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1792)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 475248)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1856)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 491632)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1920)];
  C[(((((((((((int)blockIdx.y) >> 1) * 2097152) + ((((int)blockIdx.x) >> 4) * 524288)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 1) * 2048)) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) & 15)) + 508016)] = ((half*)buf_dyn_shmem)[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)) + 1984)];
}

