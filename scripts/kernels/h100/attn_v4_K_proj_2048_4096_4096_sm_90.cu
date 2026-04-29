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
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_reindex_shared_dyn_wmma_matrix_a[16];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_reindex_shared_dyn_wmma_matrix_b[8];
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[4], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[5], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[8], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[9], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[12], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[13], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[2], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[3], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[6], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[7], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[10], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[11], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[14], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[15], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[16], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[17], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[20], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[21], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[24], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[25], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[28], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[29], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[18], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[19], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[22], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[23], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[26], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[27], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[30], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[31], 0.000000e+00f);
  for (int ax2_0_0 = 0; ax2_0_0 < 128; ++ax2_0_0) {
    __syncthreads();
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4352)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 4992)) = *(uint4*)(A + ((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 65536));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 5632)) = *(uint4*)(A + ((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 131072));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6272)) = *(uint4*)(A + ((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 196608));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 6912)) = *(uint4*)(A + ((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 262144));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 7552)) = *(uint4*)(A + ((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 327680));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8192)) = *(uint4*)(A + ((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 393216));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)) + 8832)) = *(uint4*)(A + ((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 32768)) + ((((int)threadIdx.x) >> 2) * 4096)) + (ax2_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 458752));
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(B + ((((((ax2_0_0 * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 544)) = *(uint4*)(B + (((((((ax2_0_0 * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 16384));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 1088)) = *(uint4*)(B + (((((((ax2_0_0 * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 1632)) = *(uint4*)(B + (((((((ax2_0_0 * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 49152));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 2176)) = *(uint4*)(B + (((((((ax2_0_0 * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 2720)) = *(uint4*)(B + (((((((ax2_0_0 * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 81920));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 3264)) = *(uint4*)(B + (((((((ax2_0_0 * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 98304));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 272) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)) + 3808)) = *(uint4*)(B + (((((((ax2_0_0 * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 114688));
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[4352])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[4368])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[4992])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[5008])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[4], (&(((half*)buf_dyn_shmem)[5632])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[5], (&(((half*)buf_dyn_shmem)[5648])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[6], (&(((half*)buf_dyn_shmem)[6272])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[7], (&(((half*)buf_dyn_shmem)[6288])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[8], (&(((half*)buf_dyn_shmem)[6912])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[9], (&(((half*)buf_dyn_shmem)[6928])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[10], (&(((half*)buf_dyn_shmem)[7552])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[11], (&(((half*)buf_dyn_shmem)[7568])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[12], (&(((half*)buf_dyn_shmem)[8192])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[13], (&(((half*)buf_dyn_shmem)[8208])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[14], (&(((half*)buf_dyn_shmem)[8832])), 40);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[15], (&(((half*)buf_dyn_shmem)[8848])), 40);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 64)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 16)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 32)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 48)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 2176)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 2192)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 2208)])), 136);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 64) + 2224)])), 136);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[31]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[31]);
  }
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[1], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[2], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[3], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(C + ((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2))) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32768)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 16)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32784)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32800)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 640));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 48)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32816)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 896));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 64)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32832)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1152));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 80)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32848)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1408));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 96)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32864)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1664));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 112)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32880)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1920));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[4], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[5], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[6], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[7], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 65536)) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 98304)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 65552)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 98320)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 65568)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 98336)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 640));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 65584)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 98352)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 896));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 65600)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 98368)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1152));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 65616)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 98384)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1408));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 65632)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 98400)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1664));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 65648)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 98416)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1920));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[8], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[9], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[10], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[11], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 131072)) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 163840)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 131088)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 163856)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 131104)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 163872)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 640));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 131120)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 163888)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 896));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 131136)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 163904)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1152));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 131152)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 163920)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1408));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 131168)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 163936)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1664));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 131184)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 163952)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1920));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[12], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[13], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[14], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[15], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 196608)) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 229376)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 196624)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 229392)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 196640)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 229408)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 640));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 196656)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 229424)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 896));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 196672)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 229440)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1152));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 196688)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 229456)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1408));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 196704)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 229472)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1664));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 196720)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 229488)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1920));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[16], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[17], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[18], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[19], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 262144)) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 294912)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 262160)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 294928)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 262176)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 294944)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 640));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 262192)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 294960)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 896));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 262208)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 294976)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1152));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 262224)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 294992)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1408));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 262240)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 295008)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1664));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 262256)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 295024)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1920));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[20], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[21], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[22], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[23], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 327680)) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 360448)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 327696)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 360464)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 327712)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 360480)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 640));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 327728)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 360496)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 896));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 327744)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 360512)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1152));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 327760)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 360528)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1408));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 327776)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 360544)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1664));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 327792)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 360560)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1920));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[24], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[25], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[26], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[27], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 393216)) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 425984)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 393232)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 426000)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 393248)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 426016)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 640));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 393264)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 426032)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 896));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 393280)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 426048)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1152));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 393296)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 426064)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1408));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 393312)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 426080)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1664));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 393328)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 426096)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1920));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 1024)])), C_reindex_shared_dyn_wmma_accumulator[28], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 256)])), C_reindex_shared_dyn_wmma_accumulator[29], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 512)])), C_reindex_shared_dyn_wmma_accumulator[30], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 768)])), C_reindex_shared_dyn_wmma_accumulator[31], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 458752)) = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 491520)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 128));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 458768)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 256));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 491536)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 384));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 458784)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 512));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 491552)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 640));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 458800)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 768));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 491568)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 896));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 458816)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1024));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 491584)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1152));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 458832)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1280));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 491600)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1408));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 458848)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1536));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 491616)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1664));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 458864)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1792));
  *(half2*)(C + (((((((((((int)blockIdx.y) >> 2) * 1048576) + ((((int)blockIdx.x) >> 3) * 524288)) + (((int)threadIdx.y) * 16384)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)blockIdx.y) & 3) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 491632)) = *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 2)) + 1920));
}

