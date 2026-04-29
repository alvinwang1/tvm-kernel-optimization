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
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_reindex_shared_dyn_wmma_accumulator[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_reindex_shared_dyn_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> B_reindex_shared_dyn_wmma_matrix_b[2];
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + ((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12288));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + ((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 3072));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 6144));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 9216));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 12288));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 15360));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 18432));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 21504));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 24576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 27648));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 30720));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 33792));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 36864));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 39936));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 43008));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 46080));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 64));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12352));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 49152));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 52224));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 55296));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 58368));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 61440));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 64512));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 67584));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 70656));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 73728));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 76800));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 79872));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 82944));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 86016));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 89088));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 92160));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 95232));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 128));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12416));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 98304));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 101376));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 104448));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 107520));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 110592));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 113664));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 116736));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 119808));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 122880));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 125952));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 129024));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 132096));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 135168));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 138240));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 141312));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 144384));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 192));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12480));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 147456));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 150528));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 153600));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 156672));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 159744));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 162816));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 165888));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 168960));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 172032));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 175104));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 178176));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 181248));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 184320));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 187392));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 190464));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 193536));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 256));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12544));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 196608));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 199680));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 202752));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 205824));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 208896));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 211968));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 215040));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 218112));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 221184));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 224256));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 227328));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 230400));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 233472));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 236544));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 239616));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 242688));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 320));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12608));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 245760));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 248832));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 251904));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 254976));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 258048));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 261120));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 264192));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 267264));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 270336));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 273408));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 276480));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 279552));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 282624));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 285696));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 288768));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 291840));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 384));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12672));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 294912));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 297984));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 301056));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 304128));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 307200));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 310272));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 313344));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 316416));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 319488));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 322560));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 325632));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 328704));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 331776));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 334848));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 337920));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 340992));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 448));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12736));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 344064));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 347136));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 350208));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 353280));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 356352));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 359424));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 362496));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 365568));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 368640));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 371712));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 374784));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 377856));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 380928));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 384000));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 387072));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 390144));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 512));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12800));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 393216));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 396288));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 399360));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 402432));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 405504));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 408576));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 411648));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 414720));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 417792));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 420864));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 423936));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 427008));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 430080));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 433152));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 436224));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 439296));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 576));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12864));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 442368));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 445440));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 448512));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 451584));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 454656));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 457728));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 460800));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 463872));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 466944));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 470016));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 473088));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 476160));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 479232));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 482304));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 485376));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 488448));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 640));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12928));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 491520));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 494592));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 497664));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 500736));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 503808));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 506880));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 509952));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 513024));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 516096));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 519168));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 522240));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 525312));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 528384));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 531456));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 534528));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 537600));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 704));
  *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(A + (((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + (((int)threadIdx.y) * 3072)) + ((((int)threadIdx.x) >> 3) * 768)) + ((((int)threadIdx.x) & 7) * 8)) + 12992));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2304)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 540672));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2592)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 543744));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 2880)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 546816));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3168)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 549888));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3456)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 552960));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 3744)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 556032));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4032)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 559104));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4320)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 562176));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4608)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 565248));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 4896)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 568320));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5184)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 571392));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5472)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 574464));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 5760)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 577536));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6048)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 580608));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6336)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 583680));
  *(half2*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 6624)) = *(half2*)(B + (((((((int)threadIdx.y) * 768) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.x) * 2)) + 586752));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) >> 1) * 1152)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2304)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 2320)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 16)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3456)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 3472)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 32)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4608)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 4624)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) >> 1) * 1152) + 48)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5760)])), 72);
  nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[(((((int)threadIdx.y) & 1) * 32) + 5776)])), 72);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 512)])), C_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 512) + 256)])), C_reindex_shared_dyn_wmma_accumulator[1], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + ((((int)threadIdx.x) >> 1) * 768)) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)));
  *(uint4*)(C + (((((((((((int)blockIdx.y) >> 2) * 49152) + ((((int)blockIdx.x) / 3) * 24576)) + ((((int)threadIdx.x) >> 1) * 768)) + ((((int)blockIdx.y) & 3) * 192)) + ((((int)blockIdx.x) % 3) * 64)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 12288)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 1024));
}

