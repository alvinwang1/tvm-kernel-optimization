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
extern "C" __global__ void __launch_bounds__(192) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK);
extern "C" __global__ void __launch_bounds__(192) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK) {
  half QK_local[4];
  __shared__ half Q_shared[3072];
  __shared__ half K_shared[3072];
  QK_local[0] = __float2half_rn(0.000000e+00f);
  QK_local[1] = __float2half_rn(0.000000e+00f);
  QK_local[2] = __float2half_rn(0.000000e+00f);
  QK_local[3] = __float2half_rn(0.000000e+00f);
  *(half4*)(Q_shared + (((int)threadIdx.x) * 4)) = *(half4*)(Q + ((((((((int)blockIdx.x) >> 6) * 24576) + ((((int)threadIdx.x) >> 6) * 8192)) + (((((int)blockIdx.x) & 63) >> 3) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 4)));
  if (((int)threadIdx.x) < 96) {
    *(uint4*)(K_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(K + ((((((((int)blockIdx.x) >> 6) * 24576) + ((((int)threadIdx.x) >> 5) * 8192)) + ((((int)blockIdx.x) & 7) * 1024)) + (((((int)threadIdx.x) & 31) >> 1) * 64)) + ((((int)threadIdx.x) & 1) * 8)));
  }
__asm__ __volatile__("cp.async.commit_group;");

  *(half4*)(Q_shared + ((((int)threadIdx.x) * 4) + 768)) = *(half4*)(Q + (((((((((int)blockIdx.x) >> 6) * 24576) + ((((int)threadIdx.x) >> 6) * 8192)) + (((((int)blockIdx.x) & 63) >> 3) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 4)) + 16));
  if (((int)threadIdx.x) < 96) {
    *(uint4*)(K_shared + ((((int)threadIdx.x) * 8) + 768)) = *(uint4*)(K + (((((((((int)blockIdx.x) >> 6) * 24576) + ((((int)threadIdx.x) >> 5) * 8192)) + ((((int)blockIdx.x) & 7) * 1024)) + (((((int)threadIdx.x) & 31) >> 1) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 16));
  }
__asm__ __volatile__("cp.async.commit_group;");

  *(half4*)(Q_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(half4*)(Q + (((((((((int)blockIdx.x) >> 6) * 24576) + ((((int)threadIdx.x) >> 6) * 8192)) + (((((int)blockIdx.x) & 63) >> 3) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 4)) + 32));
  if (((int)threadIdx.x) < 96) {
    *(uint4*)(K_shared + ((((int)threadIdx.x) * 8) + 1536)) = *(uint4*)(K + (((((((((int)blockIdx.x) >> 6) * 24576) + ((((int)threadIdx.x) >> 5) * 8192)) + ((((int)blockIdx.x) & 7) * 1024)) + (((((int)threadIdx.x) & 31) >> 1) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 32));
  }
__asm__ __volatile__("cp.async.commit_group;");

  *(half4*)(Q_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(half4*)(Q + (((((((((int)blockIdx.x) >> 6) * 24576) + ((((int)threadIdx.x) >> 6) * 8192)) + (((((int)blockIdx.x) & 63) >> 3) * 1024)) + (((((int)threadIdx.x) & 63) >> 2) * 64)) + ((((int)threadIdx.x) & 3) * 4)) + 48));
  if (((int)threadIdx.x) < 96) {
    *(uint4*)(K_shared + ((((int)threadIdx.x) * 8) + 2304)) = *(uint4*)(K + (((((((((int)blockIdx.x) >> 6) * 24576) + ((((int)threadIdx.x) >> 5) * 8192)) + ((((int)blockIdx.x) & 7) * 1024)) + (((((int)threadIdx.x) & 31) >> 1) * 64)) + ((((int)threadIdx.x) & 1) * 8)) + 48));
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16))] * K_shared[(((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16))]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16))] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 128)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 128)] * K_shared[(((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16))]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 128)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 128)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 129)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 129)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 129)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 129)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 130)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 130)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 130)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 130)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 3)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 3)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 131)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 131)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 131)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 131)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 4)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 4)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 132)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 132)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 132)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 132)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 5)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 5)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 133)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 133)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 133)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 133)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 6)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 6)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 134)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 134)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 134)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 134)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 7)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 7)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 135)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 135)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 135)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 135)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 8)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 8)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 136)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 136)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 136)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 136)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 9)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 9)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 137)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 137)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 137)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 137)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 10)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 10)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 138)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 138)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 138)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 138)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 11)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 11)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 139)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 139)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 139)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 139)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 12)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 12)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 140)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 140)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 140)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 140)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 13)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 13)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 141)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 141)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 141)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 141)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 14)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 14)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 142)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 142)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 142)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 142)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 15)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 15)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 143)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 143)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 143)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 143)]) * __float2half_rn(1.250000e-01f)));
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 768)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 768)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 896)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 896)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 896)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 896)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 769)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 769)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 897)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 897)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 897)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 897)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 770)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 770)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 898)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 898)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 898)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 898)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 771)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 771)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 899)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 899)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 899)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 899)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 772)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 772)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 900)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 900)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 900)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 900)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 773)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 773)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 901)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 901)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 901)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 901)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 774)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 774)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 902)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 902)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 902)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 902)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 775)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 775)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 903)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 903)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 903)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 903)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 776)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 776)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 904)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 904)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 904)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 904)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 777)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 777)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 905)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 905)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 905)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 905)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 778)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 778)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 906)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 906)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 906)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 906)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 779)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 779)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 907)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 907)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 907)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 907)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 780)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 780)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 908)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 908)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 908)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 908)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 781)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 781)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 909)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 909)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 909)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 909)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 782)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 782)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 910)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 910)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 910)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 910)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 783)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 783)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 911)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 911)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 911)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 911)]) * __float2half_rn(1.250000e-01f)));
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1536)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1536)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1536)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1664)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1664)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1536)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1664)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1664)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1537)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1537)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1537)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1665)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1665)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1537)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1665)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1665)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1538)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1538)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1538)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1666)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1666)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1538)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1666)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1666)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1539)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1539)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1539)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1667)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1667)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1539)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1667)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1667)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1540)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1540)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1540)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1668)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1668)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1540)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1668)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1668)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1541)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1541)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1541)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1669)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1669)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1541)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1669)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1669)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1542)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1542)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1542)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1670)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1670)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1542)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1670)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1670)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1543)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1543)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1543)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1671)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1671)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1543)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1671)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1671)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1544)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1544)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1544)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1672)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1672)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1544)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1672)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1672)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1545)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1545)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1545)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1673)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1673)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1545)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1673)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1673)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1546)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1546)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1546)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1674)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1674)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1546)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1674)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1674)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1547)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1547)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1547)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1675)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1675)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1547)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1675)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1675)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1548)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1548)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1548)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1676)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1676)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1548)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1676)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1676)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1549)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1549)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1549)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1677)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1677)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1549)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1677)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1677)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1550)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1550)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1550)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1678)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1678)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1550)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1678)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1678)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1551)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1551)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1551)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1679)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1679)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1551)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1679)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 1679)]) * __float2half_rn(1.250000e-01f)));
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2304)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2304)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2304)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2432)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2432)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2304)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2432)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2432)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2305)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2305)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2305)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2433)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2433)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2305)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2433)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2433)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2306)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2306)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2306)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2434)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2434)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2306)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2434)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2434)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2307)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2307)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2307)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2435)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2435)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2307)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2435)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2435)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2308)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2308)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2308)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2436)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2436)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2308)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2436)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2436)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2309)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2309)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2309)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2437)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2437)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2309)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2437)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2437)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2310)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2310)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2310)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2438)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2438)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2310)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2438)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2438)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2311)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2311)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2311)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2439)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2439)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2311)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2439)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2439)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2312)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2312)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2312)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2440)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2440)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2312)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2440)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2440)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2313)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2313)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2313)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2441)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2441)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2313)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2441)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2441)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2314)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2314)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2314)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2442)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2442)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2314)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2442)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2442)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2315)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2315)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2315)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2443)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2443)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2315)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2443)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2443)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2316)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2316)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2316)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2444)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2444)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2316)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2444)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2444)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2317)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2317)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2317)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2445)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2445)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2317)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2445)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2445)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2318)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2318)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2318)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2446)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2446)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2318)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2446)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2446)]) * __float2half_rn(1.250000e-01f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2319)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2319)]) * __float2half_rn(1.250000e-01f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2319)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2447)]) * __float2half_rn(1.250000e-01f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2447)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2319)]) * __float2half_rn(1.250000e-01f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 256) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2447)] * K_shared[((((((int)threadIdx.x) >> 6) * 256) + ((((int)threadIdx.x) & 7) * 16)) + 2447)]) * __float2half_rn(1.250000e-01f)));
  QK[(((((((((int)blockIdx.x) >> 6) * 49152) + ((((int)threadIdx.x) >> 6) * 16384)) + (((((int)blockIdx.x) & 63) >> 3) * 2048)) + (((((int)threadIdx.x) & 63) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7))] = QK_local[0];
  QK[((((((((((int)blockIdx.x) >> 6) * 49152) + ((((int)threadIdx.x) >> 6) * 16384)) + (((((int)blockIdx.x) & 63) >> 3) * 2048)) + (((((int)threadIdx.x) & 63) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 8)] = QK_local[1];
  QK[((((((((((int)blockIdx.x) >> 6) * 49152) + ((((int)threadIdx.x) >> 6) * 16384)) + (((((int)blockIdx.x) & 63) >> 3) * 2048)) + (((((int)threadIdx.x) & 63) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 1024)] = QK_local[2];
  QK[((((((((((int)blockIdx.x) >> 6) * 49152) + ((((int)threadIdx.x) >> 6) * 16384)) + (((((int)blockIdx.x) & 63) >> 3) * 2048)) + (((((int)threadIdx.x) & 63) >> 3) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 7)) + 1032)] = QK_local[3];
}

