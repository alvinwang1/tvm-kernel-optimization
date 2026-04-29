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
extern "C" __global__ void __launch_bounds__(384) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK);
extern "C" __global__ void __launch_bounds__(384) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK) {
  half QK_local[32];
  __shared__ half Q_shared[3072];
  __shared__ half K_shared[3072];
  QK_local[0] = __float2half_rn(0.000000e+00f);
  QK_local[2] = __float2half_rn(0.000000e+00f);
  QK_local[4] = __float2half_rn(0.000000e+00f);
  QK_local[6] = __float2half_rn(0.000000e+00f);
  QK_local[8] = __float2half_rn(0.000000e+00f);
  QK_local[10] = __float2half_rn(0.000000e+00f);
  QK_local[12] = __float2half_rn(0.000000e+00f);
  QK_local[14] = __float2half_rn(0.000000e+00f);
  QK_local[16] = __float2half_rn(0.000000e+00f);
  QK_local[18] = __float2half_rn(0.000000e+00f);
  QK_local[20] = __float2half_rn(0.000000e+00f);
  QK_local[22] = __float2half_rn(0.000000e+00f);
  QK_local[24] = __float2half_rn(0.000000e+00f);
  QK_local[26] = __float2half_rn(0.000000e+00f);
  QK_local[28] = __float2half_rn(0.000000e+00f);
  QK_local[30] = __float2half_rn(0.000000e+00f);
  QK_local[1] = __float2half_rn(0.000000e+00f);
  QK_local[3] = __float2half_rn(0.000000e+00f);
  QK_local[5] = __float2half_rn(0.000000e+00f);
  QK_local[7] = __float2half_rn(0.000000e+00f);
  QK_local[9] = __float2half_rn(0.000000e+00f);
  QK_local[11] = __float2half_rn(0.000000e+00f);
  QK_local[13] = __float2half_rn(0.000000e+00f);
  QK_local[15] = __float2half_rn(0.000000e+00f);
  QK_local[17] = __float2half_rn(0.000000e+00f);
  QK_local[19] = __float2half_rn(0.000000e+00f);
  QK_local[21] = __float2half_rn(0.000000e+00f);
  QK_local[23] = __float2half_rn(0.000000e+00f);
  QK_local[25] = __float2half_rn(0.000000e+00f);
  QK_local[27] = __float2half_rn(0.000000e+00f);
  QK_local[29] = __float2half_rn(0.000000e+00f);
  QK_local[31] = __float2half_rn(0.000000e+00f);
  for (int k1_0 = 0; k1_0 < 4; ++k1_0) {
    __syncthreads();
    *(uint4*)(Q_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(Q + (((((((((int)blockIdx.x) >> 6) * 98304) + ((((int)threadIdx.x) >> 7) * 32768)) + (((((int)blockIdx.x) & 63) >> 3) * 4096)) + (((((int)threadIdx.x) & 127) >> 1) * 64)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 1) * 8)));
    *(uint4*)(K_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(K + (((((((((int)blockIdx.x) >> 6) * 98304) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 7) * 4096)) + (((((int)threadIdx.x) & 127) >> 1) * 64)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 1) * 8)));
    __syncthreads();
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32))] * K_shared[(((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32))] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 256)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32))] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 512)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32))] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 256)] * K_shared[(((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 256)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 256)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 256)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 512)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 256)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 512)] * K_shared[(((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 512)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 256)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 512)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 512)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 512)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 768)] * K_shared[(((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 768)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 256)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 768)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 512)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 768)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 16)] * K_shared[(((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 16)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 256)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 16)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 512)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 16)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 272)] * K_shared[(((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 272)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 256)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 272)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 512)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 272)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 528)] * K_shared[(((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 528)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 256)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 528)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 512)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 528)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 784)] * K_shared[(((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 784)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 256)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 784)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 512)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 784)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 768)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 1)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 1)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 257)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 1)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 513)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 1)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 257)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 257)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 257)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 257)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 513)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 257)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 513)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 513)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 257)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 513)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 513)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 513)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 769)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 769)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 257)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 769)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 513)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 769)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 17)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 17)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 257)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 17)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 513)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 17)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 273)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 273)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 257)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 273)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 513)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 273)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 529)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 529)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 257)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 529)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 513)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 529)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 785)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 785)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 257)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 785)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 513)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 785)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 769)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 2)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 2)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 258)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 2)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 514)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 2)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 258)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 258)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 258)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 258)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 514)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 258)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 514)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 514)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 258)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 514)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 514)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 514)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 770)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 770)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 258)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 770)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 514)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 770)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 18)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 18)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 258)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 18)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 514)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 18)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 274)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 274)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 258)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 274)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 514)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 274)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 530)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 530)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 258)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 530)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 514)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 530)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 786)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 786)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 258)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 786)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 514)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 786)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 770)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 3)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 3)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 259)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 3)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 515)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 3)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 259)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 259)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 259)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 259)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 515)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 259)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 515)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 515)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 259)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 515)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 515)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 515)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 771)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 771)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 259)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 771)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 515)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 771)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 19)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 19)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 259)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 19)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 515)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 19)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 275)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 275)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 259)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 275)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 515)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 275)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 531)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 531)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 259)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 531)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 515)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 531)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 787)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 787)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 259)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 787)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 515)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 787)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 771)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 4)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 4)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 260)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 4)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 516)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 4)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 260)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 260)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 260)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 260)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 516)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 260)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 516)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 516)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 260)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 516)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 516)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 516)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 772)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 772)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 260)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 772)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 516)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 772)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 20)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 20)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 260)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 20)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 516)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 20)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 276)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 276)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 260)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 276)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 516)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 276)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 532)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 532)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 260)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 532)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 516)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 532)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 788)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 788)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 260)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 788)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 516)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 788)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 772)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 5)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 5)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 261)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 5)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 517)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 5)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 261)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 261)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 261)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 261)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 517)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 261)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 517)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 517)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 261)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 517)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 517)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 517)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 773)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 773)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 261)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 773)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 517)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 773)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 21)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 21)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 261)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 21)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 517)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 21)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 277)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 277)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 261)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 277)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 517)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 277)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 533)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 533)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 261)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 533)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 517)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 533)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 789)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 789)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 261)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 789)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 517)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 789)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 773)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 6)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 6)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 262)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 6)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 518)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 6)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 262)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 262)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 262)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 262)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 518)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 262)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 518)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 518)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 262)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 518)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 518)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 518)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 774)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 774)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 262)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 774)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 518)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 774)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 22)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 22)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 262)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 22)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 518)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 22)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 278)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 278)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 262)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 278)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 518)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 278)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 534)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 534)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 262)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 534)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 518)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 534)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 790)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 790)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 262)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 790)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 518)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 790)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 774)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 7)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 7)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 263)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 7)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 519)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 7)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 263)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 263)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 263)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 263)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 519)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 263)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 519)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 519)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 263)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 519)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 519)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 519)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 775)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 775)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 263)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 775)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 519)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 775)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 23)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 23)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 263)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 23)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 519)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 23)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 279)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 279)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 263)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 279)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 519)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 279)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 535)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 535)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 263)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 535)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 519)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 535)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 791)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 791)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 263)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 791)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 519)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 791)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 775)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 8)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 8)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 264)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 8)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 520)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 8)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 264)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 264)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 264)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 264)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 520)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 264)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 520)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 520)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 264)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 520)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 520)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 520)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 776)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 776)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 264)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 776)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 520)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 776)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 24)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 24)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 264)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 24)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 520)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 24)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 280)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 280)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 264)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 280)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 520)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 280)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 536)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 536)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 264)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 536)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 520)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 536)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 792)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 792)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 264)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 792)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 520)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 792)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 776)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 9)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 9)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 265)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 9)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 521)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 9)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 265)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 265)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 265)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 265)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 521)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 265)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 521)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 521)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 265)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 521)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 521)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 521)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 777)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 777)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 265)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 777)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 521)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 777)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 25)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 25)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 265)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 25)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 521)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 25)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 281)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 281)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 265)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 281)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 521)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 281)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 537)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 537)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 265)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 537)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 521)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 537)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 793)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 793)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 265)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 793)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 521)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 793)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 777)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 10)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 10)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 266)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 10)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 522)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 10)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 266)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 266)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 266)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 266)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 522)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 266)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 522)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 522)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 266)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 522)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 522)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 522)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 778)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 778)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 266)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 778)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 522)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 778)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 26)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 26)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 266)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 26)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 522)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 26)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 282)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 282)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 266)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 282)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 522)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 282)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 538)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 538)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 266)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 538)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 522)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 538)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 794)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 794)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 266)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 794)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 522)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 794)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 778)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 11)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 11)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 267)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 11)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 523)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 11)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 267)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 267)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 267)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 267)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 523)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 267)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 523)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 523)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 267)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 523)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 523)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 523)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 779)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 779)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 267)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 779)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 523)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 779)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 27)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 27)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 267)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 27)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 523)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 27)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 283)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 283)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 267)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 283)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 523)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 283)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 539)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 539)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 267)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 539)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 523)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 539)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 795)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 795)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 267)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 795)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 523)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 795)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 779)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 12)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 12)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 268)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 12)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 524)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 12)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 268)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 268)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 268)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 268)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 524)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 268)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 524)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 524)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 268)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 524)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 524)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 524)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 780)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 780)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 268)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 780)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 524)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 780)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 28)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 28)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 268)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 28)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 524)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 28)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 284)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 284)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 268)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 284)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 524)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 284)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 540)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 540)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 268)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 540)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 524)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 540)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 796)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 796)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 268)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 796)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 524)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 796)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 780)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 13)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 13)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 269)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 13)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 525)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 13)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 269)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 269)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 269)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 269)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 525)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 269)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 525)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 525)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 269)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 525)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 525)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 525)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 781)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 781)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 269)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 781)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 525)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 781)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 29)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 29)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 269)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 29)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 525)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 29)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 285)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 285)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 269)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 285)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 525)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 285)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 541)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 541)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 269)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 541)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 525)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 541)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 797)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 797)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 269)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 797)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 525)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 797)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 781)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 14)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 14)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 270)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 14)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 526)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 14)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 270)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 270)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 270)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 270)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 526)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 270)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 526)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 526)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 270)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 526)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 526)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 526)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 782)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 782)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 270)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 782)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 526)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 782)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 30)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 30)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 270)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 30)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 526)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 30)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 286)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 286)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 270)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 286)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 526)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 286)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 542)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 542)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 270)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 542)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 526)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 542)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 798)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 798)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 270)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 798)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 526)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 798)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 782)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 15)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 15)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 271)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 15)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 527)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 15)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 271)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 271)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 271)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 271)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 527)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 271)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 527)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 527)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 271)]) * __float2half_rn(1.250000e-01f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 527)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 527)]) * __float2half_rn(1.250000e-01f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 527)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 783)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 783)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 271)]) * __float2half_rn(1.250000e-01f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 783)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 527)]) * __float2half_rn(1.250000e-01f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 783)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 31)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 31)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 271)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 31)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 527)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 31)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 287)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 287)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 271)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 287)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 527)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 287)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 543)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 543)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 271)]) * __float2half_rn(1.250000e-01f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 543)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 527)]) * __float2half_rn(1.250000e-01f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 543)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 799)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 799)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 271)]) * __float2half_rn(1.250000e-01f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 799)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 527)]) * __float2half_rn(1.250000e-01f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 7) * 1024) + (((((int)threadIdx.x) & 127) >> 4) * 32)) + 799)] * K_shared[((((((int)threadIdx.x) >> 7) * 1024) + ((((int)threadIdx.x) & 15) * 16)) + 783)]) * __float2half_rn(1.250000e-01f)));
  }
  QK[(((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15))] = QK_local[0];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 16)] = QK_local[2];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 32)] = QK_local[4];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 48)] = QK_local[6];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 8192)] = QK_local[8];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 8208)] = QK_local[10];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 8224)] = QK_local[12];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 8240)] = QK_local[14];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 16384)] = QK_local[16];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 16400)] = QK_local[18];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 16416)] = QK_local[20];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 16432)] = QK_local[22];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 24576)] = QK_local[24];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 24592)] = QK_local[26];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 24608)] = QK_local[28];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 24624)] = QK_local[30];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 512)] = QK_local[1];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 528)] = QK_local[3];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 544)] = QK_local[5];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 560)] = QK_local[7];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 8704)] = QK_local[9];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 8720)] = QK_local[11];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 8736)] = QK_local[13];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 8752)] = QK_local[15];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 16896)] = QK_local[17];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 16912)] = QK_local[19];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 16928)] = QK_local[21];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 16944)] = QK_local[23];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 25088)] = QK_local[25];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 25104)] = QK_local[27];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 25120)] = QK_local[29];
  QK[((((((((((int)blockIdx.x) >> 6) * 786432) + ((((int)threadIdx.x) >> 7) * 262144)) + (((((int)blockIdx.x) & 63) >> 3) * 32768)) + (((((int)threadIdx.x) & 127) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 15)) + 25136)] = QK_local[31];
}

