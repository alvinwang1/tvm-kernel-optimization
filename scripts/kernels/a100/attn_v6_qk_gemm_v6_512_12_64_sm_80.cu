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
extern "C" __global__ void __launch_bounds__(1024) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK);
extern "C" __global__ void __launch_bounds__(1024) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK) {
  half QK_local[16];
  __shared__ half Q_shared[4096];
  __shared__ half K_shared[2048];
  QK_local[0] = __float2half_rn(0.000000e+00f);
  QK_local[8] = __float2half_rn(0.000000e+00f);
  QK_local[1] = __float2half_rn(0.000000e+00f);
  QK_local[9] = __float2half_rn(0.000000e+00f);
  QK_local[2] = __float2half_rn(0.000000e+00f);
  QK_local[10] = __float2half_rn(0.000000e+00f);
  QK_local[3] = __float2half_rn(0.000000e+00f);
  QK_local[11] = __float2half_rn(0.000000e+00f);
  QK_local[4] = __float2half_rn(0.000000e+00f);
  QK_local[12] = __float2half_rn(0.000000e+00f);
  QK_local[5] = __float2half_rn(0.000000e+00f);
  QK_local[13] = __float2half_rn(0.000000e+00f);
  QK_local[6] = __float2half_rn(0.000000e+00f);
  QK_local[14] = __float2half_rn(0.000000e+00f);
  QK_local[7] = __float2half_rn(0.000000e+00f);
  QK_local[15] = __float2half_rn(0.000000e+00f);
  for (int k1_0 = 0; k1_0 < 4; ++k1_0) {
    __syncthreads();
    *(half2*)(Q_shared + (((int)threadIdx.x) * 2)) = *(half2*)(Q + ((((((((int)blockIdx.x) >> 5) * 65536) + (((((int)blockIdx.x) & 31) >> 3) * 8192)) + ((((int)threadIdx.x) >> 3) * 64)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 7) * 2)));
    *(half2*)(Q_shared + ((((int)threadIdx.x) * 2) + 2048)) = *(half2*)(Q + (((((((((int)blockIdx.x) >> 5) * 65536) + (((((int)blockIdx.x) & 31) >> 3) * 8192)) + ((((int)threadIdx.x) >> 3) * 64)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 32768));
    if (((int)threadIdx.x) < 512) {
      *(half4*)(K_shared + (((int)threadIdx.x) * 4)) = *(half4*)(K + (((((((((int)blockIdx.x) >> 5) * 65536) + ((((int)threadIdx.x) >> 8) * 32768)) + ((((int)blockIdx.x) & 7) * 4096)) + (((((int)threadIdx.x) & 255) >> 2) * 64)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)));
    }
    __syncthreads();
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128))] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1024)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 16)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1040)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 32)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1056)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 48)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1072)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1025)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 17)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1041)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 33)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1057)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 49)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1073)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 2)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1026)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 18)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1042)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 34)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1058)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 50)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1074)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 3)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1027)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 19)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1043)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 35)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1059)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 51)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1075)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 64)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1088)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 80)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1104)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 96)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1120)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 112)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1136)] * K_shared[(((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16))]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 65)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1089)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 81)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1105)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 97)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1121)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 113)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1137)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 1)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 66)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1090)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 82)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1106)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 98)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1122)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 114)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1138)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 2)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 67)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1091)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 83)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1107)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 99)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1123)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 115)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1139)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 3)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 4)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1028)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 20)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1044)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 36)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1060)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 52)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1076)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 5)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1029)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 21)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1045)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 37)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1061)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 53)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1077)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 6)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1030)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 22)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1046)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 38)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1062)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 54)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1078)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 7)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1031)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 23)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1047)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 39)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1063)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 55)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1079)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 68)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1092)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 84)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1108)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 100)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1124)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 116)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1140)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 4)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 69)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1093)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 85)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1109)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 101)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1125)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 117)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1141)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 5)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 70)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1094)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 86)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1110)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 102)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1126)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 118)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1142)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 6)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 71)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1095)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 87)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1111)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 103)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1127)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 119)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1143)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 7)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 8)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1032)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 24)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1048)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 40)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1064)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 56)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1080)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 9)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1033)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 25)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1049)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 41)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1065)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 57)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1081)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 10)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1034)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 26)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1050)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 42)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1066)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 58)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1082)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 11)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1035)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 27)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1051)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 43)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1067)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 59)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1083)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 72)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1096)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 88)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1112)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 104)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1128)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 120)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1144)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 8)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 73)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1097)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 89)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1113)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 105)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1129)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 121)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1145)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 9)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 74)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1098)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 90)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1114)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 106)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1130)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 122)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1146)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 10)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 75)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1099)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 91)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1115)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 107)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1131)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 123)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1147)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 11)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 12)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1036)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 28)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1052)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 44)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1068)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 60)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1084)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 13)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1037)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 29)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1053)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 45)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1069)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 61)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1085)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 14)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1038)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 30)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1054)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 46)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1070)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 62)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1086)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 15)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1039)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 31)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1055)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 47)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1071)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 63)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1087)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 76)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1100)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 92)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1116)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 108)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1132)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 124)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1148)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 12)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 77)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1101)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 93)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1117)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 109)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1133)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 125)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1149)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 13)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 78)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1102)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 94)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1118)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 110)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1134)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 126)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1150)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 14)]) * __float2half_rn(1.250000e-01f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 79)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1103)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 95)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1119)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 111)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1135)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 127)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 9) * 2048) + (((((int)threadIdx.x) & 511) >> 6) * 128)) + 1151)] * K_shared[((((((int)threadIdx.x) >> 9) * 1024) + ((((int)threadIdx.x) & 63) * 16)) + 15)]) * __float2half_rn(1.250000e-01f)));
  }
  QK[(((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63))] = QK_local[0];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 32768)] = QK_local[8];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 512)] = QK_local[1];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 33280)] = QK_local[9];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 1024)] = QK_local[2];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 33792)] = QK_local[10];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 1536)] = QK_local[3];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 34304)] = QK_local[11];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 2048)] = QK_local[4];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 34816)] = QK_local[12];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 2560)] = QK_local[5];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 35328)] = QK_local[13];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 3072)] = QK_local[6];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 35840)] = QK_local[14];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 3584)] = QK_local[7];
  QK[((((((((((int)blockIdx.x) >> 5) * 524288) + ((((int)threadIdx.x) >> 9) * 262144)) + (((((int)blockIdx.x) & 31) >> 3) * 65536)) + (((((int)threadIdx.x) & 511) >> 6) * 4096)) + ((((int)blockIdx.x) & 7) * 64)) + (((int)threadIdx.x) & 63)) + 36352)] = QK_local[15];
}

