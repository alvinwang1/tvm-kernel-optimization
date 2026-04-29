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
extern "C" __global__ void __launch_bounds__(32) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum);
extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax);
extern "C" __global__ void __launch_bounds__(128) main_kernel_2(half* __restrict__ Out, half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum, half* __restrict__ V);
extern "C" __global__ void __launch_bounds__(32) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum) {
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __float2half_rn(0.000000e+00f);
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128))] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 1)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 2)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 3)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 4)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 5)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 6)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 7)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 8)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 9)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 10)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 11)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 12)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 13)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 14)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 15)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 16)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 17)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 18)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 19)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 20)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 21)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 22)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 23)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 24)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 25)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 26)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 27)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 28)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 29)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 30)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 31)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 32)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 33)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 34)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 35)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 36)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 37)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 38)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 39)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 40)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 41)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 42)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 43)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 44)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 45)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 46)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 47)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 48)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 49)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 50)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 51)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 52)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 53)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 54)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 55)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 56)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 57)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 58)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 59)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 60)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 61)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 62)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 63)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 64)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 65)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 66)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 67)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 68)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 69)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 70)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 71)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 72)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 73)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 74)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 75)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 76)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 77)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 78)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 79)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 80)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 81)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 82)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 83)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 84)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 85)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 86)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 87)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 88)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 89)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 90)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 91)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 92)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 93)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 94)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 95)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 96)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 97)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 98)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 99)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 100)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 101)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 102)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 103)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 104)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 105)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 106)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 107)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 108)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 109)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 110)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 111)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 112)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 113)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 114)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 115)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 116)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 117)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 118)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 119)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 120)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 121)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 122)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 123)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 124)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 125)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 126)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 127)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
}

extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax) {
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __float2half_rn(-6.550400e+04f);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128))]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 1)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 2)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 3)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 4)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 5)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 6)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 7)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 8)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 9)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 10)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 11)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 12)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 13)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 14)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 15)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 16)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 17)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 18)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 19)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 20)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 21)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 22)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 23)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 24)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 25)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 26)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 27)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 28)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 29)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 30)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 31)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 32)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 33)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 34)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 35)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 36)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 37)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 38)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 39)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 40)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 41)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 42)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 43)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 44)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 45)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 46)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 47)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 48)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 49)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 50)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 51)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 52)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 53)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 54)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 55)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 56)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 57)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 58)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 59)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 60)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 61)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 62)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 63)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 64)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 65)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 66)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 67)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 68)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 69)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 70)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 71)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 72)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 73)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 74)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 75)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 76)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 77)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 78)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 79)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 80)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 81)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 82)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 83)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 84)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 85)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 86)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 87)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 88)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 89)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 90)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 91)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 92)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 93)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 94)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 95)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 96)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 97)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 98)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 99)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 100)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 101)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 102)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 103)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 104)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 105)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 106)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 107)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 108)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 109)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 110)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 111)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 112)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 113)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 114)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 115)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 116)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 117)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 118)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 119)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 120)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 121)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 122)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 123)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 124)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 125)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 126)]);
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + 127)]);
}

extern "C" __global__ void __launch_bounds__(128) main_kernel_2(half* __restrict__ Out, half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum, half* __restrict__ V) {
  half Out_local[2];
  __shared__ half Attn_shared[256];
  __shared__ half V_shared[256];
  Out_local[0] = __float2half_rn(0.000000e+00f);
  Out_local[1] = __float2half_rn(0.000000e+00f);
  half2 __1;
    half2 __2;
    half2 __3;
      half2 v_ = *(half2*)(QK + ((((((int)blockIdx.x) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 2)));
      half2 v__1 = make_half2(RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
      __3.x = (v_.x-v__1.x);
      __3.y = (v_.y-v__1.y);
    __2.x = hexp(__3.x);
    __2.y = hexp(__3.y);
    half2 v__2 = make_half2(RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
    __1.x = (__2.x/v__2.x);
    __1.y = (__2.y/v__2.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __1;
  *(half2*)(V_shared + (((int)threadIdx.x) * 2)) = *(half2*)(V + (((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)));
  __syncthreads();
  Out_local[0] = (Out_local[0] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[((((int)threadIdx.x) & 7) * 2)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 16)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 32)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 48)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 64)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 80)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 96)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 112)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 1)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 17)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 33)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 49)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 65)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 81)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 97)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 113)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 128)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 144)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 160)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 176)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 192)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 208)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 224)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 240)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 129)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 145)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 161)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 177)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 193)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 209)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 225)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 241)]));
  __syncthreads();
  half2 __4;
    half2 __5;
    half2 __6;
      half2 v__3 = *(half2*)(QK + (((((((int)blockIdx.x) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 16));
      half2 v__4 = make_half2(RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
      __6.x = (v__3.x-v__4.x);
      __6.y = (v__3.y-v__4.y);
    __5.x = hexp(__6.x);
    __5.y = hexp(__6.y);
    half2 v__5 = make_half2(RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
    __4.x = (__5.x/v__5.x);
    __4.y = (__5.y/v__5.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __4;
  *(half2*)(V_shared + (((int)threadIdx.x) * 2)) = *(half2*)(V + ((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1024));
  __syncthreads();
  Out_local[0] = (Out_local[0] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[((((int)threadIdx.x) & 7) * 2)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 16)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 32)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 48)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 64)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 80)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 96)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 112)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 1)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 17)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 33)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 49)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 65)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 81)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 97)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 113)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 128)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 144)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 160)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 176)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 192)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 208)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 224)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 240)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 129)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 145)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 161)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 177)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 193)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 209)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 225)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 241)]));
  __syncthreads();
  half2 __7;
    half2 __8;
    half2 __9;
      half2 v__6 = *(half2*)(QK + (((((((int)blockIdx.x) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 32));
      half2 v__7 = make_half2(RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
      __9.x = (v__6.x-v__7.x);
      __9.y = (v__6.y-v__7.y);
    __8.x = hexp(__9.x);
    __8.y = hexp(__9.y);
    half2 v__8 = make_half2(RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
    __7.x = (__8.x/v__8.x);
    __7.y = (__8.y/v__8.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __7;
  *(half2*)(V_shared + (((int)threadIdx.x) * 2)) = *(half2*)(V + ((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 2048));
  __syncthreads();
  Out_local[0] = (Out_local[0] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[((((int)threadIdx.x) & 7) * 2)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 16)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 32)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 48)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 64)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 80)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 96)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 112)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 1)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 17)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 33)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 49)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 65)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 81)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 97)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 113)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 128)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 144)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 160)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 176)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 192)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 208)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 224)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 240)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 129)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 145)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 161)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 177)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 193)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 209)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 225)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 241)]));
  __syncthreads();
  half2 __10;
    half2 __11;
    half2 __12;
      half2 v__9 = *(half2*)(QK + (((((((int)blockIdx.x) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 48));
      half2 v__10 = make_half2(RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
      __12.x = (v__9.x-v__10.x);
      __12.y = (v__9.y-v__10.y);
    __11.x = hexp(__12.x);
    __11.y = hexp(__12.y);
    half2 v__11 = make_half2(RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
    __10.x = (__11.x/v__11.x);
    __10.y = (__11.y/v__11.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __10;
  *(half2*)(V_shared + (((int)threadIdx.x) * 2)) = *(half2*)(V + ((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 3072));
  __syncthreads();
  Out_local[0] = (Out_local[0] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[((((int)threadIdx.x) & 7) * 2)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 16)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 32)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 48)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 64)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 80)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 96)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 112)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 1)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 17)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 33)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 49)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 65)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 81)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 97)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 113)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 128)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 144)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 160)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 176)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 192)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 208)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 224)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 240)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 129)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 145)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 161)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 177)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 193)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 209)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 225)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 241)]));
  __syncthreads();
  half2 __13;
    half2 __14;
    half2 __15;
      half2 v__12 = *(half2*)(QK + (((((((int)blockIdx.x) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 64));
      half2 v__13 = make_half2(RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
      __15.x = (v__12.x-v__13.x);
      __15.y = (v__12.y-v__13.y);
    __14.x = hexp(__15.x);
    __14.y = hexp(__15.y);
    half2 v__14 = make_half2(RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
    __13.x = (__14.x/v__14.x);
    __13.y = (__14.y/v__14.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __13;
  *(half2*)(V_shared + (((int)threadIdx.x) * 2)) = *(half2*)(V + ((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 4096));
  __syncthreads();
  Out_local[0] = (Out_local[0] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[((((int)threadIdx.x) & 7) * 2)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 16)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 32)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 48)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 64)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 80)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 96)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 112)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 1)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 17)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 33)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 49)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 65)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 81)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 97)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 113)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 128)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 144)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 160)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 176)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 192)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 208)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 224)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 240)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 129)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 145)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 161)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 177)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 193)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 209)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 225)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 241)]));
  __syncthreads();
  half2 __16;
    half2 __17;
    half2 __18;
      half2 v__15 = *(half2*)(QK + (((((((int)blockIdx.x) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 80));
      half2 v__16 = make_half2(RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
      __18.x = (v__15.x-v__16.x);
      __18.y = (v__15.y-v__16.y);
    __17.x = hexp(__18.x);
    __17.y = hexp(__18.y);
    half2 v__17 = make_half2(RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
    __16.x = (__17.x/v__17.x);
    __16.y = (__17.y/v__17.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __16;
  *(half2*)(V_shared + (((int)threadIdx.x) * 2)) = *(half2*)(V + ((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 5120));
  __syncthreads();
  Out_local[0] = (Out_local[0] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[((((int)threadIdx.x) & 7) * 2)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 16)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 32)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 48)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 64)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 80)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 96)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 112)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 1)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 17)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 33)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 49)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 65)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 81)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 97)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 113)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 128)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 144)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 160)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 176)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 192)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 208)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 224)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 240)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 129)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 145)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 161)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 177)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 193)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 209)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 225)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 241)]));
  __syncthreads();
  half2 __19;
    half2 __20;
    half2 __21;
      half2 v__18 = *(half2*)(QK + (((((((int)blockIdx.x) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 96));
      half2 v__19 = make_half2(RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
      __21.x = (v__18.x-v__19.x);
      __21.y = (v__18.y-v__19.y);
    __20.x = hexp(__21.x);
    __20.y = hexp(__21.y);
    half2 v__20 = make_half2(RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
    __19.x = (__20.x/v__20.x);
    __19.y = (__20.y/v__20.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __19;
  *(half2*)(V_shared + (((int)threadIdx.x) * 2)) = *(half2*)(V + ((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 6144));
  __syncthreads();
  Out_local[0] = (Out_local[0] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[((((int)threadIdx.x) & 7) * 2)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 16)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 32)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 48)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 64)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 80)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 96)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 112)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 1)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 17)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 33)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 49)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 65)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 81)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 97)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 113)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 128)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 144)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 160)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 176)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 192)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 208)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 224)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 240)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 129)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 145)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 161)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 177)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 193)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 209)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 225)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 241)]));
  __syncthreads();
  half2 __22;
    half2 __23;
    half2 __24;
      half2 v__21 = *(half2*)(QK + (((((((int)blockIdx.x) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 2)) + 112));
      half2 v__22 = make_half2(RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowMax[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
      __24.x = (v__21.x-v__22.x);
      __24.y = (v__21.y-v__22.y);
    __23.x = hexp(__24.x);
    __23.y = hexp(__24.y);
    half2 v__23 = make_half2(RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))], RowSum[(((((int)blockIdx.x) >> 2) * 16) + (((int)threadIdx.x) >> 3))]);
    __22.x = (__23.x/v__23.x);
    __22.y = (__23.y/v__23.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __22;
  *(half2*)(V_shared + (((int)threadIdx.x) * 2)) = *(half2*)(V + ((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 7168));
  __syncthreads();
  Out_local[0] = (Out_local[0] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[((((int)threadIdx.x) & 7) * 2)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 16)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 32)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 48)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 64)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 80)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 96)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 112)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[((((int)threadIdx.x) >> 3) * 16)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 1)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 1)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 17)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 2)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 33)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 3)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 49)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 4)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 65)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 5)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 81)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 6)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 97)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 7)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 113)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 128)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 144)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 160)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 176)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 192)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 208)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 224)]));
  Out_local[0] = (Out_local[0] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 240)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 8)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 129)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 9)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 145)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 10)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 161)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 11)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 177)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 12)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 193)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 13)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 209)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 14)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 225)]));
  Out_local[1] = (Out_local[1] + (Attn_shared[(((((int)threadIdx.x) >> 3) * 16) + 15)] * V_shared[(((((int)threadIdx.x) & 7) * 2) + 241)]));
  Out[(((((((int)blockIdx.x) >> 2) * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2))] = Out_local[0];
  Out[((((((((int)blockIdx.x) >> 2) * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)blockIdx.x) & 3) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 1)] = Out_local[1];
}

