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
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK);
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK) {
  half QK_local[16];
  __shared__ half Q_shared[1024];
  __shared__ half K_shared[1024];
  QK_local[0] = __float2half_rn(0.000000e+00f);
  QK_local[1] = __float2half_rn(0.000000e+00f);
  QK_local[2] = __float2half_rn(0.000000e+00f);
  QK_local[3] = __float2half_rn(0.000000e+00f);
  QK_local[4] = __float2half_rn(0.000000e+00f);
  QK_local[5] = __float2half_rn(0.000000e+00f);
  QK_local[6] = __float2half_rn(0.000000e+00f);
  QK_local[7] = __float2half_rn(0.000000e+00f);
  QK_local[8] = __float2half_rn(0.000000e+00f);
  QK_local[9] = __float2half_rn(0.000000e+00f);
  QK_local[10] = __float2half_rn(0.000000e+00f);
  QK_local[11] = __float2half_rn(0.000000e+00f);
  QK_local[12] = __float2half_rn(0.000000e+00f);
  QK_local[13] = __float2half_rn(0.000000e+00f);
  QK_local[14] = __float2half_rn(0.000000e+00f);
  QK_local[15] = __float2half_rn(0.000000e+00f);
  for (int k1_0 = 0; k1_0 < 8; ++k1_0) {
    __syncthreads();
    *(half4*)(Q_shared + (((int)threadIdx.x) * 4)) = *(half4*)(Q + ((((((((int)blockIdx.x) >> 12) * 524288) + (((((int)blockIdx.x) & 4095) >> 6) * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)));
    *(half4*)(Q_shared + ((((int)threadIdx.x) * 4) + 512)) = *(half4*)(Q + (((((((((int)blockIdx.x) >> 12) * 524288) + (((((int)blockIdx.x) & 4095) >> 6) * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 262144));
    *(half4*)(K_shared + (((int)threadIdx.x) * 4)) = *(half4*)(K + ((((((((int)blockIdx.x) >> 12) * 524288) + ((((int)blockIdx.x) & 63) * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)));
    *(half4*)(K_shared + ((((int)threadIdx.x) * 4) + 512)) = *(half4*)(K + (((((((((int)blockIdx.x) >> 12) * 524288) + ((((int)blockIdx.x) & 63) * 4096)) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 262144));
    __syncthreads();
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16))] * K_shared[(((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16))]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16))] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 128)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16))] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 256)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16))] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 384)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 128)] * K_shared[(((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16))]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 128)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 128)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 128)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 256)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 128)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 384)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 256)] * K_shared[(((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16))]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 256)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 128)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 256)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 256)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 256)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 384)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 384)] * K_shared[(((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16))]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 384)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 128)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 384)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 256)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 384)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 384)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 129)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 257)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 1)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 385)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 129)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 129)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 129)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 129)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 257)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 129)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 385)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 257)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 257)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 129)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 257)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 257)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 257)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 385)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 385)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 385)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 129)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 385)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 257)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 385)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 385)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 130)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 258)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 2)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 386)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 130)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 130)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 130)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 130)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 258)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 130)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 386)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 258)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 258)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 130)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 258)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 258)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 258)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 386)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 386)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 386)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 130)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 386)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 258)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 386)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 386)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 3)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 3)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 131)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 3)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 259)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 3)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 387)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 131)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 131)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 131)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 131)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 259)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 131)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 387)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 259)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 259)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 131)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 259)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 259)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 259)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 387)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 387)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 387)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 131)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 387)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 259)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 387)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 387)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 4)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 4)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 132)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 4)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 260)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 4)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 388)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 132)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 132)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 132)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 132)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 260)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 132)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 388)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 260)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 260)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 132)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 260)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 260)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 260)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 388)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 388)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 388)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 132)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 388)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 260)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 388)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 388)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 5)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 5)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 133)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 5)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 261)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 5)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 389)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 133)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 133)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 133)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 133)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 261)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 133)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 389)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 261)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 261)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 133)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 261)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 261)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 261)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 389)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 389)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 389)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 133)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 389)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 261)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 389)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 389)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 6)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 6)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 134)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 6)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 262)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 6)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 390)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 134)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 134)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 134)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 134)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 262)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 134)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 390)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 262)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 262)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 134)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 262)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 262)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 262)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 390)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 390)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 390)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 134)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 390)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 262)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 390)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 390)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 7)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 7)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 135)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 7)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 263)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 7)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 391)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 135)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 135)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 135)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 135)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 263)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 135)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 391)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 263)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 263)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 135)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 263)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 263)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 263)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 391)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 391)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 391)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 135)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 391)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 263)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 391)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 391)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 8)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 8)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 136)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 8)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 264)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 8)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 392)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 136)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 136)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 136)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 136)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 264)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 136)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 392)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 264)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 264)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 136)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 264)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 264)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 264)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 392)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 392)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 392)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 136)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 392)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 264)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 392)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 392)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 9)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 9)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 137)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 9)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 265)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 9)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 393)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 137)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 137)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 137)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 137)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 265)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 137)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 393)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 265)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 265)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 137)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 265)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 265)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 265)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 393)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 393)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 393)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 137)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 393)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 265)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 393)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 393)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 10)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 10)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 138)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 10)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 266)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 10)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 394)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 138)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 138)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 138)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 138)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 266)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 138)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 394)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 266)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 266)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 138)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 266)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 266)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 266)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 394)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 394)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 394)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 138)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 394)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 266)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 394)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 394)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 11)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 11)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 139)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 11)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 267)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 11)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 395)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 139)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 139)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 139)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 139)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 267)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 139)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 395)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 267)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 267)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 139)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 267)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 267)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 267)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 395)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 395)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 395)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 139)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 395)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 267)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 395)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 395)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 12)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 12)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 140)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 12)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 268)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 12)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 396)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 140)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 140)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 140)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 140)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 268)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 140)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 396)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 268)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 268)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 140)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 268)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 268)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 268)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 396)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 396)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 396)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 140)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 396)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 268)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 396)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 396)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 13)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 13)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 141)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 13)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 269)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 13)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 397)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 141)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 141)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 141)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 141)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 269)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 141)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 397)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 269)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 269)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 141)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 269)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 269)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 269)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 397)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 397)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 397)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 141)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 397)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 269)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 397)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 397)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 14)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 14)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 142)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 14)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 270)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 14)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 398)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 142)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 142)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 142)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 142)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 270)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 142)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 398)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 270)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 270)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 142)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 270)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 270)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 270)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 398)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 398)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 398)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 142)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 398)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 270)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 398)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 398)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 15)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 15)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 143)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 15)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 271)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 15)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 399)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 143)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 143)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 143)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 143)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 271)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 143)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 399)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 271)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 271)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 143)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 271)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 271)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 271)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 399)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 399)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 399)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 143)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 399)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 271)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 6) * 512) + (((((int)threadIdx.x) & 63) >> 3) * 16)) + 399)] * K_shared[((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 16)) + 399)]) * __float2half_rn(8.838835e-02f)));
  }
  QK[(((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7))] = QK_local[0];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 8)] = QK_local[1];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 16)] = QK_local[2];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 24)] = QK_local[3];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 16384)] = QK_local[4];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 16392)] = QK_local[5];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 16400)] = QK_local[6];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 16408)] = QK_local[7];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 32768)] = QK_local[8];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 32776)] = QK_local[9];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 32784)] = QK_local[10];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 32792)] = QK_local[11];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 49152)] = QK_local[12];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 49160)] = QK_local[13];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 49168)] = QK_local[14];
  QK[((((((((((int)blockIdx.x) >> 12) * 8388608) + ((((int)threadIdx.x) >> 6) * 4194304)) + (((((int)blockIdx.x) & 4095) >> 6) * 65536)) + (((((int)threadIdx.x) & 63) >> 3) * 2048)) + ((((int)blockIdx.x) & 63) * 32)) + (((int)threadIdx.x) & 7)) + 49176)] = QK_local[15];
}

