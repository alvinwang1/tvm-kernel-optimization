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
extern "C" __global__ void __launch_bounds__(512) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK);
extern "C" __global__ void __launch_bounds__(512) main_kernel(half* __restrict__ K, half* __restrict__ Q, half* __restrict__ QK) {
  half QK_local[8];
  __shared__ half Q_shared[1536];
  __shared__ half K_shared[6144];
  QK_local[0] = __float2half_rn(0.000000e+00f);
  QK_local[4] = __float2half_rn(0.000000e+00f);
  QK_local[1] = __float2half_rn(0.000000e+00f);
  QK_local[5] = __float2half_rn(0.000000e+00f);
  QK_local[2] = __float2half_rn(0.000000e+00f);
  QK_local[6] = __float2half_rn(0.000000e+00f);
  QK_local[3] = __float2half_rn(0.000000e+00f);
  QK_local[7] = __float2half_rn(0.000000e+00f);
  if (((int)threadIdx.x) < 64) {
    *(uint4*)(Q_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(Q + (((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 1) * 8)));
  }
  *(half4*)(K_shared + (((int)threadIdx.x) * 4)) = *(half4*)(K + ((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)));
__asm__ __volatile__("cp.async.commit_group;");

  if (((int)threadIdx.x) < 64) {
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 512)) = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 1) * 8)) + 16));
  }
  *(half4*)(K_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(half4*)(K + (((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 16));
__asm__ __volatile__("cp.async.commit_group;");

  if (((int)threadIdx.x) < 64) {
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 1024)) = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 1) * 8)) + 32));
  }
  *(half4*)(K_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(half4*)(K + (((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 32));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[((((int)threadIdx.x) >> 6) * 64)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[((((int)threadIdx.x) >> 6) * 64)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 16)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 16)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 32)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 32)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 48)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 48)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 17)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 17)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 33)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 33)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 49)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 49)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 2)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 2)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 18)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 18)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 34)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 34)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 50)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 50)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 3)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 3)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 19)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 19)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 35)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 35)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 51)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 51)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 4)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 4)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 20)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 20)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 36)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 36)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 52)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 52)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 5)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 5)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 21)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 21)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 37)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 37)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 53)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 53)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 6)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 6)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 22)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 22)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 38)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 38)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 54)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 54)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 7)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 7)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 23)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 23)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 39)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 39)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 55)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 55)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 8)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 8)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 24)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 24)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 40)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 40)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 56)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 56)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 9)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 9)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 25)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 25)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 41)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 41)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 57)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 57)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 10)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 10)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 26)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 26)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 42)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 42)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 58)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 58)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 11)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 11)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 27)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 27)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 43)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 43)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 59)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 59)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 12)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 12)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 28)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 28)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 44)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 44)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 60)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 60)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 13)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 13)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 29)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 29)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 45)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 45)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 61)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 61)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 14)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 14)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 30)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 30)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 46)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 46)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 62)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 62)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 15)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 15)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 31)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 31)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 47)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 47)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 63)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 63)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    *(uint4*)(Q_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 1) * 8)) + 48));
  }
  *(half4*)(K_shared + (((int)threadIdx.x) * 4)) = *(half4*)(K + (((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 48));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 512)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 512)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 528)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 528)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 544)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 544)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 560)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 560)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 513)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 513)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 529)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 529)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 545)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 545)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 561)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 561)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 514)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 514)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 530)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 530)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 546)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 546)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 562)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 562)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 515)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 515)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 531)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 531)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 547)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 547)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 563)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 563)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 516)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 516)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 532)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 532)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 548)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 548)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 564)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 564)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 517)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 517)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 533)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 533)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 549)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 549)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 565)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 565)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 518)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 518)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 534)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 534)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 550)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 550)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 566)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 566)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 519)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 519)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 535)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 535)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 551)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 551)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 567)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 567)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 520)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 520)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 536)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 536)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 552)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 552)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 568)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 568)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 521)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 521)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 537)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 537)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 553)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 553)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 569)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 569)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 522)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 522)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 538)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 538)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 554)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 554)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 570)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 570)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 523)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 523)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 539)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 539)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 555)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 555)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 571)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 571)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 524)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 524)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 540)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 540)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 556)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 556)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 572)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 572)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 525)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 525)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 541)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 541)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 557)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 557)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 573)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 573)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 526)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 526)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 542)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 542)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 558)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 558)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 574)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 574)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 527)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 527)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 543)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 543)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 559)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 559)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 575)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 575)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 512)) = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 1) * 8)) + 64));
  }
  *(half4*)(K_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(half4*)(K + (((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 64));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1024)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1024)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5120)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1040)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1040)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5120)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1056)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1056)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5120)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1072)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1072)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5120)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1025)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1025)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5121)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1041)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1041)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5121)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1057)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1057)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5121)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1073)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1073)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5121)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1026)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1026)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5122)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1042)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1042)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5122)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1058)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1058)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5122)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1074)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1074)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5122)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1027)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1027)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5123)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1043)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1043)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5123)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1059)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1059)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5123)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1075)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1075)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5123)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1028)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1028)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5124)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1044)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1044)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5124)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1060)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1060)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5124)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1076)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1076)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5124)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1029)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1029)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5125)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1045)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1045)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5125)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1061)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1061)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5125)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1077)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1077)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5125)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1030)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1030)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5126)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1046)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1046)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5126)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1062)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1062)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5126)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1078)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1078)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5126)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1031)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1031)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5127)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1047)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1047)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5127)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1063)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1063)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5127)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1079)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1079)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5127)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1032)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1032)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5128)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1048)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1048)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5128)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1064)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1064)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5128)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1080)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1080)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5128)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1033)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1033)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5129)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1049)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1049)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5129)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1065)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1065)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5129)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1081)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1081)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5129)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1034)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1034)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5130)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1050)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1050)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5130)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1066)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1066)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5130)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1082)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1082)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5130)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1035)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1035)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5131)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1051)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1051)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5131)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1067)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1067)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5131)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1083)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1083)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5131)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1036)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1036)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5132)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1052)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1052)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5132)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1068)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1068)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5132)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1084)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1084)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5132)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1037)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1037)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5133)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1053)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1053)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5133)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1069)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1069)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5133)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1085)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1085)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5133)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1038)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1038)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5134)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1054)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1054)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5134)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1070)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1070)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5134)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1086)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1086)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5134)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1039)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1039)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5135)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1055)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1055)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5135)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1071)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1071)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5135)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1087)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1087)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5135)]) * __float2half_rn(8.838835e-02f)));
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 1024)) = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 1) * 8)) + 80));
  }
  *(half4*)(K_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(half4*)(K + (((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 80));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[((((int)threadIdx.x) >> 6) * 64)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[((((int)threadIdx.x) >> 6) * 64)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 16)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 16)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 32)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 32)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 48)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 48)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 17)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 17)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 33)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 33)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 49)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 49)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 2)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 2)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 18)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 18)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 34)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 34)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 50)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 50)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 3)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 3)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 19)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 19)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 35)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 35)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 51)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 51)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 4)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 4)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 20)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 20)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 36)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 36)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 52)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 52)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 5)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 5)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 21)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 21)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 37)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 37)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 53)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 53)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 6)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 6)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 22)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 22)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 38)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 38)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 54)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 54)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 7)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 7)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 23)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 23)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 39)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 39)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 55)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 55)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 8)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 8)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 24)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 24)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 40)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 40)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 56)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 56)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 9)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 9)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 25)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 25)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 41)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 41)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 57)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 57)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 10)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 10)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 26)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 26)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 42)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 42)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 58)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 58)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 11)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 11)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 27)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 27)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 43)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 43)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 59)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 59)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 12)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 12)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 28)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 28)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 44)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 44)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 60)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 60)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 13)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 13)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 29)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 29)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 45)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 45)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 61)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 61)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 14)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 14)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 30)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 30)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 46)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 46)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 62)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 62)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 15)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 15)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 31)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 31)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 47)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 47)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 63)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 63)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    *(uint4*)(Q_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 1) * 8)) + 96));
  }
  *(half4*)(K_shared + (((int)threadIdx.x) * 4)) = *(half4*)(K + (((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 96));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 512)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 512)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 528)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 528)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 544)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 544)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 560)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 560)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 513)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 513)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 529)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 529)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 545)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 545)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 561)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 561)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 514)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 514)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 530)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 530)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 546)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 546)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 562)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 562)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 515)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 515)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 531)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 531)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 547)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 547)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 563)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 563)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 516)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 516)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 532)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 532)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 548)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 548)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 564)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 564)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 517)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 517)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 533)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 533)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 549)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 549)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 565)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 565)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 518)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 518)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 534)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 534)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 550)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 550)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 566)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 566)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 519)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 519)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 535)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 535)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 551)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 551)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 567)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 567)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 520)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 520)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 536)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 536)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 552)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 552)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 568)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 568)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 521)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 521)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 537)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 537)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 553)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 553)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 569)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 569)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 522)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 522)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 538)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 538)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 554)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 554)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 570)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 570)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 523)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 523)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 539)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 539)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 555)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 555)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 571)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 571)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 524)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 524)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 540)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 540)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 556)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 556)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 572)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 572)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 525)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 525)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 541)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 541)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 557)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 557)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 573)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 573)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 526)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 526)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 542)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 542)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 558)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 558)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 574)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 574)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 527)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 527)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 543)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 543)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 559)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 559)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 575)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 575)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 512)) = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 1) * 8)) + 112));
  }
  *(half4*)(K_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(half4*)(K + (((((((int)blockIdx.x) >> 2) * 16384) + ((((int)threadIdx.x) >> 2) * 128)) + ((((int)threadIdx.x) & 3) * 4)) + 112));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1024)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1024)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5120)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1040)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1040)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5120)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1056)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1056)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5120)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1072)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1072)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5120)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1025)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1025)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5121)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1041)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1041)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5121)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1057)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1057)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5121)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1073)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1073)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5121)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1026)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1026)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5122)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1042)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1042)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5122)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1058)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1058)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5122)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1074)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1074)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5122)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1027)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1027)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5123)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1043)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1043)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5123)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1059)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1059)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5123)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1075)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1075)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5123)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1028)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1028)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5124)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1044)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1044)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5124)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1060)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1060)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5124)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1076)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1076)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5124)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1029)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1029)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5125)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1045)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1045)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5125)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1061)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1061)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5125)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1077)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1077)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5125)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1030)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1030)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5126)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1046)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1046)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5126)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1062)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1062)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5126)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1078)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1078)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5126)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1031)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1031)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5127)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1047)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1047)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5127)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1063)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1063)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5127)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1079)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1079)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5127)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1032)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1032)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5128)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1048)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1048)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5128)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1064)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1064)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5128)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1080)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1080)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5128)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1033)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1033)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5129)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1049)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1049)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5129)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1065)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1065)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5129)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1081)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1081)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5129)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1034)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1034)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5130)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1050)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1050)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5130)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1066)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1066)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5130)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1082)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1082)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5130)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1035)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1035)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5131)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1051)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1051)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5131)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1067)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1067)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5131)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1083)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1083)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5131)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1036)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1036)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5132)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1052)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1052)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5132)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1068)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1068)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5132)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1084)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1084)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5132)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1037)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1037)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5133)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1053)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1053)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5133)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1069)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1069)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5133)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1085)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1085)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5133)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1038)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1038)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5134)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1054)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1054)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5134)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1070)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1070)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5134)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1086)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1086)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5134)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1039)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1039)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5135)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1055)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1055)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5135)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1071)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1071)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5135)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1087)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1087)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5135)]) * __float2half_rn(8.838835e-02f)));
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[((((int)threadIdx.x) >> 6) * 64)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[((((int)threadIdx.x) >> 6) * 64)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 16)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 16)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 32)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 32)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 48)] * K_shared[((((int)threadIdx.x) & 63) * 16)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 48)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1024)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 1)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 17)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 17)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 33)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 33)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 49)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 49)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1025)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 2)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 2)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 18)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 18)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 34)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 34)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 50)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 50)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1026)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 3)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 3)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 19)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 19)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 35)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 35)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 51)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 51)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1027)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 4)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 4)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 20)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 20)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 36)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 36)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 52)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 52)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1028)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 5)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 5)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 21)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 21)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 37)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 37)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 53)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 53)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1029)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 6)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 6)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 22)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 22)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 38)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 38)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 54)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 54)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1030)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 7)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 7)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 23)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 23)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 39)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 39)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 55)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 55)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1031)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 8)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 8)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 24)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 24)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 40)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 40)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 56)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 56)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1032)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 9)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 9)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 25)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 25)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 41)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 41)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 57)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 57)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1033)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 10)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 10)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 26)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 26)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 42)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 42)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 58)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 58)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1034)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 11)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 11)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 27)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 27)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 43)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 43)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 59)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 59)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1035)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 12)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 12)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 28)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 28)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 44)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 44)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 60)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 60)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1036)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 13)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 13)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 29)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 29)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 45)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 45)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 61)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 61)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1037)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 14)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 14)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 30)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 30)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 46)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 46)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 62)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 62)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1038)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 15)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 15)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 31)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 31)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 47)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 47)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 63)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 63)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 1039)]) * __float2half_rn(8.838835e-02f)));
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 512)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 512)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 528)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 528)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 544)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 544)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 560)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 560)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3072)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 513)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 513)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 529)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 529)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 545)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 545)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 561)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 561)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3073)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 514)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 514)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 530)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 530)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 546)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 546)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 562)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 562)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3074)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 515)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 515)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 531)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 531)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 547)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 547)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 563)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 563)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3075)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 516)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 516)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 532)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 532)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 548)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 548)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 564)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 564)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3076)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 517)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 517)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 533)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 533)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 549)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 549)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 565)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 565)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3077)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 518)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 518)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 534)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 534)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 550)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 550)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 566)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 566)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3078)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 519)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 519)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 535)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 535)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 551)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 551)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 567)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 567)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3079)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 520)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 520)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 536)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 536)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 552)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 552)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 568)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 568)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3080)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 521)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 521)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 537)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 537)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 553)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 553)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 569)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 569)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3081)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 522)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 522)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 538)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 538)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 554)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 554)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 570)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 570)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3082)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 523)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 523)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 539)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 539)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 555)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 555)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 571)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 571)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3083)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 524)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 524)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 540)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 540)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 556)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 556)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 572)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 572)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3084)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 525)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 525)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 541)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 541)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 557)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 557)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 573)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 573)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3085)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 526)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 526)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 542)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 542)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 558)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 558)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 574)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 574)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3086)]) * __float2half_rn(8.838835e-02f)));
  QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 527)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 527)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 543)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 543)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 559)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 559)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 575)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
  QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 6) * 64) + 575)] * K_shared[(((((int)threadIdx.x) & 63) * 16) + 3087)]) * __float2half_rn(8.838835e-02f)));
  QK[(((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63))] = QK_local[0];
  QK[((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 64)] = QK_local[4];
  QK[((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 128)] = QK_local[1];
  QK[((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 192)] = QK_local[5];
  QK[((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 256)] = QK_local[2];
  QK[((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 320)] = QK_local[6];
  QK[((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 384)] = QK_local[3];
  QK[((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 6) * 512)) + (((int)threadIdx.x) & 63)) + 448)] = QK_local[7];
}

