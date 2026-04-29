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
  half QK_local[64];
  __shared__ half Q_shared[1024];
  __shared__ half K_shared[8192];
  QK_local[0] = __float2half_rn(0.000000e+00f);
  QK_local[16] = __float2half_rn(0.000000e+00f);
  QK_local[32] = __float2half_rn(0.000000e+00f);
  QK_local[48] = __float2half_rn(0.000000e+00f);
  QK_local[1] = __float2half_rn(0.000000e+00f);
  QK_local[17] = __float2half_rn(0.000000e+00f);
  QK_local[33] = __float2half_rn(0.000000e+00f);
  QK_local[49] = __float2half_rn(0.000000e+00f);
  QK_local[2] = __float2half_rn(0.000000e+00f);
  QK_local[18] = __float2half_rn(0.000000e+00f);
  QK_local[34] = __float2half_rn(0.000000e+00f);
  QK_local[50] = __float2half_rn(0.000000e+00f);
  QK_local[3] = __float2half_rn(0.000000e+00f);
  QK_local[19] = __float2half_rn(0.000000e+00f);
  QK_local[35] = __float2half_rn(0.000000e+00f);
  QK_local[51] = __float2half_rn(0.000000e+00f);
  QK_local[4] = __float2half_rn(0.000000e+00f);
  QK_local[20] = __float2half_rn(0.000000e+00f);
  QK_local[36] = __float2half_rn(0.000000e+00f);
  QK_local[52] = __float2half_rn(0.000000e+00f);
  QK_local[5] = __float2half_rn(0.000000e+00f);
  QK_local[21] = __float2half_rn(0.000000e+00f);
  QK_local[37] = __float2half_rn(0.000000e+00f);
  QK_local[53] = __float2half_rn(0.000000e+00f);
  QK_local[6] = __float2half_rn(0.000000e+00f);
  QK_local[22] = __float2half_rn(0.000000e+00f);
  QK_local[38] = __float2half_rn(0.000000e+00f);
  QK_local[54] = __float2half_rn(0.000000e+00f);
  QK_local[7] = __float2half_rn(0.000000e+00f);
  QK_local[23] = __float2half_rn(0.000000e+00f);
  QK_local[39] = __float2half_rn(0.000000e+00f);
  QK_local[55] = __float2half_rn(0.000000e+00f);
  QK_local[8] = __float2half_rn(0.000000e+00f);
  QK_local[24] = __float2half_rn(0.000000e+00f);
  QK_local[40] = __float2half_rn(0.000000e+00f);
  QK_local[56] = __float2half_rn(0.000000e+00f);
  QK_local[9] = __float2half_rn(0.000000e+00f);
  QK_local[25] = __float2half_rn(0.000000e+00f);
  QK_local[41] = __float2half_rn(0.000000e+00f);
  QK_local[57] = __float2half_rn(0.000000e+00f);
  QK_local[10] = __float2half_rn(0.000000e+00f);
  QK_local[26] = __float2half_rn(0.000000e+00f);
  QK_local[42] = __float2half_rn(0.000000e+00f);
  QK_local[58] = __float2half_rn(0.000000e+00f);
  QK_local[11] = __float2half_rn(0.000000e+00f);
  QK_local[27] = __float2half_rn(0.000000e+00f);
  QK_local[43] = __float2half_rn(0.000000e+00f);
  QK_local[59] = __float2half_rn(0.000000e+00f);
  QK_local[12] = __float2half_rn(0.000000e+00f);
  QK_local[28] = __float2half_rn(0.000000e+00f);
  QK_local[44] = __float2half_rn(0.000000e+00f);
  QK_local[60] = __float2half_rn(0.000000e+00f);
  QK_local[13] = __float2half_rn(0.000000e+00f);
  QK_local[29] = __float2half_rn(0.000000e+00f);
  QK_local[45] = __float2half_rn(0.000000e+00f);
  QK_local[61] = __float2half_rn(0.000000e+00f);
  QK_local[14] = __float2half_rn(0.000000e+00f);
  QK_local[30] = __float2half_rn(0.000000e+00f);
  QK_local[46] = __float2half_rn(0.000000e+00f);
  QK_local[62] = __float2half_rn(0.000000e+00f);
  QK_local[15] = __float2half_rn(0.000000e+00f);
  QK_local[31] = __float2half_rn(0.000000e+00f);
  QK_local[47] = __float2half_rn(0.000000e+00f);
  QK_local[63] = __float2half_rn(0.000000e+00f);
  for (int k1_0 = 0; k1_0 < 8; ++k1_0) {
    __syncthreads();
    *(half2*)(Q_shared + (((int)threadIdx.x) * 2)) = *(half2*)(Q + (((((((int)blockIdx.x) >> 2) * 8192) + ((((int)threadIdx.x) >> 3) * 128)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 7) * 2)));
    *(half4*)(K_shared + (((int)threadIdx.x) * 4)) = *(half4*)(K + ((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)blockIdx.x) & 3) * 65536)) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)));
    *(half4*)(K_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(half4*)(K + (((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)blockIdx.x) & 3) * 65536)) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 16384));
    *(half4*)(K_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(half4*)(K + (((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)blockIdx.x) & 3) * 65536)) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 32768));
    *(half4*)(K_shared + ((((int)threadIdx.x) * 4) + 6144)) = *(half4*)(K + (((((((((int)blockIdx.x) >> 7) * 262144) + ((((int)blockIdx.x) & 3) * 65536)) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 49152));
    __syncthreads();
    QK_local[0] = (QK_local[0] + ((Q_shared[((((int)threadIdx.x) >> 7) * 256)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[((((int)threadIdx.x) >> 7) * 256)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[((((int)threadIdx.x) >> 7) * 256)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[((((int)threadIdx.x) >> 7) * 256)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 16)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 16)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 16)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 16)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 32)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 32)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 32)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 32)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 48)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 48)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 48)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 48)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 64)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 64)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 64)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 64)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 80)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 80)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 80)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 80)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 96)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 96)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 96)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 96)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 112)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 112)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 112)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 112)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 1)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 1)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 1)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 1)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 17)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 17)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 17)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 17)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 33)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 33)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 33)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 33)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 49)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 49)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 49)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 49)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 65)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 65)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 65)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 65)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 81)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 81)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 81)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 81)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 97)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 97)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 97)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 97)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 113)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 113)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 113)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 113)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 2)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 2)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 2)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 2)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 18)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 18)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 18)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 18)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 34)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 34)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 34)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 34)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 50)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 50)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 50)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 50)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 66)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 66)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 66)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 66)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 82)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 82)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 82)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 82)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 98)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 98)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 98)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 98)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 114)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 114)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 114)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 114)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 3)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 3)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 3)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 3)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 19)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 19)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 19)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 19)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 35)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 35)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 35)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 35)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 51)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 51)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 51)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 51)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 67)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 67)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 67)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 67)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 83)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 83)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 83)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 83)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 99)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 99)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 99)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 99)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 115)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 115)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 115)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 115)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 128)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 128)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 128)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 128)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 144)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 144)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 144)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 144)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 160)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 160)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 160)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 160)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 176)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 176)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 176)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 176)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 192)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 192)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 192)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 192)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 208)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 208)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 208)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 208)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 224)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 224)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 224)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 224)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 240)] * K_shared[((((int)threadIdx.x) & 127) * 16)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 240)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2048)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 240)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4096)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 240)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6144)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 129)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 129)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 129)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 129)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 145)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 145)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 145)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 145)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 161)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 161)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 161)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 161)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 177)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 177)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 177)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 177)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 193)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 193)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 193)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 193)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 209)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 209)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 209)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 209)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 225)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 225)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 225)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 225)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 241)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 1)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 241)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2049)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 241)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4097)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 241)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6145)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 130)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 130)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 130)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 130)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 146)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 146)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 146)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 146)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 162)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 162)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 162)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 162)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 178)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 178)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 178)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 178)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 194)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 194)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 194)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 194)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 210)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 210)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 210)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 210)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 226)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 226)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 226)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 226)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 242)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 242)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2050)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 242)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4098)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 242)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6146)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 131)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 131)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 131)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 131)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 147)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 147)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 147)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 147)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 163)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 163)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 163)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 163)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 179)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 179)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 179)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 179)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 195)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 195)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 195)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 195)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 211)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 211)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 211)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 211)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 227)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 227)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 227)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 227)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 243)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 3)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 243)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2051)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 243)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4099)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 243)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6147)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 4)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 4)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 4)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 4)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 20)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 20)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 20)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 20)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 36)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 36)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 36)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 36)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 52)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 52)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 52)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 52)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 68)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 68)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 68)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 68)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 84)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 84)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 84)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 84)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 100)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 100)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 100)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 100)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 116)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 116)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 116)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 116)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 5)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 5)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 5)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 5)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 21)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 21)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 21)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 21)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 37)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 37)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 37)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 37)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 53)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 53)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 53)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 53)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 69)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 69)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 69)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 69)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 85)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 85)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 85)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 85)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 101)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 101)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 101)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 101)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 117)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 117)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 117)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 117)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 6)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 6)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 6)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 6)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 22)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 22)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 22)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 22)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 38)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 38)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 38)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 38)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 54)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 54)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 54)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 54)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 70)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 70)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 70)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 70)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 86)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 86)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 86)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 86)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 102)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 102)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 102)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 102)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 118)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 118)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 118)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 118)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 7)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 7)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 7)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 7)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 23)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 23)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 23)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 23)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 39)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 39)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 39)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 39)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 55)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 55)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 55)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 55)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 71)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 71)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 71)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 71)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 87)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 87)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 87)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 87)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 103)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 103)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 103)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 103)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 119)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 119)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 119)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 119)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 132)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 132)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 132)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 132)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 148)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 148)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 148)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 148)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 164)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 164)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 164)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 164)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 180)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 180)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 180)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 180)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 196)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 196)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 196)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 196)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 212)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 212)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 212)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 212)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 228)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 228)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 228)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 228)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 244)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 244)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2052)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 244)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4100)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 244)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6148)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 133)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 133)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 133)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 133)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 149)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 149)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 149)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 149)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 165)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 165)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 165)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 165)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 181)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 181)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 181)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 181)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 197)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 197)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 197)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 197)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 213)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 213)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 213)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 213)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 229)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 229)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 229)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 229)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 245)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 5)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 245)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2053)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 245)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4101)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 245)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6149)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 134)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 134)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 134)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 134)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 150)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 150)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 150)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 150)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 166)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 166)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 166)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 166)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 182)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 182)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 182)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 182)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 198)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 198)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 198)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 198)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 214)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 214)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 214)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 214)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 230)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 230)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 230)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 230)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 246)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 246)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2054)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 246)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4102)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 246)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6150)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 135)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 135)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 135)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 135)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 151)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 151)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 151)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 151)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 167)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 167)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 167)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 167)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 183)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 183)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 183)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 183)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 199)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 199)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 199)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 199)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 215)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 215)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 215)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 215)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 231)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 231)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 231)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 231)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 247)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 7)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 247)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2055)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 247)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4103)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 247)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6151)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 8)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 8)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 8)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 8)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 24)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 24)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 24)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 24)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 40)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 40)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 40)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 40)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 56)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 56)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 56)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 56)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 72)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 72)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 72)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 72)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 88)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 88)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 88)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 88)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 104)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 104)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 104)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 104)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 120)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 120)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 120)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 120)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 9)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 9)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 9)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 9)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 25)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 25)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 25)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 25)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 41)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 41)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 41)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 41)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 57)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 57)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 57)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 57)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 73)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 73)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 73)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 73)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 89)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 89)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 89)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 89)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 105)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 105)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 105)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 105)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 121)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 121)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 121)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 121)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 10)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 10)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 10)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 10)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 26)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 26)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 26)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 26)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 42)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 42)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 42)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 42)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 58)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 58)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 58)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 58)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 74)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 74)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 74)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 74)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 90)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 90)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 90)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 90)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 106)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 106)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 106)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 106)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 122)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 122)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 122)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 122)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 11)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 11)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 11)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 11)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 27)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 27)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 27)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 27)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 43)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 43)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 43)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 43)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 59)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 59)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 59)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 59)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 75)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 75)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 75)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 75)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 91)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 91)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 91)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 91)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 107)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 107)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 107)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 107)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 123)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 123)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 123)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 123)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 136)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 136)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 136)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 136)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 152)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 152)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 152)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 152)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 168)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 168)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 168)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 168)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 184)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 184)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 184)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 184)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 200)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 200)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 200)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 200)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 216)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 216)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 216)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 216)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 232)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 232)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 232)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 232)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 248)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 8)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 248)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2056)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 248)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4104)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 248)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6152)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 137)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 137)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 137)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 137)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 153)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 153)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 153)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 153)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 169)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 169)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 169)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 169)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 185)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 185)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 185)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 185)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 201)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 201)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 201)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 201)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 217)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 217)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 217)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 217)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 233)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 233)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 233)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 233)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 249)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 9)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 249)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2057)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 249)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4105)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 249)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6153)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 138)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 138)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 138)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 138)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 154)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 154)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 154)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 154)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 170)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 170)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 170)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 170)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 186)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 186)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 186)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 186)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 202)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 202)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 202)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 202)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 218)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 218)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 218)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 218)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 234)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 234)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 234)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 234)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 250)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 10)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 250)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2058)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 250)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4106)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 250)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6154)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 139)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 139)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 139)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 139)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 155)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 155)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 155)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 155)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 171)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 171)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 171)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 171)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 187)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 187)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 187)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 187)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 203)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 203)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 203)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 203)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 219)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 219)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 219)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 219)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 235)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 235)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 235)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 235)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 251)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 11)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 251)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2059)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 251)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4107)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 251)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6155)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 12)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 12)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 12)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 12)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 28)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 28)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 28)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 28)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 44)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 44)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 44)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 44)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 60)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 60)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 60)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 60)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 76)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 76)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 76)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 76)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 92)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 92)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 92)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 92)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 108)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 108)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 108)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 108)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 124)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 124)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 124)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 124)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 13)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 13)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 13)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 13)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 29)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 29)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 29)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 29)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 45)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 45)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 45)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 45)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 61)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 61)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 61)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 61)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 77)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 77)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 77)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 77)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 93)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 93)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 93)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 93)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 109)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 109)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 109)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 109)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 125)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 125)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 125)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 125)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 14)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 14)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 14)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 14)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 30)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 30)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 30)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 30)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 46)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 46)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 46)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 46)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 62)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 62)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 62)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 62)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 78)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 78)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 78)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 78)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 94)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 94)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 94)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 94)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 110)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 110)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 110)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 110)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 126)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 126)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 126)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 126)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 15)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[16] = (QK_local[16] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 15)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[32] = (QK_local[32] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 15)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[48] = (QK_local[48] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 15)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 31)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[17] = (QK_local[17] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 31)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[33] = (QK_local[33] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 31)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[49] = (QK_local[49] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 31)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[2] = (QK_local[2] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 47)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[18] = (QK_local[18] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 47)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[34] = (QK_local[34] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 47)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[50] = (QK_local[50] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 47)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[3] = (QK_local[3] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 63)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[19] = (QK_local[19] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 63)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[35] = (QK_local[35] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 63)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[51] = (QK_local[51] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 63)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[4] = (QK_local[4] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 79)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[20] = (QK_local[20] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 79)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[36] = (QK_local[36] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 79)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[52] = (QK_local[52] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 79)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[5] = (QK_local[5] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 95)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[21] = (QK_local[21] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 95)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[37] = (QK_local[37] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 95)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[53] = (QK_local[53] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 95)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[6] = (QK_local[6] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 111)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[22] = (QK_local[22] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 111)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[38] = (QK_local[38] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 111)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[54] = (QK_local[54] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 111)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[7] = (QK_local[7] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 127)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[23] = (QK_local[23] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 127)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[39] = (QK_local[39] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 127)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[55] = (QK_local[55] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 127)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 140)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 140)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 140)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 140)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 156)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 156)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 156)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 156)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 172)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 172)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 172)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 172)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 188)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 188)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 188)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 188)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 204)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 204)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 204)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 204)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 220)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 220)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 220)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 220)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 236)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 236)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 236)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 236)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 252)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 12)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 252)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2060)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 252)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4108)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 252)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6156)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 141)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 141)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 141)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 141)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 157)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 157)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 157)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 157)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 173)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 173)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 173)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 173)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 189)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 189)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 189)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 189)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 205)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 205)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 205)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 205)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 221)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 221)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 221)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 221)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 237)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 237)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 237)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 237)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 253)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 13)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 253)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2061)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 253)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4109)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 253)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6157)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 142)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 142)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 142)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 142)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 158)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 158)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 158)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 158)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 174)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 174)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 174)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 174)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 190)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 190)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 190)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 190)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 206)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 206)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 206)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 206)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 222)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 222)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 222)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 222)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 238)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 238)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 238)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 238)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 254)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 14)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 254)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2062)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 254)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4110)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 254)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6158)]) * __float2half_rn(8.838835e-02f)));
    QK_local[8] = (QK_local[8] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 143)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[24] = (QK_local[24] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 143)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[40] = (QK_local[40] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 143)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[56] = (QK_local[56] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 143)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[9] = (QK_local[9] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 159)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[25] = (QK_local[25] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 159)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[41] = (QK_local[41] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 159)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[57] = (QK_local[57] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 159)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[10] = (QK_local[10] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 175)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[26] = (QK_local[26] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 175)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[42] = (QK_local[42] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 175)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[58] = (QK_local[58] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 175)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[11] = (QK_local[11] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 191)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[27] = (QK_local[27] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 191)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[43] = (QK_local[43] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 191)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[59] = (QK_local[59] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 191)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[12] = (QK_local[12] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 207)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[28] = (QK_local[28] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 207)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[44] = (QK_local[44] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 207)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[60] = (QK_local[60] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 207)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[13] = (QK_local[13] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 223)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[29] = (QK_local[29] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 223)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[45] = (QK_local[45] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 223)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[61] = (QK_local[61] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 223)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[14] = (QK_local[14] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 239)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[30] = (QK_local[30] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 239)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[46] = (QK_local[46] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 239)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[62] = (QK_local[62] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 239)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
    QK_local[15] = (QK_local[15] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 255)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 15)]) * __float2half_rn(8.838835e-02f)));
    QK_local[31] = (QK_local[31] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 255)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 2063)]) * __float2half_rn(8.838835e-02f)));
    QK_local[47] = (QK_local[47] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 255)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 4111)]) * __float2half_rn(8.838835e-02f)));
    QK_local[63] = (QK_local[63] + ((Q_shared[(((((int)threadIdx.x) >> 7) * 256) + 255)] * K_shared[(((((int)threadIdx.x) & 127) * 16) + 6159)]) * __float2half_rn(8.838835e-02f)));
  }
  QK[(((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127))] = QK_local[0];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 128)] = QK_local[16];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 256)] = QK_local[32];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 384)] = QK_local[48];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 2048)] = QK_local[1];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 2176)] = QK_local[17];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 2304)] = QK_local[33];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 2432)] = QK_local[49];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 4096)] = QK_local[2];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 4224)] = QK_local[18];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 4352)] = QK_local[34];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 4480)] = QK_local[50];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 6144)] = QK_local[3];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 6272)] = QK_local[19];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 6400)] = QK_local[35];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 6528)] = QK_local[51];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 8192)] = QK_local[4];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 8320)] = QK_local[20];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 8448)] = QK_local[36];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 8576)] = QK_local[52];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 10240)] = QK_local[5];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 10368)] = QK_local[21];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 10496)] = QK_local[37];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 10624)] = QK_local[53];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 12288)] = QK_local[6];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 12416)] = QK_local[22];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 12544)] = QK_local[38];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 12672)] = QK_local[54];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 14336)] = QK_local[7];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 14464)] = QK_local[23];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 14592)] = QK_local[39];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 14720)] = QK_local[55];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 16384)] = QK_local[8];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 16512)] = QK_local[24];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 16640)] = QK_local[40];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 16768)] = QK_local[56];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 18432)] = QK_local[9];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 18560)] = QK_local[25];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 18688)] = QK_local[41];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 18816)] = QK_local[57];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 20480)] = QK_local[10];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 20608)] = QK_local[26];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 20736)] = QK_local[42];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 20864)] = QK_local[58];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 22528)] = QK_local[11];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 22656)] = QK_local[27];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 22784)] = QK_local[43];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 22912)] = QK_local[59];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 24576)] = QK_local[12];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 24704)] = QK_local[28];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 24832)] = QK_local[44];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 24960)] = QK_local[60];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 26624)] = QK_local[13];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 26752)] = QK_local[29];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 26880)] = QK_local[45];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 27008)] = QK_local[61];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 28672)] = QK_local[14];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 28800)] = QK_local[30];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 28928)] = QK_local[46];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 29056)] = QK_local[62];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 30720)] = QK_local[15];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 30848)] = QK_local[31];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 30976)] = QK_local[47];
  QK[((((((((int)blockIdx.x) >> 2) * 131072) + ((((int)threadIdx.x) >> 7) * 32768)) + ((((int)blockIdx.x) & 3) * 512)) + (((int)threadIdx.x) & 127)) + 31104)] = QK_local[63];
}

