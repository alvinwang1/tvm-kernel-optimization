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
  half QK_local[128];
  __shared__ half Q_shared[8192];
  __shared__ half K_shared[2048];
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
  QK_local[16] = __float2half_rn(0.000000e+00f);
  QK_local[17] = __float2half_rn(0.000000e+00f);
  QK_local[18] = __float2half_rn(0.000000e+00f);
  QK_local[19] = __float2half_rn(0.000000e+00f);
  QK_local[20] = __float2half_rn(0.000000e+00f);
  QK_local[21] = __float2half_rn(0.000000e+00f);
  QK_local[22] = __float2half_rn(0.000000e+00f);
  QK_local[23] = __float2half_rn(0.000000e+00f);
  QK_local[24] = __float2half_rn(0.000000e+00f);
  QK_local[25] = __float2half_rn(0.000000e+00f);
  QK_local[26] = __float2half_rn(0.000000e+00f);
  QK_local[27] = __float2half_rn(0.000000e+00f);
  QK_local[28] = __float2half_rn(0.000000e+00f);
  QK_local[29] = __float2half_rn(0.000000e+00f);
  QK_local[30] = __float2half_rn(0.000000e+00f);
  QK_local[31] = __float2half_rn(0.000000e+00f);
  QK_local[32] = __float2half_rn(0.000000e+00f);
  QK_local[33] = __float2half_rn(0.000000e+00f);
  QK_local[34] = __float2half_rn(0.000000e+00f);
  QK_local[35] = __float2half_rn(0.000000e+00f);
  QK_local[36] = __float2half_rn(0.000000e+00f);
  QK_local[37] = __float2half_rn(0.000000e+00f);
  QK_local[38] = __float2half_rn(0.000000e+00f);
  QK_local[39] = __float2half_rn(0.000000e+00f);
  QK_local[40] = __float2half_rn(0.000000e+00f);
  QK_local[41] = __float2half_rn(0.000000e+00f);
  QK_local[42] = __float2half_rn(0.000000e+00f);
  QK_local[43] = __float2half_rn(0.000000e+00f);
  QK_local[44] = __float2half_rn(0.000000e+00f);
  QK_local[45] = __float2half_rn(0.000000e+00f);
  QK_local[46] = __float2half_rn(0.000000e+00f);
  QK_local[47] = __float2half_rn(0.000000e+00f);
  QK_local[48] = __float2half_rn(0.000000e+00f);
  QK_local[49] = __float2half_rn(0.000000e+00f);
  QK_local[50] = __float2half_rn(0.000000e+00f);
  QK_local[51] = __float2half_rn(0.000000e+00f);
  QK_local[52] = __float2half_rn(0.000000e+00f);
  QK_local[53] = __float2half_rn(0.000000e+00f);
  QK_local[54] = __float2half_rn(0.000000e+00f);
  QK_local[55] = __float2half_rn(0.000000e+00f);
  QK_local[56] = __float2half_rn(0.000000e+00f);
  QK_local[57] = __float2half_rn(0.000000e+00f);
  QK_local[58] = __float2half_rn(0.000000e+00f);
  QK_local[59] = __float2half_rn(0.000000e+00f);
  QK_local[60] = __float2half_rn(0.000000e+00f);
  QK_local[61] = __float2half_rn(0.000000e+00f);
  QK_local[62] = __float2half_rn(0.000000e+00f);
  QK_local[63] = __float2half_rn(0.000000e+00f);
  QK_local[64] = __float2half_rn(0.000000e+00f);
  QK_local[65] = __float2half_rn(0.000000e+00f);
  QK_local[66] = __float2half_rn(0.000000e+00f);
  QK_local[67] = __float2half_rn(0.000000e+00f);
  QK_local[68] = __float2half_rn(0.000000e+00f);
  QK_local[69] = __float2half_rn(0.000000e+00f);
  QK_local[70] = __float2half_rn(0.000000e+00f);
  QK_local[71] = __float2half_rn(0.000000e+00f);
  QK_local[72] = __float2half_rn(0.000000e+00f);
  QK_local[73] = __float2half_rn(0.000000e+00f);
  QK_local[74] = __float2half_rn(0.000000e+00f);
  QK_local[75] = __float2half_rn(0.000000e+00f);
  QK_local[76] = __float2half_rn(0.000000e+00f);
  QK_local[77] = __float2half_rn(0.000000e+00f);
  QK_local[78] = __float2half_rn(0.000000e+00f);
  QK_local[79] = __float2half_rn(0.000000e+00f);
  QK_local[80] = __float2half_rn(0.000000e+00f);
  QK_local[81] = __float2half_rn(0.000000e+00f);
  QK_local[82] = __float2half_rn(0.000000e+00f);
  QK_local[83] = __float2half_rn(0.000000e+00f);
  QK_local[84] = __float2half_rn(0.000000e+00f);
  QK_local[85] = __float2half_rn(0.000000e+00f);
  QK_local[86] = __float2half_rn(0.000000e+00f);
  QK_local[87] = __float2half_rn(0.000000e+00f);
  QK_local[88] = __float2half_rn(0.000000e+00f);
  QK_local[89] = __float2half_rn(0.000000e+00f);
  QK_local[90] = __float2half_rn(0.000000e+00f);
  QK_local[91] = __float2half_rn(0.000000e+00f);
  QK_local[92] = __float2half_rn(0.000000e+00f);
  QK_local[93] = __float2half_rn(0.000000e+00f);
  QK_local[94] = __float2half_rn(0.000000e+00f);
  QK_local[95] = __float2half_rn(0.000000e+00f);
  QK_local[96] = __float2half_rn(0.000000e+00f);
  QK_local[97] = __float2half_rn(0.000000e+00f);
  QK_local[98] = __float2half_rn(0.000000e+00f);
  QK_local[99] = __float2half_rn(0.000000e+00f);
  QK_local[100] = __float2half_rn(0.000000e+00f);
  QK_local[101] = __float2half_rn(0.000000e+00f);
  QK_local[102] = __float2half_rn(0.000000e+00f);
  QK_local[103] = __float2half_rn(0.000000e+00f);
  QK_local[104] = __float2half_rn(0.000000e+00f);
  QK_local[105] = __float2half_rn(0.000000e+00f);
  QK_local[106] = __float2half_rn(0.000000e+00f);
  QK_local[107] = __float2half_rn(0.000000e+00f);
  QK_local[108] = __float2half_rn(0.000000e+00f);
  QK_local[109] = __float2half_rn(0.000000e+00f);
  QK_local[110] = __float2half_rn(0.000000e+00f);
  QK_local[111] = __float2half_rn(0.000000e+00f);
  QK_local[112] = __float2half_rn(0.000000e+00f);
  QK_local[113] = __float2half_rn(0.000000e+00f);
  QK_local[114] = __float2half_rn(0.000000e+00f);
  QK_local[115] = __float2half_rn(0.000000e+00f);
  QK_local[116] = __float2half_rn(0.000000e+00f);
  QK_local[117] = __float2half_rn(0.000000e+00f);
  QK_local[118] = __float2half_rn(0.000000e+00f);
  QK_local[119] = __float2half_rn(0.000000e+00f);
  QK_local[120] = __float2half_rn(0.000000e+00f);
  QK_local[121] = __float2half_rn(0.000000e+00f);
  QK_local[122] = __float2half_rn(0.000000e+00f);
  QK_local[123] = __float2half_rn(0.000000e+00f);
  QK_local[124] = __float2half_rn(0.000000e+00f);
  QK_local[125] = __float2half_rn(0.000000e+00f);
  QK_local[126] = __float2half_rn(0.000000e+00f);
  QK_local[127] = __float2half_rn(0.000000e+00f);
  for (int k1_0 = 0; k1_0 < 4; ++k1_0) {
    __syncthreads();
    *(uint4*)(Q_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(Q + (((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 1024)) = *(uint4*)(Q + ((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 4096));
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 2048)) = *(uint4*)(Q + ((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 8192));
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 3072)) = *(uint4*)(Q + ((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 12288));
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 4096)) = *(uint4*)(Q + ((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 16384));
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 5120)) = *(uint4*)(Q + ((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 20480));
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 6144)) = *(uint4*)(Q + ((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 24576));
    *(uint4*)(Q_shared + ((((int)threadIdx.x) * 8) + 7168)) = *(uint4*)(Q + ((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 2) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 28672));
    *(half2*)(K_shared + (((int)threadIdx.x) * 2)) = *(half2*)(K + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)blockIdx.x) & 7) * 8192)) + ((((int)threadIdx.x) >> 4) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)));
    *(half2*)(K_shared + ((((int)threadIdx.x) * 2) + 256)) = *(half2*)(K + (((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)blockIdx.x) & 7) * 8192)) + ((((int)threadIdx.x) >> 4) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 1024));
    *(half2*)(K_shared + ((((int)threadIdx.x) * 2) + 512)) = *(half2*)(K + (((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)blockIdx.x) & 7) * 8192)) + ((((int)threadIdx.x) >> 4) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 2048));
    *(half2*)(K_shared + ((((int)threadIdx.x) * 2) + 768)) = *(half2*)(K + (((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)blockIdx.x) & 7) * 8192)) + ((((int)threadIdx.x) >> 4) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 3072));
    *(half2*)(K_shared + ((((int)threadIdx.x) * 2) + 1024)) = *(half2*)(K + (((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)blockIdx.x) & 7) * 8192)) + ((((int)threadIdx.x) >> 4) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 4096));
    *(half2*)(K_shared + ((((int)threadIdx.x) * 2) + 1280)) = *(half2*)(K + (((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)blockIdx.x) & 7) * 8192)) + ((((int)threadIdx.x) >> 4) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 5120));
    *(half2*)(K_shared + ((((int)threadIdx.x) * 2) + 1536)) = *(half2*)(K + (((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)blockIdx.x) & 7) * 8192)) + ((((int)threadIdx.x) >> 4) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 6144));
    *(half2*)(K_shared + ((((int)threadIdx.x) * 2) + 1792)) = *(half2*)(K + (((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)blockIdx.x) & 7) * 8192)) + ((((int)threadIdx.x) >> 4) * 128)) + (k1_0 * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 7168));
    __syncthreads();
    for (int k1_1 = 0; k1_1 < 4; ++k1_1) {
      QK_local[0] = (QK_local[0] + ((Q_shared[(((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8))] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[1] = (QK_local[1] + ((Q_shared[(((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8))] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 3)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 3)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 4)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 4)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 5)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 5)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 6)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 6)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[0] = (QK_local[0] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 7)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[1] = (QK_local[1] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 7)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 32)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 32)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 33)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 33)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 34)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 34)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 35)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 35)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 36)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 36)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 37)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 37)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 38)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 38)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[2] = (QK_local[2] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 39)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[3] = (QK_local[3] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 39)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 64)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 64)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 65)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 65)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 66)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 66)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 67)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 67)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 68)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 68)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 69)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 69)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 70)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 70)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[4] = (QK_local[4] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 71)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[5] = (QK_local[5] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 71)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 96)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 96)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 97)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 97)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 98)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 98)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 99)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 99)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 100)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 100)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 101)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 101)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 102)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 102)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[6] = (QK_local[6] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 103)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[7] = (QK_local[7] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 103)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 128)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 128)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 129)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 129)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 130)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 130)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 131)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 131)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 132)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 132)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 133)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 133)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 134)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 134)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[8] = (QK_local[8] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 135)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[9] = (QK_local[9] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 135)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 160)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 160)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 161)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 161)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 162)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 162)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 163)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 163)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 164)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 164)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 165)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 165)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 166)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 166)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[10] = (QK_local[10] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 167)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[11] = (QK_local[11] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 167)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 192)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 192)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 193)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 193)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 194)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 194)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 195)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 195)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 196)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 196)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 197)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 197)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 198)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 198)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[12] = (QK_local[12] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 199)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[13] = (QK_local[13] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 199)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 224)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 224)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 225)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 225)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 226)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 226)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 227)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 227)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 228)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 228)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 229)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 229)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 230)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 230)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[14] = (QK_local[14] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 231)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[15] = (QK_local[15] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 231)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 256)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 256)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 257)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 257)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 258)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 258)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 259)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 259)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 260)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 260)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 261)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 261)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 262)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 262)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[16] = (QK_local[16] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 263)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[17] = (QK_local[17] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 263)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 288)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 288)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 289)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 289)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 290)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 290)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 291)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 291)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 292)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 292)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 293)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 293)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 294)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 294)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[18] = (QK_local[18] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 295)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[19] = (QK_local[19] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 295)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 320)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 320)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 321)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 321)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 322)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 322)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 323)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 323)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 324)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 324)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 325)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 325)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 326)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 326)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[20] = (QK_local[20] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 327)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[21] = (QK_local[21] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 327)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 352)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 352)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 353)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 353)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 354)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 354)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 355)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 355)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 356)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 356)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 357)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 357)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 358)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 358)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[22] = (QK_local[22] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 359)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[23] = (QK_local[23] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 359)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 384)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 384)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 385)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 385)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 386)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 386)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 387)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 387)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 388)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 388)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 389)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 389)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 390)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 390)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[24] = (QK_local[24] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 391)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[25] = (QK_local[25] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 391)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 416)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 416)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 417)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 417)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 418)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 418)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 419)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 419)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 420)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 420)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 421)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 421)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 422)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 422)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[26] = (QK_local[26] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 423)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[27] = (QK_local[27] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 423)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 448)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 448)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 449)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 449)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 450)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 450)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 451)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 451)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 452)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 452)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 453)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 453)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 454)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 454)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[28] = (QK_local[28] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 455)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[29] = (QK_local[29] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 455)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 480)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 480)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 481)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 481)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 482)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 482)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 483)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 483)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 484)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 484)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 485)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 485)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 486)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 486)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[30] = (QK_local[30] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 487)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[31] = (QK_local[31] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 487)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[32] = (QK_local[32] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 512)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[33] = (QK_local[33] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 512)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[32] = (QK_local[32] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 513)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[33] = (QK_local[33] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 513)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[32] = (QK_local[32] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 514)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[33] = (QK_local[33] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 514)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[32] = (QK_local[32] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 515)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[33] = (QK_local[33] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 515)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[32] = (QK_local[32] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 516)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[33] = (QK_local[33] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 516)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[32] = (QK_local[32] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 517)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[33] = (QK_local[33] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 517)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[32] = (QK_local[32] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 518)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[33] = (QK_local[33] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 518)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[32] = (QK_local[32] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 519)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[33] = (QK_local[33] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 519)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[34] = (QK_local[34] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 544)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[35] = (QK_local[35] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 544)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[34] = (QK_local[34] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 545)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[35] = (QK_local[35] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 545)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[34] = (QK_local[34] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 546)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[35] = (QK_local[35] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 546)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[34] = (QK_local[34] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 547)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[35] = (QK_local[35] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 547)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[34] = (QK_local[34] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 548)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[35] = (QK_local[35] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 548)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[34] = (QK_local[34] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 549)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[35] = (QK_local[35] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 549)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[34] = (QK_local[34] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 550)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[35] = (QK_local[35] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 550)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[34] = (QK_local[34] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 551)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[35] = (QK_local[35] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 551)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[36] = (QK_local[36] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 576)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[37] = (QK_local[37] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 576)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[36] = (QK_local[36] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 577)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[37] = (QK_local[37] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 577)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[36] = (QK_local[36] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 578)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[37] = (QK_local[37] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 578)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[36] = (QK_local[36] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 579)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[37] = (QK_local[37] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 579)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[36] = (QK_local[36] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 580)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[37] = (QK_local[37] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 580)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[36] = (QK_local[36] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 581)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[37] = (QK_local[37] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 581)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[36] = (QK_local[36] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 582)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[37] = (QK_local[37] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 582)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[36] = (QK_local[36] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 583)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[37] = (QK_local[37] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 583)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[38] = (QK_local[38] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 608)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[39] = (QK_local[39] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 608)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[38] = (QK_local[38] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 609)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[39] = (QK_local[39] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 609)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[38] = (QK_local[38] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 610)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[39] = (QK_local[39] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 610)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[38] = (QK_local[38] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 611)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[39] = (QK_local[39] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 611)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[38] = (QK_local[38] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 612)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[39] = (QK_local[39] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 612)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[38] = (QK_local[38] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 613)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[39] = (QK_local[39] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 613)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[38] = (QK_local[38] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 614)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[39] = (QK_local[39] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 614)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[38] = (QK_local[38] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 615)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[39] = (QK_local[39] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 615)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[40] = (QK_local[40] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 640)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[41] = (QK_local[41] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 640)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[40] = (QK_local[40] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 641)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[41] = (QK_local[41] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 641)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[40] = (QK_local[40] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 642)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[41] = (QK_local[41] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 642)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[40] = (QK_local[40] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 643)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[41] = (QK_local[41] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 643)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[40] = (QK_local[40] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 644)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[41] = (QK_local[41] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 644)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[40] = (QK_local[40] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 645)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[41] = (QK_local[41] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 645)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[40] = (QK_local[40] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 646)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[41] = (QK_local[41] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 646)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[40] = (QK_local[40] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 647)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[41] = (QK_local[41] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 647)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[42] = (QK_local[42] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 672)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[43] = (QK_local[43] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 672)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[42] = (QK_local[42] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 673)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[43] = (QK_local[43] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 673)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[42] = (QK_local[42] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 674)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[43] = (QK_local[43] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 674)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[42] = (QK_local[42] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 675)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[43] = (QK_local[43] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 675)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[42] = (QK_local[42] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 676)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[43] = (QK_local[43] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 676)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[42] = (QK_local[42] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 677)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[43] = (QK_local[43] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 677)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[42] = (QK_local[42] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 678)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[43] = (QK_local[43] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 678)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[42] = (QK_local[42] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 679)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[43] = (QK_local[43] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 679)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[44] = (QK_local[44] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 704)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[45] = (QK_local[45] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 704)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[44] = (QK_local[44] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 705)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[45] = (QK_local[45] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 705)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[44] = (QK_local[44] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 706)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[45] = (QK_local[45] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 706)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[44] = (QK_local[44] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 707)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[45] = (QK_local[45] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 707)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[44] = (QK_local[44] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 708)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[45] = (QK_local[45] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 708)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[44] = (QK_local[44] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 709)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[45] = (QK_local[45] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 709)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[44] = (QK_local[44] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 710)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[45] = (QK_local[45] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 710)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[44] = (QK_local[44] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 711)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[45] = (QK_local[45] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 711)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[46] = (QK_local[46] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 736)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[47] = (QK_local[47] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 736)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[46] = (QK_local[46] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 737)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[47] = (QK_local[47] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 737)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[46] = (QK_local[46] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 738)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[47] = (QK_local[47] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 738)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[46] = (QK_local[46] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 739)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[47] = (QK_local[47] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 739)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[46] = (QK_local[46] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 740)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[47] = (QK_local[47] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 740)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[46] = (QK_local[46] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 741)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[47] = (QK_local[47] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 741)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[46] = (QK_local[46] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 742)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[47] = (QK_local[47] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 742)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[46] = (QK_local[46] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 743)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[47] = (QK_local[47] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 743)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[48] = (QK_local[48] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 768)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[49] = (QK_local[49] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 768)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[48] = (QK_local[48] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 769)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[49] = (QK_local[49] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 769)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[48] = (QK_local[48] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 770)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[49] = (QK_local[49] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 770)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[48] = (QK_local[48] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 771)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[49] = (QK_local[49] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 771)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[48] = (QK_local[48] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 772)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[49] = (QK_local[49] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 772)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[48] = (QK_local[48] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 773)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[49] = (QK_local[49] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 773)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[48] = (QK_local[48] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 774)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[49] = (QK_local[49] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 774)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[48] = (QK_local[48] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 775)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[49] = (QK_local[49] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 775)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[50] = (QK_local[50] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 800)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[51] = (QK_local[51] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 800)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[50] = (QK_local[50] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 801)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[51] = (QK_local[51] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 801)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[50] = (QK_local[50] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 802)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[51] = (QK_local[51] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 802)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[50] = (QK_local[50] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 803)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[51] = (QK_local[51] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 803)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[50] = (QK_local[50] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 804)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[51] = (QK_local[51] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 804)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[50] = (QK_local[50] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 805)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[51] = (QK_local[51] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 805)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[50] = (QK_local[50] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 806)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[51] = (QK_local[51] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 806)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[50] = (QK_local[50] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 807)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[51] = (QK_local[51] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 807)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[52] = (QK_local[52] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 832)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[53] = (QK_local[53] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 832)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[52] = (QK_local[52] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 833)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[53] = (QK_local[53] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 833)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[52] = (QK_local[52] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 834)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[53] = (QK_local[53] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 834)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[52] = (QK_local[52] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 835)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[53] = (QK_local[53] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 835)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[52] = (QK_local[52] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 836)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[53] = (QK_local[53] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 836)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[52] = (QK_local[52] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 837)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[53] = (QK_local[53] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 837)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[52] = (QK_local[52] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 838)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[53] = (QK_local[53] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 838)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[52] = (QK_local[52] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 839)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[53] = (QK_local[53] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 839)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[54] = (QK_local[54] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 864)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[55] = (QK_local[55] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 864)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[54] = (QK_local[54] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 865)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[55] = (QK_local[55] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 865)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[54] = (QK_local[54] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 866)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[55] = (QK_local[55] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 866)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[54] = (QK_local[54] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 867)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[55] = (QK_local[55] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 867)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[54] = (QK_local[54] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 868)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[55] = (QK_local[55] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 868)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[54] = (QK_local[54] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 869)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[55] = (QK_local[55] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 869)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[54] = (QK_local[54] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 870)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[55] = (QK_local[55] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 870)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[54] = (QK_local[54] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 871)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[55] = (QK_local[55] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 871)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[56] = (QK_local[56] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 896)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[57] = (QK_local[57] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 896)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[56] = (QK_local[56] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 897)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[57] = (QK_local[57] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 897)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[56] = (QK_local[56] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 898)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[57] = (QK_local[57] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 898)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[56] = (QK_local[56] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 899)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[57] = (QK_local[57] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 899)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[56] = (QK_local[56] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 900)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[57] = (QK_local[57] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 900)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[56] = (QK_local[56] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 901)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[57] = (QK_local[57] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 901)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[56] = (QK_local[56] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 902)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[57] = (QK_local[57] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 902)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[56] = (QK_local[56] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 903)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[57] = (QK_local[57] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 903)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[58] = (QK_local[58] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 928)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[59] = (QK_local[59] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 928)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[58] = (QK_local[58] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 929)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[59] = (QK_local[59] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 929)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[58] = (QK_local[58] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 930)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[59] = (QK_local[59] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 930)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[58] = (QK_local[58] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 931)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[59] = (QK_local[59] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 931)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[58] = (QK_local[58] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 932)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[59] = (QK_local[59] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 932)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[58] = (QK_local[58] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 933)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[59] = (QK_local[59] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 933)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[58] = (QK_local[58] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 934)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[59] = (QK_local[59] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 934)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[58] = (QK_local[58] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 935)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[59] = (QK_local[59] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 935)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[60] = (QK_local[60] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 960)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[61] = (QK_local[61] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 960)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[60] = (QK_local[60] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 961)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[61] = (QK_local[61] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 961)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[60] = (QK_local[60] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 962)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[61] = (QK_local[61] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 962)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[60] = (QK_local[60] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 963)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[61] = (QK_local[61] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 963)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[60] = (QK_local[60] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 964)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[61] = (QK_local[61] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 964)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[60] = (QK_local[60] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 965)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[61] = (QK_local[61] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 965)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[60] = (QK_local[60] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 966)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[61] = (QK_local[61] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 966)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[60] = (QK_local[60] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 967)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[61] = (QK_local[61] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 967)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[62] = (QK_local[62] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 992)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[63] = (QK_local[63] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 992)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[62] = (QK_local[62] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 993)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[63] = (QK_local[63] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 993)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[62] = (QK_local[62] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 994)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[63] = (QK_local[63] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 994)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[62] = (QK_local[62] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 995)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[63] = (QK_local[63] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 995)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[62] = (QK_local[62] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 996)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[63] = (QK_local[63] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 996)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[62] = (QK_local[62] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 997)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[63] = (QK_local[63] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 997)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[62] = (QK_local[62] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 998)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[63] = (QK_local[63] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 998)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[62] = (QK_local[62] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 999)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[63] = (QK_local[63] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 999)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[64] = (QK_local[64] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1024)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[65] = (QK_local[65] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1024)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[64] = (QK_local[64] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1025)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[65] = (QK_local[65] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1025)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[64] = (QK_local[64] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1026)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[65] = (QK_local[65] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1026)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[64] = (QK_local[64] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1027)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[65] = (QK_local[65] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1027)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[64] = (QK_local[64] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1028)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[65] = (QK_local[65] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1028)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[64] = (QK_local[64] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1029)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[65] = (QK_local[65] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1029)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[64] = (QK_local[64] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1030)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[65] = (QK_local[65] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1030)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[64] = (QK_local[64] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1031)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[65] = (QK_local[65] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1031)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[66] = (QK_local[66] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1056)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[67] = (QK_local[67] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1056)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[66] = (QK_local[66] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1057)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[67] = (QK_local[67] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1057)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[66] = (QK_local[66] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1058)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[67] = (QK_local[67] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1058)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[66] = (QK_local[66] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1059)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[67] = (QK_local[67] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1059)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[66] = (QK_local[66] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1060)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[67] = (QK_local[67] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1060)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[66] = (QK_local[66] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1061)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[67] = (QK_local[67] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1061)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[66] = (QK_local[66] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1062)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[67] = (QK_local[67] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1062)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[66] = (QK_local[66] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1063)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[67] = (QK_local[67] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1063)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[68] = (QK_local[68] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1088)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[69] = (QK_local[69] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1088)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[68] = (QK_local[68] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1089)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[69] = (QK_local[69] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1089)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[68] = (QK_local[68] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1090)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[69] = (QK_local[69] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1090)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[68] = (QK_local[68] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1091)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[69] = (QK_local[69] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1091)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[68] = (QK_local[68] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1092)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[69] = (QK_local[69] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1092)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[68] = (QK_local[68] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1093)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[69] = (QK_local[69] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1093)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[68] = (QK_local[68] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1094)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[69] = (QK_local[69] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1094)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[68] = (QK_local[68] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1095)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[69] = (QK_local[69] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1095)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[70] = (QK_local[70] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1120)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[71] = (QK_local[71] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1120)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[70] = (QK_local[70] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1121)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[71] = (QK_local[71] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1121)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[70] = (QK_local[70] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1122)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[71] = (QK_local[71] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1122)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[70] = (QK_local[70] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1123)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[71] = (QK_local[71] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1123)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[70] = (QK_local[70] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1124)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[71] = (QK_local[71] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1124)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[70] = (QK_local[70] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1125)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[71] = (QK_local[71] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1125)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[70] = (QK_local[70] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1126)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[71] = (QK_local[71] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1126)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[70] = (QK_local[70] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1127)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[71] = (QK_local[71] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1127)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[72] = (QK_local[72] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1152)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[73] = (QK_local[73] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1152)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[72] = (QK_local[72] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1153)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[73] = (QK_local[73] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1153)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[72] = (QK_local[72] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1154)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[73] = (QK_local[73] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1154)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[72] = (QK_local[72] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1155)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[73] = (QK_local[73] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1155)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[72] = (QK_local[72] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1156)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[73] = (QK_local[73] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1156)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[72] = (QK_local[72] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1157)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[73] = (QK_local[73] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1157)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[72] = (QK_local[72] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1158)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[73] = (QK_local[73] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1158)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[72] = (QK_local[72] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1159)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[73] = (QK_local[73] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1159)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[74] = (QK_local[74] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1184)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[75] = (QK_local[75] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1184)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[74] = (QK_local[74] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1185)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[75] = (QK_local[75] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1185)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[74] = (QK_local[74] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1186)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[75] = (QK_local[75] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1186)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[74] = (QK_local[74] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1187)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[75] = (QK_local[75] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1187)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[74] = (QK_local[74] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1188)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[75] = (QK_local[75] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1188)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[74] = (QK_local[74] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1189)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[75] = (QK_local[75] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1189)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[74] = (QK_local[74] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1190)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[75] = (QK_local[75] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1190)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[74] = (QK_local[74] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1191)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[75] = (QK_local[75] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1191)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[76] = (QK_local[76] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1216)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[77] = (QK_local[77] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1216)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[76] = (QK_local[76] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1217)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[77] = (QK_local[77] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1217)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[76] = (QK_local[76] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1218)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[77] = (QK_local[77] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1218)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[76] = (QK_local[76] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1219)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[77] = (QK_local[77] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1219)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[76] = (QK_local[76] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1220)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[77] = (QK_local[77] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1220)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[76] = (QK_local[76] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1221)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[77] = (QK_local[77] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1221)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[76] = (QK_local[76] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1222)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[77] = (QK_local[77] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1222)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[76] = (QK_local[76] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1223)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[77] = (QK_local[77] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1223)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[78] = (QK_local[78] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1248)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[79] = (QK_local[79] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1248)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[78] = (QK_local[78] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1249)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[79] = (QK_local[79] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1249)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[78] = (QK_local[78] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1250)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[79] = (QK_local[79] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1250)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[78] = (QK_local[78] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1251)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[79] = (QK_local[79] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1251)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[78] = (QK_local[78] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1252)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[79] = (QK_local[79] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1252)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[78] = (QK_local[78] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1253)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[79] = (QK_local[79] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1253)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[78] = (QK_local[78] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1254)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[79] = (QK_local[79] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1254)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[78] = (QK_local[78] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1255)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[79] = (QK_local[79] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1255)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[80] = (QK_local[80] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1280)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[81] = (QK_local[81] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1280)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[80] = (QK_local[80] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1281)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[81] = (QK_local[81] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1281)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[80] = (QK_local[80] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1282)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[81] = (QK_local[81] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1282)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[80] = (QK_local[80] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1283)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[81] = (QK_local[81] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1283)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[80] = (QK_local[80] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1284)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[81] = (QK_local[81] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1284)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[80] = (QK_local[80] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1285)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[81] = (QK_local[81] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1285)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[80] = (QK_local[80] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1286)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[81] = (QK_local[81] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1286)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[80] = (QK_local[80] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1287)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[81] = (QK_local[81] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1287)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[82] = (QK_local[82] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1312)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[83] = (QK_local[83] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1312)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[82] = (QK_local[82] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1313)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[83] = (QK_local[83] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1313)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[82] = (QK_local[82] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1314)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[83] = (QK_local[83] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1314)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[82] = (QK_local[82] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1315)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[83] = (QK_local[83] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1315)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[82] = (QK_local[82] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1316)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[83] = (QK_local[83] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1316)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[82] = (QK_local[82] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1317)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[83] = (QK_local[83] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1317)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[82] = (QK_local[82] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1318)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[83] = (QK_local[83] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1318)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[82] = (QK_local[82] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1319)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[83] = (QK_local[83] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1319)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[84] = (QK_local[84] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1344)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[85] = (QK_local[85] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1344)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[84] = (QK_local[84] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1345)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[85] = (QK_local[85] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1345)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[84] = (QK_local[84] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1346)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[85] = (QK_local[85] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1346)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[84] = (QK_local[84] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1347)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[85] = (QK_local[85] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1347)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[84] = (QK_local[84] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1348)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[85] = (QK_local[85] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1348)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[84] = (QK_local[84] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1349)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[85] = (QK_local[85] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1349)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[84] = (QK_local[84] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1350)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[85] = (QK_local[85] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1350)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[84] = (QK_local[84] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1351)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[85] = (QK_local[85] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1351)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[86] = (QK_local[86] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1376)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[87] = (QK_local[87] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1376)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[86] = (QK_local[86] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1377)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[87] = (QK_local[87] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1377)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[86] = (QK_local[86] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1378)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[87] = (QK_local[87] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1378)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[86] = (QK_local[86] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1379)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[87] = (QK_local[87] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1379)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[86] = (QK_local[86] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1380)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[87] = (QK_local[87] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1380)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[86] = (QK_local[86] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1381)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[87] = (QK_local[87] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1381)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[86] = (QK_local[86] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1382)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[87] = (QK_local[87] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1382)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[86] = (QK_local[86] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1383)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[87] = (QK_local[87] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1383)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[88] = (QK_local[88] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1408)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[89] = (QK_local[89] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1408)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[88] = (QK_local[88] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1409)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[89] = (QK_local[89] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1409)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[88] = (QK_local[88] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1410)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[89] = (QK_local[89] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1410)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[88] = (QK_local[88] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1411)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[89] = (QK_local[89] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1411)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[88] = (QK_local[88] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1412)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[89] = (QK_local[89] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1412)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[88] = (QK_local[88] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1413)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[89] = (QK_local[89] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1413)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[88] = (QK_local[88] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1414)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[89] = (QK_local[89] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1414)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[88] = (QK_local[88] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1415)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[89] = (QK_local[89] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1415)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[90] = (QK_local[90] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1440)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[91] = (QK_local[91] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1440)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[90] = (QK_local[90] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1441)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[91] = (QK_local[91] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1441)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[90] = (QK_local[90] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1442)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[91] = (QK_local[91] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1442)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[90] = (QK_local[90] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1443)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[91] = (QK_local[91] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1443)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[90] = (QK_local[90] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1444)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[91] = (QK_local[91] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1444)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[90] = (QK_local[90] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1445)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[91] = (QK_local[91] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1445)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[90] = (QK_local[90] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1446)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[91] = (QK_local[91] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1446)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[90] = (QK_local[90] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1447)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[91] = (QK_local[91] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1447)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[92] = (QK_local[92] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1472)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[93] = (QK_local[93] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1472)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[92] = (QK_local[92] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1473)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[93] = (QK_local[93] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1473)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[92] = (QK_local[92] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1474)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[93] = (QK_local[93] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1474)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[92] = (QK_local[92] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1475)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[93] = (QK_local[93] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1475)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[92] = (QK_local[92] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1476)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[93] = (QK_local[93] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1476)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[92] = (QK_local[92] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1477)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[93] = (QK_local[93] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1477)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[92] = (QK_local[92] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1478)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[93] = (QK_local[93] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1478)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[92] = (QK_local[92] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1479)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[93] = (QK_local[93] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1479)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[94] = (QK_local[94] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1504)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[95] = (QK_local[95] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1504)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[94] = (QK_local[94] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1505)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[95] = (QK_local[95] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1505)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[94] = (QK_local[94] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1506)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[95] = (QK_local[95] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1506)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[94] = (QK_local[94] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1507)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[95] = (QK_local[95] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1507)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[94] = (QK_local[94] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1508)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[95] = (QK_local[95] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1508)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[94] = (QK_local[94] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1509)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[95] = (QK_local[95] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1509)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[94] = (QK_local[94] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1510)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[95] = (QK_local[95] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1510)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[94] = (QK_local[94] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1511)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[95] = (QK_local[95] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1511)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[96] = (QK_local[96] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1536)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[97] = (QK_local[97] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1536)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[96] = (QK_local[96] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1537)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[97] = (QK_local[97] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1537)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[96] = (QK_local[96] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1538)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[97] = (QK_local[97] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1538)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[96] = (QK_local[96] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1539)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[97] = (QK_local[97] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1539)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[96] = (QK_local[96] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1540)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[97] = (QK_local[97] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1540)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[96] = (QK_local[96] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1541)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[97] = (QK_local[97] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1541)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[96] = (QK_local[96] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1542)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[97] = (QK_local[97] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1542)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[96] = (QK_local[96] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1543)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[97] = (QK_local[97] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1543)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[98] = (QK_local[98] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1568)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[99] = (QK_local[99] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1568)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[98] = (QK_local[98] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1569)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[99] = (QK_local[99] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1569)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[98] = (QK_local[98] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1570)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[99] = (QK_local[99] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1570)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[98] = (QK_local[98] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1571)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[99] = (QK_local[99] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1571)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[98] = (QK_local[98] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1572)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[99] = (QK_local[99] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1572)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[98] = (QK_local[98] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1573)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[99] = (QK_local[99] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1573)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[98] = (QK_local[98] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1574)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[99] = (QK_local[99] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1574)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[98] = (QK_local[98] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1575)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[99] = (QK_local[99] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1575)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[100] = (QK_local[100] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1600)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[101] = (QK_local[101] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1600)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[100] = (QK_local[100] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1601)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[101] = (QK_local[101] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1601)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[100] = (QK_local[100] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1602)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[101] = (QK_local[101] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1602)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[100] = (QK_local[100] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1603)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[101] = (QK_local[101] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1603)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[100] = (QK_local[100] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1604)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[101] = (QK_local[101] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1604)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[100] = (QK_local[100] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1605)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[101] = (QK_local[101] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1605)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[100] = (QK_local[100] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1606)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[101] = (QK_local[101] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1606)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[100] = (QK_local[100] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1607)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[101] = (QK_local[101] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1607)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[102] = (QK_local[102] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1632)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[103] = (QK_local[103] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1632)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[102] = (QK_local[102] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1633)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[103] = (QK_local[103] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1633)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[102] = (QK_local[102] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1634)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[103] = (QK_local[103] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1634)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[102] = (QK_local[102] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1635)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[103] = (QK_local[103] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1635)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[102] = (QK_local[102] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1636)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[103] = (QK_local[103] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1636)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[102] = (QK_local[102] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1637)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[103] = (QK_local[103] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1637)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[102] = (QK_local[102] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1638)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[103] = (QK_local[103] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1638)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[102] = (QK_local[102] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1639)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[103] = (QK_local[103] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1639)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[104] = (QK_local[104] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1664)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[105] = (QK_local[105] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1664)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[104] = (QK_local[104] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1665)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[105] = (QK_local[105] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1665)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[104] = (QK_local[104] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1666)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[105] = (QK_local[105] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1666)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[104] = (QK_local[104] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1667)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[105] = (QK_local[105] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1667)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[104] = (QK_local[104] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1668)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[105] = (QK_local[105] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1668)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[104] = (QK_local[104] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1669)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[105] = (QK_local[105] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1669)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[104] = (QK_local[104] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1670)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[105] = (QK_local[105] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1670)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[104] = (QK_local[104] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1671)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[105] = (QK_local[105] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1671)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[106] = (QK_local[106] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1696)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[107] = (QK_local[107] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1696)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[106] = (QK_local[106] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1697)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[107] = (QK_local[107] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1697)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[106] = (QK_local[106] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1698)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[107] = (QK_local[107] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1698)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[106] = (QK_local[106] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1699)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[107] = (QK_local[107] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1699)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[106] = (QK_local[106] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1700)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[107] = (QK_local[107] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1700)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[106] = (QK_local[106] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1701)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[107] = (QK_local[107] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1701)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[106] = (QK_local[106] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1702)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[107] = (QK_local[107] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1702)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[106] = (QK_local[106] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1703)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[107] = (QK_local[107] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1703)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[108] = (QK_local[108] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1728)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[109] = (QK_local[109] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1728)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[108] = (QK_local[108] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1729)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[109] = (QK_local[109] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1729)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[108] = (QK_local[108] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1730)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[109] = (QK_local[109] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1730)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[108] = (QK_local[108] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1731)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[109] = (QK_local[109] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1731)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[108] = (QK_local[108] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1732)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[109] = (QK_local[109] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1732)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[108] = (QK_local[108] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1733)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[109] = (QK_local[109] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1733)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[108] = (QK_local[108] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1734)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[109] = (QK_local[109] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1734)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[108] = (QK_local[108] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1735)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[109] = (QK_local[109] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1735)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[110] = (QK_local[110] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1760)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[111] = (QK_local[111] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1760)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[110] = (QK_local[110] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1761)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[111] = (QK_local[111] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1761)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[110] = (QK_local[110] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1762)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[111] = (QK_local[111] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1762)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[110] = (QK_local[110] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1763)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[111] = (QK_local[111] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1763)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[110] = (QK_local[110] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1764)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[111] = (QK_local[111] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1764)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[110] = (QK_local[110] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1765)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[111] = (QK_local[111] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1765)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[110] = (QK_local[110] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1766)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[111] = (QK_local[111] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1766)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[110] = (QK_local[110] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1767)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[111] = (QK_local[111] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1767)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[112] = (QK_local[112] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1792)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[113] = (QK_local[113] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1792)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[112] = (QK_local[112] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1793)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[113] = (QK_local[113] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1793)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[112] = (QK_local[112] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1794)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[113] = (QK_local[113] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1794)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[112] = (QK_local[112] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1795)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[113] = (QK_local[113] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1795)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[112] = (QK_local[112] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1796)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[113] = (QK_local[113] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1796)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[112] = (QK_local[112] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1797)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[113] = (QK_local[113] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1797)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[112] = (QK_local[112] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1798)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[113] = (QK_local[113] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1798)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[112] = (QK_local[112] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1799)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[113] = (QK_local[113] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1799)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[114] = (QK_local[114] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1824)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[115] = (QK_local[115] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1824)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[114] = (QK_local[114] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1825)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[115] = (QK_local[115] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1825)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[114] = (QK_local[114] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1826)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[115] = (QK_local[115] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1826)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[114] = (QK_local[114] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1827)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[115] = (QK_local[115] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1827)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[114] = (QK_local[114] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1828)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[115] = (QK_local[115] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1828)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[114] = (QK_local[114] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1829)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[115] = (QK_local[115] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1829)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[114] = (QK_local[114] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1830)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[115] = (QK_local[115] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1830)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[114] = (QK_local[114] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1831)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[115] = (QK_local[115] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1831)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[116] = (QK_local[116] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1856)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[117] = (QK_local[117] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1856)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[116] = (QK_local[116] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1857)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[117] = (QK_local[117] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1857)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[116] = (QK_local[116] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1858)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[117] = (QK_local[117] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1858)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[116] = (QK_local[116] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1859)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[117] = (QK_local[117] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1859)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[116] = (QK_local[116] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1860)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[117] = (QK_local[117] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1860)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[116] = (QK_local[116] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1861)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[117] = (QK_local[117] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1861)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[116] = (QK_local[116] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1862)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[117] = (QK_local[117] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1862)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[116] = (QK_local[116] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1863)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[117] = (QK_local[117] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1863)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[118] = (QK_local[118] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1888)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[119] = (QK_local[119] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1888)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[118] = (QK_local[118] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1889)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[119] = (QK_local[119] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1889)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[118] = (QK_local[118] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1890)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[119] = (QK_local[119] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1890)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[118] = (QK_local[118] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1891)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[119] = (QK_local[119] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1891)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[118] = (QK_local[118] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1892)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[119] = (QK_local[119] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1892)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[118] = (QK_local[118] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1893)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[119] = (QK_local[119] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1893)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[118] = (QK_local[118] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1894)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[119] = (QK_local[119] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1894)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[118] = (QK_local[118] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1895)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[119] = (QK_local[119] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1895)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[120] = (QK_local[120] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1920)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[121] = (QK_local[121] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1920)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[120] = (QK_local[120] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1921)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[121] = (QK_local[121] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1921)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[120] = (QK_local[120] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1922)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[121] = (QK_local[121] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1922)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[120] = (QK_local[120] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1923)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[121] = (QK_local[121] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1923)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[120] = (QK_local[120] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1924)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[121] = (QK_local[121] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1924)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[120] = (QK_local[120] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1925)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[121] = (QK_local[121] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1925)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[120] = (QK_local[120] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1926)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[121] = (QK_local[121] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1926)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[120] = (QK_local[120] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1927)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[121] = (QK_local[121] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1927)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[122] = (QK_local[122] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1952)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[123] = (QK_local[123] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1952)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[122] = (QK_local[122] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1953)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[123] = (QK_local[123] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1953)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[122] = (QK_local[122] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1954)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[123] = (QK_local[123] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1954)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[122] = (QK_local[122] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1955)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[123] = (QK_local[123] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1955)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[122] = (QK_local[122] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1956)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[123] = (QK_local[123] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1956)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[122] = (QK_local[122] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1957)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[123] = (QK_local[123] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1957)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[122] = (QK_local[122] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1958)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[123] = (QK_local[123] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1958)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[122] = (QK_local[122] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1959)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[123] = (QK_local[123] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1959)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[124] = (QK_local[124] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1984)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[125] = (QK_local[125] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1984)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[124] = (QK_local[124] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1985)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[125] = (QK_local[125] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1985)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[124] = (QK_local[124] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1986)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[125] = (QK_local[125] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1986)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[124] = (QK_local[124] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1987)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[125] = (QK_local[125] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1987)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[124] = (QK_local[124] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1988)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[125] = (QK_local[125] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1988)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[124] = (QK_local[124] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1989)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[125] = (QK_local[125] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1989)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[124] = (QK_local[124] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1990)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[125] = (QK_local[125] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1990)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[124] = (QK_local[124] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1991)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[125] = (QK_local[125] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 1991)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
      QK_local[126] = (QK_local[126] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2016)] * K_shared[(((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8))]) * __float2half_rn(8.838835e-02f)));
      QK_local[127] = (QK_local[127] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2016)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 32)]) * __float2half_rn(8.838835e-02f)));
      QK_local[126] = (QK_local[126] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2017)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 1)]) * __float2half_rn(8.838835e-02f)));
      QK_local[127] = (QK_local[127] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2017)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 33)]) * __float2half_rn(8.838835e-02f)));
      QK_local[126] = (QK_local[126] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2018)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 2)]) * __float2half_rn(8.838835e-02f)));
      QK_local[127] = (QK_local[127] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2018)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 34)]) * __float2half_rn(8.838835e-02f)));
      QK_local[126] = (QK_local[126] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2019)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 3)]) * __float2half_rn(8.838835e-02f)));
      QK_local[127] = (QK_local[127] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2019)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 35)]) * __float2half_rn(8.838835e-02f)));
      QK_local[126] = (QK_local[126] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2020)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 4)]) * __float2half_rn(8.838835e-02f)));
      QK_local[127] = (QK_local[127] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2020)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 36)]) * __float2half_rn(8.838835e-02f)));
      QK_local[126] = (QK_local[126] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2021)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 5)]) * __float2half_rn(8.838835e-02f)));
      QK_local[127] = (QK_local[127] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2021)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 37)]) * __float2half_rn(8.838835e-02f)));
      QK_local[126] = (QK_local[126] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2022)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 6)]) * __float2half_rn(8.838835e-02f)));
      QK_local[127] = (QK_local[127] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2022)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 38)]) * __float2half_rn(8.838835e-02f)));
      QK_local[126] = (QK_local[126] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2023)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 7)]) * __float2half_rn(8.838835e-02f)));
      QK_local[127] = (QK_local[127] + ((Q_shared[((((((int)threadIdx.x) >> 5) * 2048) + (k1_1 * 8)) + 2023)] * K_shared[((((((int)threadIdx.x) & 31) * 64) + (k1_1 * 8)) + 39)]) * __float2half_rn(8.838835e-02f)));
    }
  }
  QK[(((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2))] = QK_local[0];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 1)] = QK_local[1];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 512)] = QK_local[2];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 513)] = QK_local[3];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 1024)] = QK_local[4];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 1025)] = QK_local[5];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 1536)] = QK_local[6];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 1537)] = QK_local[7];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 2048)] = QK_local[8];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 2049)] = QK_local[9];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 2560)] = QK_local[10];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 2561)] = QK_local[11];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 3072)] = QK_local[12];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 3073)] = QK_local[13];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 3584)] = QK_local[14];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 3585)] = QK_local[15];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 4096)] = QK_local[16];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 4097)] = QK_local[17];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 4608)] = QK_local[18];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 4609)] = QK_local[19];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 5120)] = QK_local[20];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 5121)] = QK_local[21];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 5632)] = QK_local[22];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 5633)] = QK_local[23];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 6144)] = QK_local[24];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 6145)] = QK_local[25];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 6656)] = QK_local[26];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 6657)] = QK_local[27];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 7168)] = QK_local[28];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 7169)] = QK_local[29];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 7680)] = QK_local[30];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 7681)] = QK_local[31];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 8192)] = QK_local[32];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 8193)] = QK_local[33];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 8704)] = QK_local[34];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 8705)] = QK_local[35];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 9216)] = QK_local[36];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 9217)] = QK_local[37];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 9728)] = QK_local[38];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 9729)] = QK_local[39];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 10240)] = QK_local[40];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 10241)] = QK_local[41];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 10752)] = QK_local[42];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 10753)] = QK_local[43];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 11264)] = QK_local[44];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 11265)] = QK_local[45];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 11776)] = QK_local[46];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 11777)] = QK_local[47];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 12288)] = QK_local[48];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 12289)] = QK_local[49];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 12800)] = QK_local[50];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 12801)] = QK_local[51];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 13312)] = QK_local[52];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 13313)] = QK_local[53];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 13824)] = QK_local[54];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 13825)] = QK_local[55];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 14336)] = QK_local[56];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 14337)] = QK_local[57];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 14848)] = QK_local[58];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 14849)] = QK_local[59];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 15360)] = QK_local[60];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 15361)] = QK_local[61];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 15872)] = QK_local[62];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 15873)] = QK_local[63];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 16384)] = QK_local[64];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 16385)] = QK_local[65];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 16896)] = QK_local[66];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 16897)] = QK_local[67];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 17408)] = QK_local[68];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 17409)] = QK_local[69];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 17920)] = QK_local[70];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 17921)] = QK_local[71];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 18432)] = QK_local[72];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 18433)] = QK_local[73];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 18944)] = QK_local[74];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 18945)] = QK_local[75];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 19456)] = QK_local[76];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 19457)] = QK_local[77];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 19968)] = QK_local[78];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 19969)] = QK_local[79];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 20480)] = QK_local[80];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 20481)] = QK_local[81];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 20992)] = QK_local[82];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 20993)] = QK_local[83];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 21504)] = QK_local[84];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 21505)] = QK_local[85];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 22016)] = QK_local[86];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 22017)] = QK_local[87];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 22528)] = QK_local[88];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 22529)] = QK_local[89];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 23040)] = QK_local[90];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 23041)] = QK_local[91];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 23552)] = QK_local[92];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 23553)] = QK_local[93];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 24064)] = QK_local[94];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 24065)] = QK_local[95];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 24576)] = QK_local[96];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 24577)] = QK_local[97];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 25088)] = QK_local[98];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 25089)] = QK_local[99];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 25600)] = QK_local[100];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 25601)] = QK_local[101];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 26112)] = QK_local[102];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 26113)] = QK_local[103];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 26624)] = QK_local[104];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 26625)] = QK_local[105];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 27136)] = QK_local[106];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 27137)] = QK_local[107];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 27648)] = QK_local[108];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 27649)] = QK_local[109];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 28160)] = QK_local[110];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 28161)] = QK_local[111];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 28672)] = QK_local[112];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 28673)] = QK_local[113];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 29184)] = QK_local[114];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 29185)] = QK_local[115];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 29696)] = QK_local[116];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 29697)] = QK_local[117];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 30208)] = QK_local[118];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 30209)] = QK_local[119];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 30720)] = QK_local[120];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 30721)] = QK_local[121];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 31232)] = QK_local[122];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 31233)] = QK_local[123];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 31744)] = QK_local[124];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 31745)] = QK_local[125];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 32256)] = QK_local[126];
  QK[((((((((int)blockIdx.x) >> 3) * 131072) + ((((int)threadIdx.x) >> 5) * 32768)) + ((((int)blockIdx.x) & 7) * 64)) + ((((int)threadIdx.x) & 31) * 2)) + 32257)] = QK_local[127];
}

