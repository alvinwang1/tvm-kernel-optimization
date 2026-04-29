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
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax);
extern "C" __global__ void __launch_bounds__(128) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum);
extern "C" __global__ void __launch_bounds__(64) main_kernel_2(half* __restrict__ Out, half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum, half* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax) {
  RowMax[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = __float2half_rn(-6.550400e+04f);
  for (int m_ax = 0; m_ax < 2048; ++m_ax) {
    RowMax[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 2048)) + m_ax)]);
  }
}

extern "C" __global__ void __launch_bounds__(128) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum) {
  RowSum[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = __float2half_rn(0.000000e+00f);
  for (int s_ax = 0; s_ax < 2048; ++s_ax) {
    RowSum[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 262144) + (((int)threadIdx.x) * 2048)) + s_ax)] - RowMax[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))])));
  }
}

extern "C" __global__ void __launch_bounds__(64) main_kernel_2(half* __restrict__ Out, half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum, half* __restrict__ V) {
  half Out_local[256];
  __shared__ half Attn_shared[2048];
  __shared__ half V_shared[4096];
  for (int d_3_init = 0; d_3_init < 8; ++d_3_init) {
    Out_local[d_3_init] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 16)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 32)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 48)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 64)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 80)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 96)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 112)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 128)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 144)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 160)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 176)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 192)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 208)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 224)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 240)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 8)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 24)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 40)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 56)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 72)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 88)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 104)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 120)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 136)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 152)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 168)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 184)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 200)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 216)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 232)] = __float2half_rn(0.000000e+00f);
    Out_local[(d_3_init + 248)] = __float2half_rn(0.000000e+00f);
  }
  Attn_shared[((int)threadIdx.x)] = (hexp((QK[(((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3))] - RowMax[((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2))])) / RowSum[((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2))]);
  Attn_shared[(((int)threadIdx.x) + 64)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 32768)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 16)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 16)]);
  Attn_shared[(((int)threadIdx.x) + 128)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 65536)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 32)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 32)]);
  Attn_shared[(((int)threadIdx.x) + 192)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 98304)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 48)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 48)]);
  Attn_shared[(((int)threadIdx.x) + 256)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4194304)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2048)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2048)]);
  Attn_shared[(((int)threadIdx.x) + 320)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4227072)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2064)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2064)]);
  Attn_shared[(((int)threadIdx.x) + 384)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4259840)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2080)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2080)]);
  Attn_shared[(((int)threadIdx.x) + 448)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4292608)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2096)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2096)]);
  *(half4*)(V_shared + (((int)threadIdx.x) * 4)) = *(half4*)(V + (((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 256)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 256));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 512)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 262144));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 768)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 262400));
__asm__ __volatile__("cp.async.commit_group;");

  Attn_shared[(((int)threadIdx.x) + 512)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4)] - RowMax[((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2))])) / RowSum[((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2))]);
  Attn_shared[(((int)threadIdx.x) + 576)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 32772)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 16)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 16)]);
  Attn_shared[(((int)threadIdx.x) + 640)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 65540)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 32)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 32)]);
  Attn_shared[(((int)threadIdx.x) + 704)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 98308)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 48)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 48)]);
  Attn_shared[(((int)threadIdx.x) + 768)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4194308)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2048)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2048)]);
  Attn_shared[(((int)threadIdx.x) + 832)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4227076)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2064)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2064)]);
  Attn_shared[(((int)threadIdx.x) + 896)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4259844)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2080)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2080)]);
  Attn_shared[(((int)threadIdx.x) + 960)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4292612)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2096)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2096)]);
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 768));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 262656));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 262912));
__asm__ __volatile__("cp.async.commit_group;");

  Attn_shared[(((int)threadIdx.x) + 1024)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 8)] - RowMax[((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2))])) / RowSum[((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2))]);
  Attn_shared[(((int)threadIdx.x) + 1088)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 32776)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 16)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 16)]);
  Attn_shared[(((int)threadIdx.x) + 1152)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 65544)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 32)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 32)]);
  Attn_shared[(((int)threadIdx.x) + 1216)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 98312)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 48)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 48)]);
  Attn_shared[(((int)threadIdx.x) + 1280)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4194312)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2048)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2048)]);
  Attn_shared[(((int)threadIdx.x) + 1344)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4227080)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2064)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2064)]);
  Attn_shared[(((int)threadIdx.x) + 1408)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4259848)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2080)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2080)]);
  Attn_shared[(((int)threadIdx.x) + 1472)] = (hexp((QK[((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)threadIdx.x) & 3)) + 4292616)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2096)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2096)]);
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 1024));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 2304)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 1280));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 263168));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 2816)) = *(half4*)(V + ((((((int)blockIdx.x) >> 5) * 524288) + (((int)threadIdx.x) * 4)) + 263424));
__asm__ __volatile__("cp.async.commit_group;");

  for (int k2_0_fused = 0; k2_0_fused < 509; ++k2_0_fused) {
    __syncthreads();
    Attn_shared[((((k2_0_fused + 3) & 3) * 512) + ((int)threadIdx.x))] = (hexp((QK[(((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k2_0_fused * 4)) + (((int)threadIdx.x) & 3)) + 12)] - RowMax[((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2))])) / RowSum[((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2))]);
    Attn_shared[(((((k2_0_fused + 3) & 3) * 512) + ((int)threadIdx.x)) + 64)] = (hexp((QK[(((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k2_0_fused * 4)) + (((int)threadIdx.x) & 3)) + 32780)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 16)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 16)]);
    Attn_shared[(((((k2_0_fused + 3) & 3) * 512) + ((int)threadIdx.x)) + 128)] = (hexp((QK[(((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k2_0_fused * 4)) + (((int)threadIdx.x) & 3)) + 65548)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 32)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 32)]);
    Attn_shared[(((((k2_0_fused + 3) & 3) * 512) + ((int)threadIdx.x)) + 192)] = (hexp((QK[(((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k2_0_fused * 4)) + (((int)threadIdx.x) & 3)) + 98316)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 48)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 48)]);
    Attn_shared[(((((k2_0_fused + 3) & 3) * 512) + ((int)threadIdx.x)) + 256)] = (hexp((QK[(((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k2_0_fused * 4)) + (((int)threadIdx.x) & 3)) + 4194316)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2048)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2048)]);
    Attn_shared[(((((k2_0_fused + 3) & 3) * 512) + ((int)threadIdx.x)) + 320)] = (hexp((QK[(((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k2_0_fused * 4)) + (((int)threadIdx.x) & 3)) + 4227084)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2064)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2064)]);
    Attn_shared[(((((k2_0_fused + 3) & 3) * 512) + ((int)threadIdx.x)) + 384)] = (hexp((QK[(((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k2_0_fused * 4)) + (((int)threadIdx.x) & 3)) + 4259852)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2080)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2080)]);
    Attn_shared[(((((k2_0_fused + 3) & 3) * 512) + ((int)threadIdx.x)) + 448)] = (hexp((QK[(((((((((int)blockIdx.x) >> 5) * 8388608) + ((((int)blockIdx.x) & 31) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (k2_0_fused * 4)) + (((int)threadIdx.x) & 3)) + 4292620)] - RowMax[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2096)])) / RowSum[(((((((int)blockIdx.x) >> 5) * 4096) + ((((int)blockIdx.x) & 31) * 64)) + (((int)threadIdx.x) >> 2)) + 2096)]);
    *(half4*)(V_shared + ((((k2_0_fused + 3) & 3) * 1024) + (((int)threadIdx.x) * 4))) = *(half4*)(V + (((((((int)blockIdx.x) >> 5) * 524288) + (k2_0_fused * 512)) + (((int)threadIdx.x) * 4)) + 1536));
    *(half4*)(V_shared + (((((k2_0_fused + 3) & 3) * 1024) + (((int)threadIdx.x) * 4)) + 256)) = *(half4*)(V + (((((((int)blockIdx.x) >> 5) * 524288) + (k2_0_fused * 512)) + (((int)threadIdx.x) * 4)) + 1792));
    *(half4*)(V_shared + (((((k2_0_fused + 3) & 3) * 1024) + (((int)threadIdx.x) * 4)) + 512)) = *(half4*)(V + (((((((int)blockIdx.x) >> 5) * 524288) + (k2_0_fused * 512)) + (((int)threadIdx.x) * 4)) + 263680));
    *(half4*)(V_shared + (((((k2_0_fused + 3) & 3) * 1024) + (((int)threadIdx.x) * 4)) + 768)) = *(half4*)(V + (((((((int)blockIdx.x) >> 5) * 524288) + (k2_0_fused * 512)) + (((int)threadIdx.x) * 4)) + 263936));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

    __syncthreads();
    for (int k2_1 = 0; k2_1 < 2; ++k2_1) {
      for (int d_3 = 0; d_3 < 8; ++d_3) {
        Out_local[d_3] = (Out_local[d_3] + (Attn_shared[((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2))] * V_shared[(((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3)]));
        Out_local[(d_3 + 16)] = (Out_local[(d_3 + 16)] + (Attn_shared[((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2))] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 64)]));
        Out_local[(d_3 + 32)] = (Out_local[(d_3 + 32)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 32)] * V_shared[(((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3)]));
        Out_local[(d_3 + 48)] = (Out_local[(d_3 + 48)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 32)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 64)]));
        Out_local[(d_3 + 64)] = (Out_local[(d_3 + 64)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 64)] * V_shared[(((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3)]));
        Out_local[(d_3 + 80)] = (Out_local[(d_3 + 80)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 64)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 64)]));
        Out_local[(d_3 + 96)] = (Out_local[(d_3 + 96)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 96)] * V_shared[(((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3)]));
        Out_local[(d_3 + 112)] = (Out_local[(d_3 + 112)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 96)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 64)]));
        Out_local[(d_3 + 128)] = (Out_local[(d_3 + 128)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 128)] * V_shared[(((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3)]));
        Out_local[(d_3 + 144)] = (Out_local[(d_3 + 144)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 128)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 64)]));
        Out_local[(d_3 + 160)] = (Out_local[(d_3 + 160)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 160)] * V_shared[(((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3)]));
        Out_local[(d_3 + 176)] = (Out_local[(d_3 + 176)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 160)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 64)]));
        Out_local[(d_3 + 192)] = (Out_local[(d_3 + 192)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 192)] * V_shared[(((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3)]));
        Out_local[(d_3 + 208)] = (Out_local[(d_3 + 208)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 192)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 64)]));
        Out_local[(d_3 + 224)] = (Out_local[(d_3 + 224)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 224)] * V_shared[(((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3)]));
        Out_local[(d_3 + 240)] = (Out_local[(d_3 + 240)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 224)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 64)]));
        Out_local[(d_3 + 8)] = (Out_local[(d_3 + 8)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 256)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 512)]));
        Out_local[(d_3 + 24)] = (Out_local[(d_3 + 24)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 256)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 576)]));
        Out_local[(d_3 + 40)] = (Out_local[(d_3 + 40)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 288)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 512)]));
        Out_local[(d_3 + 56)] = (Out_local[(d_3 + 56)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 288)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 576)]));
        Out_local[(d_3 + 72)] = (Out_local[(d_3 + 72)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 320)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 512)]));
        Out_local[(d_3 + 88)] = (Out_local[(d_3 + 88)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 320)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 576)]));
        Out_local[(d_3 + 104)] = (Out_local[(d_3 + 104)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 352)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 512)]));
        Out_local[(d_3 + 120)] = (Out_local[(d_3 + 120)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 352)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 576)]));
        Out_local[(d_3 + 136)] = (Out_local[(d_3 + 136)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 384)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 512)]));
        Out_local[(d_3 + 152)] = (Out_local[(d_3 + 152)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 384)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 576)]));
        Out_local[(d_3 + 168)] = (Out_local[(d_3 + 168)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 416)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 512)]));
        Out_local[(d_3 + 184)] = (Out_local[(d_3 + 184)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 416)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 576)]));
        Out_local[(d_3 + 200)] = (Out_local[(d_3 + 200)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 448)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 512)]));
        Out_local[(d_3 + 216)] = (Out_local[(d_3 + 216)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 448)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 576)]));
        Out_local[(d_3 + 232)] = (Out_local[(d_3 + 232)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 480)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 512)]));
        Out_local[(d_3 + 248)] = (Out_local[(d_3 + 248)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 480)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 576)]));
        Out_local[d_3] = (Out_local[d_3] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 1)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 128)]));
        Out_local[(d_3 + 16)] = (Out_local[(d_3 + 16)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 1)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 192)]));
        Out_local[(d_3 + 32)] = (Out_local[(d_3 + 32)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 33)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 128)]));
        Out_local[(d_3 + 48)] = (Out_local[(d_3 + 48)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 33)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 192)]));
        Out_local[(d_3 + 64)] = (Out_local[(d_3 + 64)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 65)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 128)]));
        Out_local[(d_3 + 80)] = (Out_local[(d_3 + 80)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 65)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 192)]));
        Out_local[(d_3 + 96)] = (Out_local[(d_3 + 96)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 97)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 128)]));
        Out_local[(d_3 + 112)] = (Out_local[(d_3 + 112)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 97)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 192)]));
        Out_local[(d_3 + 128)] = (Out_local[(d_3 + 128)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 129)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 128)]));
        Out_local[(d_3 + 144)] = (Out_local[(d_3 + 144)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 129)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 192)]));
        Out_local[(d_3 + 160)] = (Out_local[(d_3 + 160)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 161)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 128)]));
        Out_local[(d_3 + 176)] = (Out_local[(d_3 + 176)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 161)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 192)]));
        Out_local[(d_3 + 192)] = (Out_local[(d_3 + 192)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 193)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 128)]));
        Out_local[(d_3 + 208)] = (Out_local[(d_3 + 208)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 193)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 192)]));
        Out_local[(d_3 + 224)] = (Out_local[(d_3 + 224)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 225)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 128)]));
        Out_local[(d_3 + 240)] = (Out_local[(d_3 + 240)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 225)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 192)]));
        Out_local[(d_3 + 8)] = (Out_local[(d_3 + 8)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 257)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 640)]));
        Out_local[(d_3 + 24)] = (Out_local[(d_3 + 24)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 257)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 704)]));
        Out_local[(d_3 + 40)] = (Out_local[(d_3 + 40)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 289)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 640)]));
        Out_local[(d_3 + 56)] = (Out_local[(d_3 + 56)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 289)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 704)]));
        Out_local[(d_3 + 72)] = (Out_local[(d_3 + 72)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 321)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 640)]));
        Out_local[(d_3 + 88)] = (Out_local[(d_3 + 88)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 321)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 704)]));
        Out_local[(d_3 + 104)] = (Out_local[(d_3 + 104)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 353)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 640)]));
        Out_local[(d_3 + 120)] = (Out_local[(d_3 + 120)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 353)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 704)]));
        Out_local[(d_3 + 136)] = (Out_local[(d_3 + 136)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 385)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 640)]));
        Out_local[(d_3 + 152)] = (Out_local[(d_3 + 152)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 385)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 704)]));
        Out_local[(d_3 + 168)] = (Out_local[(d_3 + 168)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 417)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 640)]));
        Out_local[(d_3 + 184)] = (Out_local[(d_3 + 184)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 417)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 704)]));
        Out_local[(d_3 + 200)] = (Out_local[(d_3 + 200)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 449)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 640)]));
        Out_local[(d_3 + 216)] = (Out_local[(d_3 + 216)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 449)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 704)]));
        Out_local[(d_3 + 232)] = (Out_local[(d_3 + 232)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 481)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 640)]));
        Out_local[(d_3 + 248)] = (Out_local[(d_3 + 248)] + (Attn_shared[(((((k2_0_fused & 3) * 512) + ((((int)threadIdx.x) >> 3) * 4)) + (k2_1 * 2)) + 481)] * V_shared[((((((k2_0_fused & 3) * 1024) + (k2_1 * 256)) + ((((int)threadIdx.x) & 7) * 8)) + d_3) + 704)]));
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  for (int k2_1_1 = 0; k2_1_1 < 2; ++k2_1_1) {
    for (int d_3_1 = 0; d_3_1 < 8; ++d_3_1) {
      Out_local[d_3_1] = (Out_local[d_3_1] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 512)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1024)]));
      Out_local[(d_3_1 + 16)] = (Out_local[(d_3_1 + 16)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 512)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1088)]));
      Out_local[(d_3_1 + 32)] = (Out_local[(d_3_1 + 32)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 544)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1024)]));
      Out_local[(d_3_1 + 48)] = (Out_local[(d_3_1 + 48)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 544)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1088)]));
      Out_local[(d_3_1 + 64)] = (Out_local[(d_3_1 + 64)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 576)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1024)]));
      Out_local[(d_3_1 + 80)] = (Out_local[(d_3_1 + 80)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 576)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1088)]));
      Out_local[(d_3_1 + 96)] = (Out_local[(d_3_1 + 96)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 608)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1024)]));
      Out_local[(d_3_1 + 112)] = (Out_local[(d_3_1 + 112)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 608)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1088)]));
      Out_local[(d_3_1 + 128)] = (Out_local[(d_3_1 + 128)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 640)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1024)]));
      Out_local[(d_3_1 + 144)] = (Out_local[(d_3_1 + 144)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 640)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1088)]));
      Out_local[(d_3_1 + 160)] = (Out_local[(d_3_1 + 160)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 672)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1024)]));
      Out_local[(d_3_1 + 176)] = (Out_local[(d_3_1 + 176)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 672)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1088)]));
      Out_local[(d_3_1 + 192)] = (Out_local[(d_3_1 + 192)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 704)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1024)]));
      Out_local[(d_3_1 + 208)] = (Out_local[(d_3_1 + 208)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 704)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1088)]));
      Out_local[(d_3_1 + 224)] = (Out_local[(d_3_1 + 224)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 736)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1024)]));
      Out_local[(d_3_1 + 240)] = (Out_local[(d_3_1 + 240)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 736)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1088)]));
      Out_local[(d_3_1 + 8)] = (Out_local[(d_3_1 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 768)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1536)]));
      Out_local[(d_3_1 + 24)] = (Out_local[(d_3_1 + 24)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 768)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1600)]));
      Out_local[(d_3_1 + 40)] = (Out_local[(d_3_1 + 40)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 800)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1536)]));
      Out_local[(d_3_1 + 56)] = (Out_local[(d_3_1 + 56)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 800)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1600)]));
      Out_local[(d_3_1 + 72)] = (Out_local[(d_3_1 + 72)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 832)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1536)]));
      Out_local[(d_3_1 + 88)] = (Out_local[(d_3_1 + 88)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 832)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1600)]));
      Out_local[(d_3_1 + 104)] = (Out_local[(d_3_1 + 104)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 864)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1536)]));
      Out_local[(d_3_1 + 120)] = (Out_local[(d_3_1 + 120)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 864)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1600)]));
      Out_local[(d_3_1 + 136)] = (Out_local[(d_3_1 + 136)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 896)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1536)]));
      Out_local[(d_3_1 + 152)] = (Out_local[(d_3_1 + 152)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 896)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1600)]));
      Out_local[(d_3_1 + 168)] = (Out_local[(d_3_1 + 168)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 928)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1536)]));
      Out_local[(d_3_1 + 184)] = (Out_local[(d_3_1 + 184)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 928)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1600)]));
      Out_local[(d_3_1 + 200)] = (Out_local[(d_3_1 + 200)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 960)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1536)]));
      Out_local[(d_3_1 + 216)] = (Out_local[(d_3_1 + 216)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 960)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1600)]));
      Out_local[(d_3_1 + 232)] = (Out_local[(d_3_1 + 232)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 992)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1536)]));
      Out_local[(d_3_1 + 248)] = (Out_local[(d_3_1 + 248)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 992)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1600)]));
      Out_local[d_3_1] = (Out_local[d_3_1] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 513)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1152)]));
      Out_local[(d_3_1 + 16)] = (Out_local[(d_3_1 + 16)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 513)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1216)]));
      Out_local[(d_3_1 + 32)] = (Out_local[(d_3_1 + 32)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 545)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1152)]));
      Out_local[(d_3_1 + 48)] = (Out_local[(d_3_1 + 48)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 545)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1216)]));
      Out_local[(d_3_1 + 64)] = (Out_local[(d_3_1 + 64)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 577)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1152)]));
      Out_local[(d_3_1 + 80)] = (Out_local[(d_3_1 + 80)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 577)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1216)]));
      Out_local[(d_3_1 + 96)] = (Out_local[(d_3_1 + 96)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 609)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1152)]));
      Out_local[(d_3_1 + 112)] = (Out_local[(d_3_1 + 112)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 609)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1216)]));
      Out_local[(d_3_1 + 128)] = (Out_local[(d_3_1 + 128)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 641)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1152)]));
      Out_local[(d_3_1 + 144)] = (Out_local[(d_3_1 + 144)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 641)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1216)]));
      Out_local[(d_3_1 + 160)] = (Out_local[(d_3_1 + 160)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 673)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1152)]));
      Out_local[(d_3_1 + 176)] = (Out_local[(d_3_1 + 176)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 673)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1216)]));
      Out_local[(d_3_1 + 192)] = (Out_local[(d_3_1 + 192)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 705)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1152)]));
      Out_local[(d_3_1 + 208)] = (Out_local[(d_3_1 + 208)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 705)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1216)]));
      Out_local[(d_3_1 + 224)] = (Out_local[(d_3_1 + 224)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 737)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1152)]));
      Out_local[(d_3_1 + 240)] = (Out_local[(d_3_1 + 240)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 737)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1216)]));
      Out_local[(d_3_1 + 8)] = (Out_local[(d_3_1 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 769)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1664)]));
      Out_local[(d_3_1 + 24)] = (Out_local[(d_3_1 + 24)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 769)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1728)]));
      Out_local[(d_3_1 + 40)] = (Out_local[(d_3_1 + 40)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 801)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1664)]));
      Out_local[(d_3_1 + 56)] = (Out_local[(d_3_1 + 56)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 801)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1728)]));
      Out_local[(d_3_1 + 72)] = (Out_local[(d_3_1 + 72)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 833)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1664)]));
      Out_local[(d_3_1 + 88)] = (Out_local[(d_3_1 + 88)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 833)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1728)]));
      Out_local[(d_3_1 + 104)] = (Out_local[(d_3_1 + 104)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 865)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1664)]));
      Out_local[(d_3_1 + 120)] = (Out_local[(d_3_1 + 120)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 865)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1728)]));
      Out_local[(d_3_1 + 136)] = (Out_local[(d_3_1 + 136)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 897)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1664)]));
      Out_local[(d_3_1 + 152)] = (Out_local[(d_3_1 + 152)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 897)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1728)]));
      Out_local[(d_3_1 + 168)] = (Out_local[(d_3_1 + 168)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 929)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1664)]));
      Out_local[(d_3_1 + 184)] = (Out_local[(d_3_1 + 184)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 929)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1728)]));
      Out_local[(d_3_1 + 200)] = (Out_local[(d_3_1 + 200)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 961)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1664)]));
      Out_local[(d_3_1 + 216)] = (Out_local[(d_3_1 + 216)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 961)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1728)]));
      Out_local[(d_3_1 + 232)] = (Out_local[(d_3_1 + 232)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 993)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1664)]));
      Out_local[(d_3_1 + 248)] = (Out_local[(d_3_1 + 248)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_1 * 2)) + 993)] * V_shared[((((k2_1_1 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_1) + 1728)]));
    }
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int k2_1_2 = 0; k2_1_2 < 2; ++k2_1_2) {
    for (int d_3_2 = 0; d_3_2 < 8; ++d_3_2) {
      Out_local[d_3_2] = (Out_local[d_3_2] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1024)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2048)]));
      Out_local[(d_3_2 + 16)] = (Out_local[(d_3_2 + 16)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1024)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2112)]));
      Out_local[(d_3_2 + 32)] = (Out_local[(d_3_2 + 32)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1056)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2048)]));
      Out_local[(d_3_2 + 48)] = (Out_local[(d_3_2 + 48)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1056)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2112)]));
      Out_local[(d_3_2 + 64)] = (Out_local[(d_3_2 + 64)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1088)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2048)]));
      Out_local[(d_3_2 + 80)] = (Out_local[(d_3_2 + 80)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1088)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2112)]));
      Out_local[(d_3_2 + 96)] = (Out_local[(d_3_2 + 96)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1120)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2048)]));
      Out_local[(d_3_2 + 112)] = (Out_local[(d_3_2 + 112)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1120)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2112)]));
      Out_local[(d_3_2 + 128)] = (Out_local[(d_3_2 + 128)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1152)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2048)]));
      Out_local[(d_3_2 + 144)] = (Out_local[(d_3_2 + 144)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1152)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2112)]));
      Out_local[(d_3_2 + 160)] = (Out_local[(d_3_2 + 160)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1184)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2048)]));
      Out_local[(d_3_2 + 176)] = (Out_local[(d_3_2 + 176)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1184)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2112)]));
      Out_local[(d_3_2 + 192)] = (Out_local[(d_3_2 + 192)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1216)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2048)]));
      Out_local[(d_3_2 + 208)] = (Out_local[(d_3_2 + 208)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1216)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2112)]));
      Out_local[(d_3_2 + 224)] = (Out_local[(d_3_2 + 224)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1248)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2048)]));
      Out_local[(d_3_2 + 240)] = (Out_local[(d_3_2 + 240)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1248)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2112)]));
      Out_local[(d_3_2 + 8)] = (Out_local[(d_3_2 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1280)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2560)]));
      Out_local[(d_3_2 + 24)] = (Out_local[(d_3_2 + 24)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1280)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2624)]));
      Out_local[(d_3_2 + 40)] = (Out_local[(d_3_2 + 40)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1312)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2560)]));
      Out_local[(d_3_2 + 56)] = (Out_local[(d_3_2 + 56)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1312)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2624)]));
      Out_local[(d_3_2 + 72)] = (Out_local[(d_3_2 + 72)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1344)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2560)]));
      Out_local[(d_3_2 + 88)] = (Out_local[(d_3_2 + 88)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1344)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2624)]));
      Out_local[(d_3_2 + 104)] = (Out_local[(d_3_2 + 104)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1376)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2560)]));
      Out_local[(d_3_2 + 120)] = (Out_local[(d_3_2 + 120)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1376)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2624)]));
      Out_local[(d_3_2 + 136)] = (Out_local[(d_3_2 + 136)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1408)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2560)]));
      Out_local[(d_3_2 + 152)] = (Out_local[(d_3_2 + 152)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1408)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2624)]));
      Out_local[(d_3_2 + 168)] = (Out_local[(d_3_2 + 168)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1440)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2560)]));
      Out_local[(d_3_2 + 184)] = (Out_local[(d_3_2 + 184)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1440)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2624)]));
      Out_local[(d_3_2 + 200)] = (Out_local[(d_3_2 + 200)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1472)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2560)]));
      Out_local[(d_3_2 + 216)] = (Out_local[(d_3_2 + 216)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1472)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2624)]));
      Out_local[(d_3_2 + 232)] = (Out_local[(d_3_2 + 232)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1504)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2560)]));
      Out_local[(d_3_2 + 248)] = (Out_local[(d_3_2 + 248)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1504)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2624)]));
      Out_local[d_3_2] = (Out_local[d_3_2] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1025)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2176)]));
      Out_local[(d_3_2 + 16)] = (Out_local[(d_3_2 + 16)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1025)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2240)]));
      Out_local[(d_3_2 + 32)] = (Out_local[(d_3_2 + 32)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1057)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2176)]));
      Out_local[(d_3_2 + 48)] = (Out_local[(d_3_2 + 48)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1057)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2240)]));
      Out_local[(d_3_2 + 64)] = (Out_local[(d_3_2 + 64)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1089)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2176)]));
      Out_local[(d_3_2 + 80)] = (Out_local[(d_3_2 + 80)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1089)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2240)]));
      Out_local[(d_3_2 + 96)] = (Out_local[(d_3_2 + 96)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1121)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2176)]));
      Out_local[(d_3_2 + 112)] = (Out_local[(d_3_2 + 112)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1121)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2240)]));
      Out_local[(d_3_2 + 128)] = (Out_local[(d_3_2 + 128)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1153)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2176)]));
      Out_local[(d_3_2 + 144)] = (Out_local[(d_3_2 + 144)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1153)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2240)]));
      Out_local[(d_3_2 + 160)] = (Out_local[(d_3_2 + 160)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1185)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2176)]));
      Out_local[(d_3_2 + 176)] = (Out_local[(d_3_2 + 176)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1185)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2240)]));
      Out_local[(d_3_2 + 192)] = (Out_local[(d_3_2 + 192)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1217)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2176)]));
      Out_local[(d_3_2 + 208)] = (Out_local[(d_3_2 + 208)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1217)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2240)]));
      Out_local[(d_3_2 + 224)] = (Out_local[(d_3_2 + 224)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1249)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2176)]));
      Out_local[(d_3_2 + 240)] = (Out_local[(d_3_2 + 240)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1249)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2240)]));
      Out_local[(d_3_2 + 8)] = (Out_local[(d_3_2 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1281)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2688)]));
      Out_local[(d_3_2 + 24)] = (Out_local[(d_3_2 + 24)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1281)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2752)]));
      Out_local[(d_3_2 + 40)] = (Out_local[(d_3_2 + 40)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1313)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2688)]));
      Out_local[(d_3_2 + 56)] = (Out_local[(d_3_2 + 56)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1313)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2752)]));
      Out_local[(d_3_2 + 72)] = (Out_local[(d_3_2 + 72)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1345)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2688)]));
      Out_local[(d_3_2 + 88)] = (Out_local[(d_3_2 + 88)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1345)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2752)]));
      Out_local[(d_3_2 + 104)] = (Out_local[(d_3_2 + 104)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1377)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2688)]));
      Out_local[(d_3_2 + 120)] = (Out_local[(d_3_2 + 120)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1377)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2752)]));
      Out_local[(d_3_2 + 136)] = (Out_local[(d_3_2 + 136)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1409)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2688)]));
      Out_local[(d_3_2 + 152)] = (Out_local[(d_3_2 + 152)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1409)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2752)]));
      Out_local[(d_3_2 + 168)] = (Out_local[(d_3_2 + 168)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1441)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2688)]));
      Out_local[(d_3_2 + 184)] = (Out_local[(d_3_2 + 184)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1441)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2752)]));
      Out_local[(d_3_2 + 200)] = (Out_local[(d_3_2 + 200)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1473)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2688)]));
      Out_local[(d_3_2 + 216)] = (Out_local[(d_3_2 + 216)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1473)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2752)]));
      Out_local[(d_3_2 + 232)] = (Out_local[(d_3_2 + 232)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1505)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2688)]));
      Out_local[(d_3_2 + 248)] = (Out_local[(d_3_2 + 248)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_2 * 2)) + 1505)] * V_shared[((((k2_1_2 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_2) + 2752)]));
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int k2_1_3 = 0; k2_1_3 < 2; ++k2_1_3) {
    for (int d_3_3 = 0; d_3_3 < 8; ++d_3_3) {
      Out_local[d_3_3] = (Out_local[d_3_3] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1536)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3072)]));
      Out_local[(d_3_3 + 16)] = (Out_local[(d_3_3 + 16)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1536)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3136)]));
      Out_local[(d_3_3 + 32)] = (Out_local[(d_3_3 + 32)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1568)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3072)]));
      Out_local[(d_3_3 + 48)] = (Out_local[(d_3_3 + 48)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1568)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3136)]));
      Out_local[(d_3_3 + 64)] = (Out_local[(d_3_3 + 64)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1600)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3072)]));
      Out_local[(d_3_3 + 80)] = (Out_local[(d_3_3 + 80)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1600)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3136)]));
      Out_local[(d_3_3 + 96)] = (Out_local[(d_3_3 + 96)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1632)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3072)]));
      Out_local[(d_3_3 + 112)] = (Out_local[(d_3_3 + 112)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1632)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3136)]));
      Out_local[(d_3_3 + 128)] = (Out_local[(d_3_3 + 128)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1664)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3072)]));
      Out_local[(d_3_3 + 144)] = (Out_local[(d_3_3 + 144)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1664)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3136)]));
      Out_local[(d_3_3 + 160)] = (Out_local[(d_3_3 + 160)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1696)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3072)]));
      Out_local[(d_3_3 + 176)] = (Out_local[(d_3_3 + 176)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1696)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3136)]));
      Out_local[(d_3_3 + 192)] = (Out_local[(d_3_3 + 192)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1728)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3072)]));
      Out_local[(d_3_3 + 208)] = (Out_local[(d_3_3 + 208)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1728)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3136)]));
      Out_local[(d_3_3 + 224)] = (Out_local[(d_3_3 + 224)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1760)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3072)]));
      Out_local[(d_3_3 + 240)] = (Out_local[(d_3_3 + 240)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1760)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3136)]));
      Out_local[(d_3_3 + 8)] = (Out_local[(d_3_3 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1792)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3584)]));
      Out_local[(d_3_3 + 24)] = (Out_local[(d_3_3 + 24)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1792)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3648)]));
      Out_local[(d_3_3 + 40)] = (Out_local[(d_3_3 + 40)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1824)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3584)]));
      Out_local[(d_3_3 + 56)] = (Out_local[(d_3_3 + 56)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1824)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3648)]));
      Out_local[(d_3_3 + 72)] = (Out_local[(d_3_3 + 72)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1856)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3584)]));
      Out_local[(d_3_3 + 88)] = (Out_local[(d_3_3 + 88)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1856)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3648)]));
      Out_local[(d_3_3 + 104)] = (Out_local[(d_3_3 + 104)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1888)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3584)]));
      Out_local[(d_3_3 + 120)] = (Out_local[(d_3_3 + 120)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1888)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3648)]));
      Out_local[(d_3_3 + 136)] = (Out_local[(d_3_3 + 136)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1920)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3584)]));
      Out_local[(d_3_3 + 152)] = (Out_local[(d_3_3 + 152)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1920)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3648)]));
      Out_local[(d_3_3 + 168)] = (Out_local[(d_3_3 + 168)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1952)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3584)]));
      Out_local[(d_3_3 + 184)] = (Out_local[(d_3_3 + 184)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1952)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3648)]));
      Out_local[(d_3_3 + 200)] = (Out_local[(d_3_3 + 200)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1984)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3584)]));
      Out_local[(d_3_3 + 216)] = (Out_local[(d_3_3 + 216)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1984)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3648)]));
      Out_local[(d_3_3 + 232)] = (Out_local[(d_3_3 + 232)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 2016)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3584)]));
      Out_local[(d_3_3 + 248)] = (Out_local[(d_3_3 + 248)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 2016)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3648)]));
      Out_local[d_3_3] = (Out_local[d_3_3] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1537)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3200)]));
      Out_local[(d_3_3 + 16)] = (Out_local[(d_3_3 + 16)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1537)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3264)]));
      Out_local[(d_3_3 + 32)] = (Out_local[(d_3_3 + 32)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1569)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3200)]));
      Out_local[(d_3_3 + 48)] = (Out_local[(d_3_3 + 48)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1569)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3264)]));
      Out_local[(d_3_3 + 64)] = (Out_local[(d_3_3 + 64)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1601)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3200)]));
      Out_local[(d_3_3 + 80)] = (Out_local[(d_3_3 + 80)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1601)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3264)]));
      Out_local[(d_3_3 + 96)] = (Out_local[(d_3_3 + 96)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1633)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3200)]));
      Out_local[(d_3_3 + 112)] = (Out_local[(d_3_3 + 112)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1633)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3264)]));
      Out_local[(d_3_3 + 128)] = (Out_local[(d_3_3 + 128)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1665)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3200)]));
      Out_local[(d_3_3 + 144)] = (Out_local[(d_3_3 + 144)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1665)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3264)]));
      Out_local[(d_3_3 + 160)] = (Out_local[(d_3_3 + 160)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1697)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3200)]));
      Out_local[(d_3_3 + 176)] = (Out_local[(d_3_3 + 176)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1697)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3264)]));
      Out_local[(d_3_3 + 192)] = (Out_local[(d_3_3 + 192)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1729)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3200)]));
      Out_local[(d_3_3 + 208)] = (Out_local[(d_3_3 + 208)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1729)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3264)]));
      Out_local[(d_3_3 + 224)] = (Out_local[(d_3_3 + 224)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1761)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3200)]));
      Out_local[(d_3_3 + 240)] = (Out_local[(d_3_3 + 240)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1761)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3264)]));
      Out_local[(d_3_3 + 8)] = (Out_local[(d_3_3 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1793)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3712)]));
      Out_local[(d_3_3 + 24)] = (Out_local[(d_3_3 + 24)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1793)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3776)]));
      Out_local[(d_3_3 + 40)] = (Out_local[(d_3_3 + 40)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1825)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3712)]));
      Out_local[(d_3_3 + 56)] = (Out_local[(d_3_3 + 56)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1825)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3776)]));
      Out_local[(d_3_3 + 72)] = (Out_local[(d_3_3 + 72)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1857)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3712)]));
      Out_local[(d_3_3 + 88)] = (Out_local[(d_3_3 + 88)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1857)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3776)]));
      Out_local[(d_3_3 + 104)] = (Out_local[(d_3_3 + 104)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1889)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3712)]));
      Out_local[(d_3_3 + 120)] = (Out_local[(d_3_3 + 120)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1889)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3776)]));
      Out_local[(d_3_3 + 136)] = (Out_local[(d_3_3 + 136)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1921)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3712)]));
      Out_local[(d_3_3 + 152)] = (Out_local[(d_3_3 + 152)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1921)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3776)]));
      Out_local[(d_3_3 + 168)] = (Out_local[(d_3_3 + 168)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1953)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3712)]));
      Out_local[(d_3_3 + 184)] = (Out_local[(d_3_3 + 184)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1953)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3776)]));
      Out_local[(d_3_3 + 200)] = (Out_local[(d_3_3 + 200)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1985)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3712)]));
      Out_local[(d_3_3 + 216)] = (Out_local[(d_3_3 + 216)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 1985)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3776)]));
      Out_local[(d_3_3 + 232)] = (Out_local[(d_3_3 + 232)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 2017)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3712)]));
      Out_local[(d_3_3 + 248)] = (Out_local[(d_3_3 + 248)] + (Attn_shared[((((((int)threadIdx.x) >> 3) * 4) + (k2_1_3 * 2)) + 2017)] * V_shared[((((k2_1_3 * 256) + ((((int)threadIdx.x) & 7) * 8)) + d_3_3) + 3776)]));
    }
  }
  for (int ax0 = 0; ax0 < 2; ++ax0) {
    for (int ax2 = 0; ax2 < 8; ++ax2) {
      Out[(((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2)] = Out_local[((ax0 * 8) + ax2)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 64)] = Out_local[(((ax0 * 8) + ax2) + 16)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 1024)] = Out_local[(((ax0 * 8) + ax2) + 32)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 1088)] = Out_local[(((ax0 * 8) + ax2) + 48)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 2048)] = Out_local[(((ax0 * 8) + ax2) + 64)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 2112)] = Out_local[(((ax0 * 8) + ax2) + 80)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 3072)] = Out_local[(((ax0 * 8) + ax2) + 96)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 3136)] = Out_local[(((ax0 * 8) + ax2) + 112)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 4096)] = Out_local[(((ax0 * 8) + ax2) + 128)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 4160)] = Out_local[(((ax0 * 8) + ax2) + 144)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 5120)] = Out_local[(((ax0 * 8) + ax2) + 160)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 5184)] = Out_local[(((ax0 * 8) + ax2) + 176)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 6144)] = Out_local[(((ax0 * 8) + ax2) + 192)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 6208)] = Out_local[(((ax0 * 8) + ax2) + 208)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 7168)] = Out_local[(((ax0 * 8) + ax2) + 224)];
      Out[((((((((((int)blockIdx.x) >> 5) * 524288) + (ax0 * 262144)) + ((((int)blockIdx.x) & 31) * 8192)) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 8)) + ax2) + 7232)] = Out_local[(((ax0 * 8) + ax2) + 240)];
    }
  }
}

