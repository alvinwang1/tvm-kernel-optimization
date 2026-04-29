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
extern "C" __global__ void __launch_bounds__(64) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum);
extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax);
extern "C" __global__ void __launch_bounds__(128) main_kernel_2(half* __restrict__ Out, half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum, half* __restrict__ V);
extern "C" __global__ void __launch_bounds__(64) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum) {
  RowSum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = __float2half_rn(0.000000e+00f);
  for (int s_ax = 0; s_ax < 512; ++s_ax) {
    RowSum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 512)) + s_ax)] - RowMax[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))])));
  }
}

extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax) {
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __float2half_rn(-6.550400e+04f);
  for (int m_ax = 0; m_ax < 512; ++m_ax) {
    RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 512)) + m_ax)]);
  }
}

extern "C" __global__ void __launch_bounds__(128) main_kernel_2(half* __restrict__ Out, half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum, half* __restrict__ V) {
  half Out_local[16];
  __shared__ half Attn_shared[1024];
  __shared__ half V_shared[8192];
  Out_local[0] = __float2half_rn(0.000000e+00f);
  Out_local[1] = __float2half_rn(0.000000e+00f);
  Out_local[2] = __float2half_rn(0.000000e+00f);
  Out_local[3] = __float2half_rn(0.000000e+00f);
  Out_local[4] = __float2half_rn(0.000000e+00f);
  Out_local[5] = __float2half_rn(0.000000e+00f);
  Out_local[6] = __float2half_rn(0.000000e+00f);
  Out_local[7] = __float2half_rn(0.000000e+00f);
  Out_local[8] = __float2half_rn(0.000000e+00f);
  Out_local[9] = __float2half_rn(0.000000e+00f);
  Out_local[10] = __float2half_rn(0.000000e+00f);
  Out_local[11] = __float2half_rn(0.000000e+00f);
  Out_local[12] = __float2half_rn(0.000000e+00f);
  Out_local[13] = __float2half_rn(0.000000e+00f);
  Out_local[14] = __float2half_rn(0.000000e+00f);
  Out_local[15] = __float2half_rn(0.000000e+00f);
  half2 __1;
    half2 __2;
    half2 __3;
      half2 v_ = *(half2*)(QK + (((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)));
      half2 v__1 = make_half2(RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
      __3.x = (v_.x-v__1.x);
      __3.y = (v_.y-v__1.y);
    __2.x = hexp(__3.x);
    __2.y = hexp(__3.y);
    half2 v__2 = make_half2(RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
    __1.x = (__2.x/v__2.x);
    __1.y = (__2.y/v__2.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __1;
  *(uint4*)(V_shared + (((int)threadIdx.x) * 8)) = *(uint4*)(V + (((((int)blockIdx.x) >> 5) * 65536) + (((int)threadIdx.x) * 8)));
  *(uint4*)(V_shared + ((((int)threadIdx.x) * 8) + 1024)) = *(uint4*)(V + ((((((int)blockIdx.x) >> 5) * 65536) + (((int)threadIdx.x) * 8)) + 1024));
__asm__ __volatile__("cp.async.commit_group;");

  half2 __4;
    half2 __5;
    half2 __6;
      half2 v__3 = *(half2*)(QK + ((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 16));
      half2 v__4 = make_half2(RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
      __6.x = (v__3.x-v__4.x);
      __6.y = (v__3.y-v__4.y);
    __5.x = hexp(__6.x);
    __5.y = hexp(__6.y);
    half2 v__5 = make_half2(RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
    __4.x = (__5.x/v__5.x);
    __4.y = (__5.y/v__5.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 256)) = __4;
  *(uint4*)(V_shared + ((((int)threadIdx.x) * 8) + 2048)) = *(uint4*)(V + ((((((int)blockIdx.x) >> 5) * 65536) + (((int)threadIdx.x) * 8)) + 2048));
  *(uint4*)(V_shared + ((((int)threadIdx.x) * 8) + 3072)) = *(uint4*)(V + ((((((int)blockIdx.x) >> 5) * 65536) + (((int)threadIdx.x) * 8)) + 3072));
__asm__ __volatile__("cp.async.commit_group;");

  half2 __7;
    half2 __8;
    half2 __9;
      half2 v__6 = *(half2*)(QK + ((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 32));
      half2 v__7 = make_half2(RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
      __9.x = (v__6.x-v__7.x);
      __9.y = (v__6.y-v__7.y);
    __8.x = hexp(__9.x);
    __8.y = hexp(__9.y);
    half2 v__8 = make_half2(RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
    __7.x = (__8.x/v__8.x);
    __7.y = (__8.y/v__8.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 512)) = __7;
  *(uint4*)(V_shared + ((((int)threadIdx.x) * 8) + 4096)) = *(uint4*)(V + ((((((int)blockIdx.x) >> 5) * 65536) + (((int)threadIdx.x) * 8)) + 4096));
  *(uint4*)(V_shared + ((((int)threadIdx.x) * 8) + 5120)) = *(uint4*)(V + ((((((int)blockIdx.x) >> 5) * 65536) + (((int)threadIdx.x) * 8)) + 5120));
__asm__ __volatile__("cp.async.commit_group;");

  for (int k2_0_fused = 0; k2_0_fused < 29; ++k2_0_fused) {
    __syncthreads();
    half2 __10;
      half2 __11;
      half2 __12;
        half2 v__9 = *(half2*)(QK + (((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (k2_0_fused * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 48));
        half2 v__10 = make_half2(RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
        __12.x = (v__9.x-v__10.x);
        __12.y = (v__9.y-v__10.y);
      __11.x = hexp(__12.x);
      __11.y = hexp(__12.y);
      half2 v__11 = make_half2(RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
      __10.x = (__11.x/v__11.x);
      __10.y = (__11.y/v__11.y);
    *(half2*)(Attn_shared + ((((k2_0_fused + 3) & 3) * 256) + (((int)threadIdx.x) * 2))) = __10;
    *(uint4*)(V_shared + ((((k2_0_fused + 3) & 3) * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(V + (((((((int)blockIdx.x) >> 5) * 65536) + (k2_0_fused * 2048)) + (((int)threadIdx.x) * 8)) + 6144));
    *(uint4*)(V_shared + (((((k2_0_fused + 3) & 3) * 2048) + (((int)threadIdx.x) * 8)) + 1024)) = *(uint4*)(V + (((((((int)blockIdx.x) >> 5) * 65536) + (k2_0_fused * 2048)) + (((int)threadIdx.x) * 8)) + 7168));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

    __syncthreads();
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16))] * V_shared[(((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4))]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16))] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16))] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 2)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16))] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 3)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 1)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 128)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 1)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 129)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 1)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 130)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 1)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 131)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 2)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 256)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 2)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 257)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 2)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 258)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 2)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 259)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 3)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 384)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 3)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 385)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 3)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 386)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 3)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 387)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 4)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 512)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 4)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 513)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 4)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 514)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 4)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 515)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 5)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 640)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 5)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 641)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 5)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 642)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 5)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 643)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 6)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 768)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 6)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 769)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 6)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 770)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 6)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 771)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 7)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 896)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 7)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 897)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 7)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 898)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 7)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 899)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 8)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1024)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 8)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1025)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 8)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1026)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 8)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1027)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 9)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1152)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 9)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1153)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 9)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1154)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 9)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1155)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 10)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1280)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 10)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1281)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 10)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1282)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 10)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1283)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 11)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1408)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 11)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1409)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 11)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1410)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 11)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1411)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 12)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1536)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 12)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1537)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 12)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1538)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 12)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1539)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 13)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1664)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 13)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1665)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 13)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1666)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 13)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1667)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 14)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1792)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 14)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1793)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 14)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1794)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 14)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1795)]));
      Out_local[(i_3 * 4)] = (Out_local[(i_3 * 4)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 15)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1920)]));
      Out_local[((i_3 * 4) + 1)] = (Out_local[((i_3 * 4) + 1)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 15)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1921)]));
      Out_local[((i_3 * 4) + 2)] = (Out_local[((i_3 * 4) + 2)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 15)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1922)]));
      Out_local[((i_3 * 4) + 3)] = (Out_local[((i_3 * 4) + 3)] + (Attn_shared[(((((k2_0_fused & 3) * 256) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + 15)] * V_shared[((((k2_0_fused & 3) * 2048) + ((((int)threadIdx.x) & 31) * 4)) + 1923)]));
    }
  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  for (int i_3_1 = 0; i_3_1 < 4; ++i_3_1) {
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 256)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2048)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 256)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2049)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 256)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2050)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 256)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2051)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 257)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2176)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 257)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2177)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 257)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2178)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 257)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2179)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 258)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2304)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 258)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2305)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 258)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2306)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 258)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2307)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 259)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2432)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 259)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2433)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 259)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2434)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 259)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2435)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 260)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2560)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 260)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2561)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 260)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2562)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 260)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2563)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 261)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2688)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 261)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2689)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 261)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2690)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 261)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2691)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 262)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2816)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 262)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2817)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 262)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2818)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 262)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2819)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 263)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2944)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 263)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2945)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 263)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2946)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 263)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 2947)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 264)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3072)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 264)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3073)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 264)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3074)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 264)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3075)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 265)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3200)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 265)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3201)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 265)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3202)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 265)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3203)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 266)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3328)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 266)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3329)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 266)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3330)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 266)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3331)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 267)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3456)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 267)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3457)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 267)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3458)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 267)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3459)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 268)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3584)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 268)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3585)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 268)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3586)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 268)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3587)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 269)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3712)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 269)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3713)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 269)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3714)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 269)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3715)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 270)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3840)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 270)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3841)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 270)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3842)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 270)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3843)]));
    Out_local[(i_3_1 * 4)] = (Out_local[(i_3_1 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 271)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3968)]));
    Out_local[((i_3_1 * 4) + 1)] = (Out_local[((i_3_1 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 271)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3969)]));
    Out_local[((i_3_1 * 4) + 2)] = (Out_local[((i_3_1 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 271)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3970)]));
    Out_local[((i_3_1 * 4) + 3)] = (Out_local[((i_3_1 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + 271)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 3971)]));
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int i_3_2 = 0; i_3_2 < 4; ++i_3_2) {
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 512)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4096)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 512)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4097)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 512)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4098)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 512)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4099)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 513)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4224)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 513)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4225)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 513)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4226)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 513)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4227)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 514)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4352)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 514)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4353)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 514)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4354)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 514)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4355)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 515)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4480)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 515)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4481)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 515)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4482)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 515)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4483)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 516)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4608)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 516)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4609)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 516)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4610)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 516)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4611)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 517)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4736)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 517)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4737)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 517)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4738)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 517)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4739)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 518)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4864)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 518)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4865)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 518)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4866)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 518)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4867)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 519)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4992)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 519)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4993)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 519)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4994)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 519)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 4995)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 520)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5120)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 520)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5121)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 520)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5122)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 520)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5123)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 521)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5248)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 521)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5249)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 521)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5250)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 521)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5251)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 522)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5376)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 522)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5377)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 522)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5378)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 522)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5379)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 523)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5504)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 523)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5505)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 523)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5506)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 523)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5507)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 524)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5632)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 524)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5633)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 524)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5634)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 524)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5635)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 525)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5760)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 525)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5761)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 525)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5762)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 525)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5763)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 526)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5888)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 526)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5889)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 526)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5890)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 526)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 5891)]));
    Out_local[(i_3_2 * 4)] = (Out_local[(i_3_2 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 527)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6016)]));
    Out_local[((i_3_2 * 4) + 1)] = (Out_local[((i_3_2 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 527)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6017)]));
    Out_local[((i_3_2 * 4) + 2)] = (Out_local[((i_3_2 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 527)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6018)]));
    Out_local[((i_3_2 * 4) + 3)] = (Out_local[((i_3_2 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + 527)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6019)]));
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int i_3_3 = 0; i_3_3 < 4; ++i_3_3) {
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 768)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6144)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 768)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6145)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 768)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6146)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 768)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6147)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 769)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6272)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 769)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6273)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 769)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6274)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 769)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6275)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 770)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6400)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 770)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6401)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 770)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6402)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 770)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6403)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 771)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6528)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 771)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6529)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 771)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6530)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 771)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6531)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 772)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6656)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 772)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6657)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 772)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6658)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 772)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6659)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 773)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6784)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 773)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6785)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 773)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6786)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 773)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6787)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 774)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6912)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 774)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6913)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 774)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6914)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 774)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 6915)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 775)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7040)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 775)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7041)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 775)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7042)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 775)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7043)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 776)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7168)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 776)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7169)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 776)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7170)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 776)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7171)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 777)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7296)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 777)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7297)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 777)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7298)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 777)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7299)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 778)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7424)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 778)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7425)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 778)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7426)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 778)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7427)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 779)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7552)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 779)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7553)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 779)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7554)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 779)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7555)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 780)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7680)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 780)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7681)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 780)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7682)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 780)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7683)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 781)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7808)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 781)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7809)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 781)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7810)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 781)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7811)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 782)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7936)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 782)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7937)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 782)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7938)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 782)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 7939)]));
    Out_local[(i_3_3 * 4)] = (Out_local[(i_3_3 * 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 783)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 8064)]));
    Out_local[((i_3_3 * 4) + 1)] = (Out_local[((i_3_3 * 4) + 1)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 783)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 8065)]));
    Out_local[((i_3_3 * 4) + 2)] = (Out_local[((i_3_3 * 4) + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 783)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 8066)]));
    Out_local[((i_3_3 * 4) + 3)] = (Out_local[((i_3_3 * 4) + 3)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + 783)] * V_shared[(((((int)threadIdx.x) & 31) * 4) + 8067)]));
  }
  Out[(((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4))] = Out_local[0];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 1)] = Out_local[1];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 2)] = Out_local[2];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 3)] = Out_local[3];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 128)] = Out_local[4];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 129)] = Out_local[5];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 130)] = Out_local[6];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 131)] = Out_local[7];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 256)] = Out_local[8];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 257)] = Out_local[9];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 258)] = Out_local[10];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 259)] = Out_local[11];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 384)] = Out_local[12];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 385)] = Out_local[13];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 386)] = Out_local[14];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 512)) + ((((int)threadIdx.x) & 31) * 4)) + 387)] = Out_local[15];
}

