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
extern "C" __global__ void __launch_bounds__(128) main_kernel_2(half* __restrict__ Out, half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum, half* __restrict__ V);
extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax);
extern "C" __global__ void __launch_bounds__(64) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum);
extern "C" __global__ void __launch_bounds__(128) main_kernel_2(half* __restrict__ Out, half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum, half* __restrict__ V) {
  half Out_local[16];
  __shared__ half Attn_shared[2048];
  __shared__ half V_shared[16384];
  Out_local[0] = __float2half_rn(0.000000e+00f);
  Out_local[2] = __float2half_rn(0.000000e+00f);
  Out_local[4] = __float2half_rn(0.000000e+00f);
  Out_local[6] = __float2half_rn(0.000000e+00f);
  Out_local[8] = __float2half_rn(0.000000e+00f);
  Out_local[10] = __float2half_rn(0.000000e+00f);
  Out_local[12] = __float2half_rn(0.000000e+00f);
  Out_local[14] = __float2half_rn(0.000000e+00f);
  Out_local[1] = __float2half_rn(0.000000e+00f);
  Out_local[3] = __float2half_rn(0.000000e+00f);
  Out_local[5] = __float2half_rn(0.000000e+00f);
  Out_local[7] = __float2half_rn(0.000000e+00f);
  Out_local[9] = __float2half_rn(0.000000e+00f);
  Out_local[11] = __float2half_rn(0.000000e+00f);
  Out_local[13] = __float2half_rn(0.000000e+00f);
  Out_local[15] = __float2half_rn(0.000000e+00f);
  half4 __1;
    half4 __2;
    half4 __3;
      half4 v_ = *(half4*)(QK + (((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 4)));
      half4 v__1 = make_half4(RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
      __3.x = (v_.x-v__1.x);
      __3.y = (v_.y-v__1.y);
      __3.z = (v_.z-v__1.z);
      __3.w = (v_.w-v__1.w);
    __2.x = hexp(__3.x);
    __2.y = hexp(__3.y);
    __2.z = hexp(__3.z);
    __2.w = hexp(__3.w);
    half4 v__2 = make_half4(RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
    __1.x = (__2.x/v__2.x);
    __1.y = (__2.y/v__2.y);
    __1.z = (__2.z/v__2.z);
    __1.w = (__2.w/v__2.w);
  *(half4*)(Attn_shared + (((int)threadIdx.x) * 4)) = __1;
  *(half4*)(V_shared + (((int)threadIdx.x) * 4)) = *(half4*)(V + (((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 512)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 1024));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 1536));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 2048));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 2560));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 3072));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 3584));
__asm__ __volatile__("cp.async.commit_group;");

  half4 __4;
    half4 __5;
    half4 __6;
      half4 v__3 = *(half4*)(QK + ((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 4)) + 32));
      half4 v__4 = make_half4(RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
      __6.x = (v__3.x-v__4.x);
      __6.y = (v__3.y-v__4.y);
      __6.z = (v__3.z-v__4.z);
      __6.w = (v__3.w-v__4.w);
    __5.x = hexp(__6.x);
    __5.y = hexp(__6.y);
    __5.z = hexp(__6.z);
    __5.w = hexp(__6.w);
    half4 v__5 = make_half4(RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
    __4.x = (__5.x/v__5.x);
    __4.y = (__5.y/v__5.y);
    __4.z = (__5.z/v__5.z);
    __4.w = (__5.w/v__5.w);
  *(half4*)(Attn_shared + ((((int)threadIdx.x) * 4) + 512)) = __4;
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 4096));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 4608)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 4608));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 5120)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 5120));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 5632)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 5632));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 6144)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 6144));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 6656)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 6656));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 7168)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 7168));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 7680)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 7680));
__asm__ __volatile__("cp.async.commit_group;");

  half4 __7;
    half4 __8;
    half4 __9;
      half4 v__6 = *(half4*)(QK + ((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 4)) + 64));
      half4 v__7 = make_half4(RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
      __9.x = (v__6.x-v__7.x);
      __9.y = (v__6.y-v__7.y);
      __9.z = (v__6.z-v__7.z);
      __9.w = (v__6.w-v__7.w);
    __8.x = hexp(__9.x);
    __8.y = hexp(__9.y);
    __8.z = hexp(__9.z);
    __8.w = hexp(__9.w);
    half4 v__8 = make_half4(RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
    __7.x = (__8.x/v__8.x);
    __7.y = (__8.y/v__8.y);
    __7.z = (__8.z/v__8.z);
    __7.w = (__8.w/v__8.w);
  *(half4*)(Attn_shared + ((((int)threadIdx.x) * 4) + 1024)) = __7;
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 8192)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 8192));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 8704)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 8704));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 9216)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 9216));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 9728)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 9728));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 10240)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 10240));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 10752)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 10752));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 11264)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 11264));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 11776)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 11776));
__asm__ __volatile__("cp.async.commit_group;");

  half4 __10;
    half4 __11;
    half4 __12;
      half4 v__9 = *(half4*)(QK + ((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + ((((int)threadIdx.x) & 7) * 4)) + 96));
      half4 v__10 = make_half4(RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
      __12.x = (v__9.x-v__10.x);
      __12.y = (v__9.y-v__10.y);
      __12.z = (v__9.z-v__10.z);
      __12.w = (v__9.w-v__10.w);
    __11.x = hexp(__12.x);
    __11.y = hexp(__12.y);
    __11.z = hexp(__12.z);
    __11.w = hexp(__12.w);
    half4 v__11 = make_half4(RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3))]);
    __10.x = (__11.x/v__11.x);
    __10.y = (__11.y/v__11.y);
    __10.z = (__11.z/v__11.z);
    __10.w = (__11.w/v__11.w);
  *(half4*)(Attn_shared + ((((int)threadIdx.x) * 4) + 1536)) = __10;
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 12288)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 12288));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 12800)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 12800));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 13312)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 13312));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 13824)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 13824));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 14336)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 14336));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 14848)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 14848));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 15360)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 15360));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 15872)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 16384) + (((int)threadIdx.x) * 4)) + 15872));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

  __syncthreads();
  for (int d_3 = 0; d_3 < 2; ++d_3) {
    for (int k2_2 = 0; k2_2 < 32; ++k2_2) {
      Out_local[d_3] = (Out_local[d_3] + (Attn_shared[(((((int)threadIdx.x) >> 5) * 32) + k2_2)] * V_shared[(((k2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3)]));
      Out_local[(d_3 + 2)] = (Out_local[(d_3 + 2)] + (Attn_shared[(((((int)threadIdx.x) >> 5) * 32) + k2_2)] * V_shared[((((k2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3) + 64)]));
      Out_local[(d_3 + 4)] = (Out_local[(d_3 + 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2) + 128)] * V_shared[(((k2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3)]));
      Out_local[(d_3 + 6)] = (Out_local[(d_3 + 6)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2) + 128)] * V_shared[((((k2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3) + 64)]));
      Out_local[(d_3 + 8)] = (Out_local[(d_3 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2) + 256)] * V_shared[(((k2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3)]));
      Out_local[(d_3 + 10)] = (Out_local[(d_3 + 10)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2) + 256)] * V_shared[((((k2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3) + 64)]));
      Out_local[(d_3 + 12)] = (Out_local[(d_3 + 12)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2) + 384)] * V_shared[(((k2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3)]));
      Out_local[(d_3 + 14)] = (Out_local[(d_3 + 14)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2) + 384)] * V_shared[((((k2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3) + 64)]));
    }
  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  for (int d_3_1 = 0; d_3_1 < 2; ++d_3_1) {
    for (int k2_2_1 = 0; k2_2_1 < 32; ++k2_2_1) {
      Out_local[d_3_1] = (Out_local[d_3_1] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_1) + 512)] * V_shared[((((k2_2_1 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_1) + 4096)]));
      Out_local[(d_3_1 + 2)] = (Out_local[(d_3_1 + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_1) + 512)] * V_shared[((((k2_2_1 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_1) + 4160)]));
      Out_local[(d_3_1 + 4)] = (Out_local[(d_3_1 + 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_1) + 640)] * V_shared[((((k2_2_1 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_1) + 4096)]));
      Out_local[(d_3_1 + 6)] = (Out_local[(d_3_1 + 6)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_1) + 640)] * V_shared[((((k2_2_1 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_1) + 4160)]));
      Out_local[(d_3_1 + 8)] = (Out_local[(d_3_1 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_1) + 768)] * V_shared[((((k2_2_1 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_1) + 4096)]));
      Out_local[(d_3_1 + 10)] = (Out_local[(d_3_1 + 10)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_1) + 768)] * V_shared[((((k2_2_1 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_1) + 4160)]));
      Out_local[(d_3_1 + 12)] = (Out_local[(d_3_1 + 12)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_1) + 896)] * V_shared[((((k2_2_1 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_1) + 4096)]));
      Out_local[(d_3_1 + 14)] = (Out_local[(d_3_1 + 14)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_1) + 896)] * V_shared[((((k2_2_1 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_1) + 4160)]));
    }
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int d_3_2 = 0; d_3_2 < 2; ++d_3_2) {
    for (int k2_2_2 = 0; k2_2_2 < 32; ++k2_2_2) {
      Out_local[d_3_2] = (Out_local[d_3_2] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_2) + 1024)] * V_shared[((((k2_2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_2) + 8192)]));
      Out_local[(d_3_2 + 2)] = (Out_local[(d_3_2 + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_2) + 1024)] * V_shared[((((k2_2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_2) + 8256)]));
      Out_local[(d_3_2 + 4)] = (Out_local[(d_3_2 + 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_2) + 1152)] * V_shared[((((k2_2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_2) + 8192)]));
      Out_local[(d_3_2 + 6)] = (Out_local[(d_3_2 + 6)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_2) + 1152)] * V_shared[((((k2_2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_2) + 8256)]));
      Out_local[(d_3_2 + 8)] = (Out_local[(d_3_2 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_2) + 1280)] * V_shared[((((k2_2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_2) + 8192)]));
      Out_local[(d_3_2 + 10)] = (Out_local[(d_3_2 + 10)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_2) + 1280)] * V_shared[((((k2_2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_2) + 8256)]));
      Out_local[(d_3_2 + 12)] = (Out_local[(d_3_2 + 12)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_2) + 1408)] * V_shared[((((k2_2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_2) + 8192)]));
      Out_local[(d_3_2 + 14)] = (Out_local[(d_3_2 + 14)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_2) + 1408)] * V_shared[((((k2_2_2 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_2) + 8256)]));
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int d_3_3 = 0; d_3_3 < 2; ++d_3_3) {
    for (int k2_2_3 = 0; k2_2_3 < 32; ++k2_2_3) {
      Out_local[d_3_3] = (Out_local[d_3_3] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_3) + 1536)] * V_shared[((((k2_2_3 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_3) + 12288)]));
      Out_local[(d_3_3 + 2)] = (Out_local[(d_3_3 + 2)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_3) + 1536)] * V_shared[((((k2_2_3 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_3) + 12352)]));
      Out_local[(d_3_3 + 4)] = (Out_local[(d_3_3 + 4)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_3) + 1664)] * V_shared[((((k2_2_3 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_3) + 12288)]));
      Out_local[(d_3_3 + 6)] = (Out_local[(d_3_3 + 6)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_3) + 1664)] * V_shared[((((k2_2_3 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_3) + 12352)]));
      Out_local[(d_3_3 + 8)] = (Out_local[(d_3_3 + 8)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_3) + 1792)] * V_shared[((((k2_2_3 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_3) + 12288)]));
      Out_local[(d_3_3 + 10)] = (Out_local[(d_3_3 + 10)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_3) + 1792)] * V_shared[((((k2_2_3 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_3) + 12352)]));
      Out_local[(d_3_3 + 12)] = (Out_local[(d_3_3 + 12)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_3) + 1920)] * V_shared[((((k2_2_3 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_3) + 12288)]));
      Out_local[(d_3_3 + 14)] = (Out_local[(d_3_3 + 14)] + (Attn_shared[((((((int)threadIdx.x) >> 5) * 32) + k2_2_3) + 1920)] * V_shared[((((k2_2_3 * 128) + ((((int)threadIdx.x) & 31) * 2)) + d_3_3) + 12352)]));
    }
  }
  Out[(((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2))] = Out_local[0];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 64)] = Out_local[2];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 512)] = Out_local[4];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 576)] = Out_local[6];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1024)] = Out_local[8];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1088)] = Out_local[10];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1536)] = Out_local[12];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1600)] = Out_local[14];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1)] = Out_local[1];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 65)] = Out_local[3];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 513)] = Out_local[5];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 577)] = Out_local[7];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1025)] = Out_local[9];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1089)] = Out_local[11];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1537)] = Out_local[13];
  Out[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1601)] = Out_local[15];
}

extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax) {
  RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __float2half_rn(-6.550400e+04f);
  for (int m_ax = 0; m_ax < 128; ++m_ax) {
    RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 128)) + m_ax)]);
  }
}

extern "C" __global__ void __launch_bounds__(64) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum) {
  RowSum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = __float2half_rn(0.000000e+00f);
  for (int s_ax = 0; s_ax < 128; ++s_ax) {
    RowSum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 128)) + s_ax)] - RowMax[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))])));
  }
}

