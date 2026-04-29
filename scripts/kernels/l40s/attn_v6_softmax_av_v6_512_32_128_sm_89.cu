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
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax);
extern "C" __global__ void __launch_bounds__(32) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum);
extern "C" __global__ void __launch_bounds__(128) main_kernel_2(half* __restrict__ Out, half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum, half* __restrict__ V) {
  half Out_local[64];
  __shared__ half Attn_shared[4096];
  __shared__ half V_shared[8192];
  for (int i_3_init = 0; i_3_init < 4; ++i_3_init) {
    Out_local[(i_3_init * 2)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 8)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 16)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 24)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 32)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 40)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 48)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 56)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 1)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 9)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 17)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 25)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 33)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 41)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 49)] = __float2half_rn(0.000000e+00f);
    Out_local[((i_3_init * 2) + 57)] = __float2half_rn(0.000000e+00f);
  }
  half2 __1;
    half2 __2;
    half2 __3;
      half2 v_ = *(half2*)(QK + (((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)));
      half2 v__1 = make_half2(RowMax[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))]);
      __3.x = (v_.x-v__1.x);
      __3.y = (v_.y-v__1.y);
    __2.x = hexp(__3.x);
    __2.y = hexp(__3.y);
    half2 v__2 = make_half2(RowSum[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))]);
    __1.x = (__2.x/v__2.x);
    __1.y = (__2.y/v__2.y);
  *(half2*)(Attn_shared + (((int)threadIdx.x) * 2)) = __1;
  half2 __4;
    half2 __5;
    half2 __6;
      half2 v__3 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 8192));
      half2 v__4 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)]);
      __6.x = (v__3.x-v__4.x);
      __6.y = (v__3.y-v__4.y);
    __5.x = hexp(__6.x);
    __5.y = hexp(__6.y);
    half2 v__5 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)]);
    __4.x = (__5.x/v__5.x);
    __4.y = (__5.y/v__5.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 256)) = __4;
  half2 __7;
    half2 __8;
    half2 __9;
      half2 v__6 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 16384));
      half2 v__7 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)]);
      __9.x = (v__6.x-v__7.x);
      __9.y = (v__6.y-v__7.y);
    __8.x = hexp(__9.x);
    __8.y = hexp(__9.y);
    half2 v__8 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)]);
    __7.x = (__8.x/v__8.x);
    __7.y = (__8.y/v__8.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 512)) = __7;
  half2 __10;
    half2 __11;
    half2 __12;
      half2 v__9 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 24576));
      half2 v__10 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)]);
      __12.x = (v__9.x-v__10.x);
      __12.y = (v__9.y-v__10.y);
    __11.x = hexp(__12.x);
    __11.y = hexp(__12.y);
    half2 v__11 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)]);
    __10.x = (__11.x/v__11.x);
    __10.y = (__11.y/v__11.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 768)) = __10;
  *(half4*)(V_shared + (((int)threadIdx.x) * 4)) = *(half4*)(V + (((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 512)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 512));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 1024));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 1536));
__asm__ __volatile__("cp.async.commit_group;");

  half2 __13;
    half2 __14;
    half2 __15;
      half2 v__12 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 16));
      half2 v__13 = make_half2(RowMax[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))]);
      __15.x = (v__12.x-v__13.x);
      __15.y = (v__12.y-v__13.y);
    __14.x = hexp(__15.x);
    __14.y = hexp(__15.y);
    half2 v__14 = make_half2(RowSum[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))]);
    __13.x = (__14.x/v__14.x);
    __13.y = (__14.y/v__14.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 1024)) = __13;
  half2 __16;
    half2 __17;
    half2 __18;
      half2 v__15 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 8208));
      half2 v__16 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)]);
      __18.x = (v__15.x-v__16.x);
      __18.y = (v__15.y-v__16.y);
    __17.x = hexp(__18.x);
    __17.y = hexp(__18.y);
    half2 v__17 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)]);
    __16.x = (__17.x/v__17.x);
    __16.y = (__17.y/v__17.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 1280)) = __16;
  half2 __19;
    half2 __20;
    half2 __21;
      half2 v__18 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 16400));
      half2 v__19 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)]);
      __21.x = (v__18.x-v__19.x);
      __21.y = (v__18.y-v__19.y);
    __20.x = hexp(__21.x);
    __20.y = hexp(__21.y);
    half2 v__20 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)]);
    __19.x = (__20.x/v__20.x);
    __19.y = (__20.y/v__20.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 1536)) = __19;
  half2 __22;
    half2 __23;
    half2 __24;
      half2 v__21 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 24592));
      half2 v__22 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)]);
      __24.x = (v__21.x-v__22.x);
      __24.y = (v__21.y-v__22.y);
    __23.x = hexp(__24.x);
    __23.y = hexp(__24.y);
    half2 v__23 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)]);
    __22.x = (__23.x/v__23.x);
    __22.y = (__23.y/v__23.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 1792)) = __22;
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 2048));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 2560));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 3072)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 3072));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 3584)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 3584));
__asm__ __volatile__("cp.async.commit_group;");

  half2 __25;
    half2 __26;
    half2 __27;
      half2 v__24 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 32));
      half2 v__25 = make_half2(RowMax[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))]);
      __27.x = (v__24.x-v__25.x);
      __27.y = (v__24.y-v__25.y);
    __26.x = hexp(__27.x);
    __26.y = hexp(__27.y);
    half2 v__26 = make_half2(RowSum[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))]);
    __25.x = (__26.x/v__26.x);
    __25.y = (__26.y/v__26.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 2048)) = __25;
  half2 __28;
    half2 __29;
    half2 __30;
      half2 v__27 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 8224));
      half2 v__28 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)]);
      __30.x = (v__27.x-v__28.x);
      __30.y = (v__27.y-v__28.y);
    __29.x = hexp(__30.x);
    __29.y = hexp(__30.y);
    half2 v__29 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)]);
    __28.x = (__29.x/v__29.x);
    __28.y = (__29.y/v__29.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 2304)) = __28;
  half2 __31;
    half2 __32;
    half2 __33;
      half2 v__30 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 16416));
      half2 v__31 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)]);
      __33.x = (v__30.x-v__31.x);
      __33.y = (v__30.y-v__31.y);
    __32.x = hexp(__33.x);
    __32.y = hexp(__33.y);
    half2 v__32 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)]);
    __31.x = (__32.x/v__32.x);
    __31.y = (__32.y/v__32.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 2560)) = __31;
  half2 __34;
    half2 __35;
    half2 __36;
      half2 v__33 = *(half2*)(QK + ((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + ((((int)threadIdx.x) & 7) * 2)) + 24608));
      half2 v__34 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)]);
      __36.x = (v__33.x-v__34.x);
      __36.y = (v__33.y-v__34.y);
    __35.x = hexp(__36.x);
    __35.y = hexp(__36.y);
    half2 v__35 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)]);
    __34.x = (__35.x/v__35.x);
    __34.y = (__35.y/v__35.y);
  *(half2*)(Attn_shared + ((((int)threadIdx.x) * 2) + 2816)) = __34;
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 4096)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 4096));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 4608)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 4608));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 5120)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 5120));
  *(half4*)(V_shared + ((((int)threadIdx.x) * 4) + 5632)) = *(half4*)(V + ((((((int)blockIdx.x) >> 3) * 65536) + (((int)threadIdx.x) * 4)) + 5632));
__asm__ __volatile__("cp.async.commit_group;");

  for (int k2_0_fused = 0; k2_0_fused < 29; ++k2_0_fused) {
    __syncthreads();
    half2 __37;
      half2 __38;
      half2 __39;
        half2 v__36 = *(half2*)(QK + (((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + (k2_0_fused * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 48));
        half2 v__37 = make_half2(RowMax[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))], RowMax[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))]);
        __39.x = (v__36.x-v__37.x);
        __39.y = (v__36.y-v__37.y);
      __38.x = hexp(__39.x);
      __38.y = hexp(__39.y);
      half2 v__38 = make_half2(RowSum[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))], RowSum[((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3))]);
      __37.x = (__38.x/v__38.x);
      __37.y = (__38.y/v__38.y);
    *(half2*)(Attn_shared + ((((k2_0_fused + 3) & 3) * 1024) + (((int)threadIdx.x) * 2))) = __37;
    half2 __40;
      half2 __41;
      half2 __42;
        half2 v__39 = *(half2*)(QK + (((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + (k2_0_fused * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 8240));
        half2 v__40 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)]);
        __42.x = (v__39.x-v__40.x);
        __42.y = (v__39.y-v__40.y);
      __41.x = hexp(__42.x);
      __41.y = hexp(__42.y);
      half2 v__41 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 16)]);
      __40.x = (__41.x/v__41.x);
      __40.y = (__41.y/v__41.y);
    *(half2*)(Attn_shared + (((((k2_0_fused + 3) & 3) * 1024) + (((int)threadIdx.x) * 2)) + 256)) = __40;
    half2 __43;
      half2 __44;
      half2 __45;
        half2 v__42 = *(half2*)(QK + (((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + (k2_0_fused * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 16432));
        half2 v__43 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)]);
        __45.x = (v__42.x-v__43.x);
        __45.y = (v__42.y-v__43.y);
      __44.x = hexp(__45.x);
      __44.y = hexp(__45.y);
      half2 v__44 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 32)]);
      __43.x = (__44.x/v__44.x);
      __43.y = (__44.y/v__44.y);
    *(half2*)(Attn_shared + (((((k2_0_fused + 3) & 3) * 1024) + (((int)threadIdx.x) * 2)) + 512)) = __43;
    half2 __46;
      half2 __47;
      half2 __48;
        half2 v__45 = *(half2*)(QK + (((((((int)blockIdx.x) * 32768) + ((((int)threadIdx.x) >> 3) * 512)) + (k2_0_fused * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 24624));
        half2 v__46 = make_half2(RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)], RowMax[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)]);
        __48.x = (v__45.x-v__46.x);
        __48.y = (v__45.y-v__46.y);
      __47.x = hexp(__48.x);
      __47.y = hexp(__48.y);
      half2 v__47 = make_half2(RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)], RowSum[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 3)) + 48)]);
      __46.x = (__47.x/v__47.x);
      __46.y = (__47.y/v__47.y);
    *(half2*)(Attn_shared + (((((k2_0_fused + 3) & 3) * 1024) + (((int)threadIdx.x) * 2)) + 768)) = __46;
    *(half4*)(V_shared + ((((k2_0_fused + 3) & 3) * 2048) + (((int)threadIdx.x) * 4))) = *(half4*)(V + (((((((int)blockIdx.x) >> 3) * 65536) + (k2_0_fused * 2048)) + (((int)threadIdx.x) * 4)) + 6144));
    *(half4*)(V_shared + (((((k2_0_fused + 3) & 3) * 2048) + (((int)threadIdx.x) * 4)) + 512)) = *(half4*)(V + (((((((int)blockIdx.x) >> 3) * 65536) + (k2_0_fused * 2048)) + (((int)threadIdx.x) * 4)) + 6656));
    *(half4*)(V_shared + (((((k2_0_fused + 3) & 3) * 2048) + (((int)threadIdx.x) * 4)) + 1024)) = *(half4*)(V + (((((((int)blockIdx.x) >> 3) * 65536) + (k2_0_fused * 2048)) + (((int)threadIdx.x) * 4)) + 7168));
    *(half4*)(V_shared + (((((k2_0_fused + 3) & 3) * 2048) + (((int)threadIdx.x) * 4)) + 1536)) = *(half4*)(V + (((((((int)blockIdx.x) >> 3) * 65536) + (k2_0_fused * 2048)) + (((int)threadIdx.x) * 4)) + 7680));
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

    __syncthreads();
    for (int k2_1 = 0; k2_1 < 2; ++k2_1) {
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        for (int k2_2 = 0; k2_2 < 8; ++k2_2) {
          Out_local[(i_3 * 2)] = (Out_local[(i_3 * 2)] + (Attn_shared[((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2)] * V_shared[(((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2))]));
          Out_local[((i_3 * 2) + 8)] = (Out_local[((i_3 * 2) + 8)] + (Attn_shared[((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
          Out_local[((i_3 * 2) + 16)] = (Out_local[((i_3 * 2) + 16)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 256)] * V_shared[(((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2))]));
          Out_local[((i_3 * 2) + 24)] = (Out_local[((i_3 * 2) + 24)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 256)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
          Out_local[((i_3 * 2) + 32)] = (Out_local[((i_3 * 2) + 32)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 512)] * V_shared[(((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2))]));
          Out_local[((i_3 * 2) + 40)] = (Out_local[((i_3 * 2) + 40)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 512)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
          Out_local[((i_3 * 2) + 48)] = (Out_local[((i_3 * 2) + 48)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 768)] * V_shared[(((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2))]));
          Out_local[((i_3 * 2) + 56)] = (Out_local[((i_3 * 2) + 56)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 768)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
          Out_local[((i_3 * 2) + 1)] = (Out_local[((i_3 * 2) + 1)] + (Attn_shared[((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
          Out_local[((i_3 * 2) + 9)] = (Out_local[((i_3 * 2) + 9)] + (Attn_shared[((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
          Out_local[((i_3 * 2) + 17)] = (Out_local[((i_3 * 2) + 17)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 256)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
          Out_local[((i_3 * 2) + 25)] = (Out_local[((i_3 * 2) + 25)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 256)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
          Out_local[((i_3 * 2) + 33)] = (Out_local[((i_3 * 2) + 33)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 512)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
          Out_local[((i_3 * 2) + 41)] = (Out_local[((i_3 * 2) + 41)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 512)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
          Out_local[((i_3 * 2) + 49)] = (Out_local[((i_3 * 2) + 49)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 768)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
          Out_local[((i_3 * 2) + 57)] = (Out_local[((i_3 * 2) + 57)] + (Attn_shared[(((((((k2_0_fused & 3) * 1024) + ((((int)threadIdx.x) >> 5) * 64)) + (i_3 * 16)) + (k2_1 * 8)) + k2_2) + 768)] * V_shared[((((((k2_0_fused & 3) * 2048) + (k2_1 * 1024)) + (k2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  for (int k2_1_1 = 0; k2_1_1 < 2; ++k2_1_1) {
    for (int i_3_1 = 0; i_3_1 < 4; ++i_3_1) {
      for (int k2_2_1 = 0; k2_2_1 < 8; ++k2_2_1) {
        Out_local[(i_3_1 * 2)] = (Out_local[(i_3_1 * 2)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1024)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        Out_local[((i_3_1 * 2) + 8)] = (Out_local[((i_3_1 * 2) + 8)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1024)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        Out_local[((i_3_1 * 2) + 16)] = (Out_local[((i_3_1 * 2) + 16)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1280)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        Out_local[((i_3_1 * 2) + 24)] = (Out_local[((i_3_1 * 2) + 24)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1280)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        Out_local[((i_3_1 * 2) + 32)] = (Out_local[((i_3_1 * 2) + 32)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1536)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        Out_local[((i_3_1 * 2) + 40)] = (Out_local[((i_3_1 * 2) + 40)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1536)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        Out_local[((i_3_1 * 2) + 48)] = (Out_local[((i_3_1 * 2) + 48)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1792)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        Out_local[((i_3_1 * 2) + 56)] = (Out_local[((i_3_1 * 2) + 56)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1792)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        Out_local[((i_3_1 * 2) + 1)] = (Out_local[((i_3_1 * 2) + 1)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1024)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        Out_local[((i_3_1 * 2) + 9)] = (Out_local[((i_3_1 * 2) + 9)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1024)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        Out_local[((i_3_1 * 2) + 17)] = (Out_local[((i_3_1 * 2) + 17)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1280)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        Out_local[((i_3_1 * 2) + 25)] = (Out_local[((i_3_1 * 2) + 25)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1280)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        Out_local[((i_3_1 * 2) + 33)] = (Out_local[((i_3_1 * 2) + 33)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1536)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        Out_local[((i_3_1 * 2) + 41)] = (Out_local[((i_3_1 * 2) + 41)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1536)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        Out_local[((i_3_1 * 2) + 49)] = (Out_local[((i_3_1 * 2) + 49)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1792)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        Out_local[((i_3_1 * 2) + 57)] = (Out_local[((i_3_1 * 2) + 57)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_1 * 16)) + (k2_1_1 * 8)) + k2_2_1) + 1792)] * V_shared[((((k2_1_1 * 1024) + (k2_2_1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int k2_1_2 = 0; k2_1_2 < 2; ++k2_1_2) {
    for (int i_3_2 = 0; i_3_2 < 4; ++i_3_2) {
      for (int k2_2_2 = 0; k2_2_2 < 8; ++k2_2_2) {
        Out_local[(i_3_2 * 2)] = (Out_local[(i_3_2 * 2)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2048)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4096)]));
        Out_local[((i_3_2 * 2) + 8)] = (Out_local[((i_3_2 * 2) + 8)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2048)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4160)]));
        Out_local[((i_3_2 * 2) + 16)] = (Out_local[((i_3_2 * 2) + 16)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2304)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4096)]));
        Out_local[((i_3_2 * 2) + 24)] = (Out_local[((i_3_2 * 2) + 24)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2304)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4160)]));
        Out_local[((i_3_2 * 2) + 32)] = (Out_local[((i_3_2 * 2) + 32)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2560)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4096)]));
        Out_local[((i_3_2 * 2) + 40)] = (Out_local[((i_3_2 * 2) + 40)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2560)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4160)]));
        Out_local[((i_3_2 * 2) + 48)] = (Out_local[((i_3_2 * 2) + 48)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2816)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4096)]));
        Out_local[((i_3_2 * 2) + 56)] = (Out_local[((i_3_2 * 2) + 56)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2816)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4160)]));
        Out_local[((i_3_2 * 2) + 1)] = (Out_local[((i_3_2 * 2) + 1)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2048)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4097)]));
        Out_local[((i_3_2 * 2) + 9)] = (Out_local[((i_3_2 * 2) + 9)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2048)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4161)]));
        Out_local[((i_3_2 * 2) + 17)] = (Out_local[((i_3_2 * 2) + 17)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2304)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4097)]));
        Out_local[((i_3_2 * 2) + 25)] = (Out_local[((i_3_2 * 2) + 25)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2304)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4161)]));
        Out_local[((i_3_2 * 2) + 33)] = (Out_local[((i_3_2 * 2) + 33)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2560)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4097)]));
        Out_local[((i_3_2 * 2) + 41)] = (Out_local[((i_3_2 * 2) + 41)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2560)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4161)]));
        Out_local[((i_3_2 * 2) + 49)] = (Out_local[((i_3_2 * 2) + 49)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2816)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4097)]));
        Out_local[((i_3_2 * 2) + 57)] = (Out_local[((i_3_2 * 2) + 57)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_2 * 16)) + (k2_1_2 * 8)) + k2_2_2) + 2816)] * V_shared[((((k2_1_2 * 1024) + (k2_2_2 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4161)]));
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int k2_1_3 = 0; k2_1_3 < 2; ++k2_1_3) {
    for (int i_3_3 = 0; i_3_3 < 4; ++i_3_3) {
      for (int k2_2_3 = 0; k2_2_3 < 8; ++k2_2_3) {
        Out_local[(i_3_3 * 2)] = (Out_local[(i_3_3 * 2)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3072)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6144)]));
        Out_local[((i_3_3 * 2) + 8)] = (Out_local[((i_3_3 * 2) + 8)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3072)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6208)]));
        Out_local[((i_3_3 * 2) + 16)] = (Out_local[((i_3_3 * 2) + 16)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3328)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6144)]));
        Out_local[((i_3_3 * 2) + 24)] = (Out_local[((i_3_3 * 2) + 24)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3328)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6208)]));
        Out_local[((i_3_3 * 2) + 32)] = (Out_local[((i_3_3 * 2) + 32)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3584)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6144)]));
        Out_local[((i_3_3 * 2) + 40)] = (Out_local[((i_3_3 * 2) + 40)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3584)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6208)]));
        Out_local[((i_3_3 * 2) + 48)] = (Out_local[((i_3_3 * 2) + 48)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3840)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6144)]));
        Out_local[((i_3_3 * 2) + 56)] = (Out_local[((i_3_3 * 2) + 56)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3840)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6208)]));
        Out_local[((i_3_3 * 2) + 1)] = (Out_local[((i_3_3 * 2) + 1)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3072)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6145)]));
        Out_local[((i_3_3 * 2) + 9)] = (Out_local[((i_3_3 * 2) + 9)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3072)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6209)]));
        Out_local[((i_3_3 * 2) + 17)] = (Out_local[((i_3_3 * 2) + 17)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3328)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6145)]));
        Out_local[((i_3_3 * 2) + 25)] = (Out_local[((i_3_3 * 2) + 25)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3328)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6209)]));
        Out_local[((i_3_3 * 2) + 33)] = (Out_local[((i_3_3 * 2) + 33)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3584)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6145)]));
        Out_local[((i_3_3 * 2) + 41)] = (Out_local[((i_3_3 * 2) + 41)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3584)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6209)]));
        Out_local[((i_3_3 * 2) + 49)] = (Out_local[((i_3_3 * 2) + 49)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3840)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6145)]));
        Out_local[((i_3_3 * 2) + 57)] = (Out_local[((i_3_3 * 2) + 57)] + (Attn_shared[((((((((int)threadIdx.x) >> 5) * 64) + (i_3_3 * 16)) + (k2_1_3 * 8)) + k2_2_3) + 3840)] * V_shared[((((k2_1_3 * 1024) + (k2_2_3 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6209)]));
      }
    }
  }
  for (int ax1 = 0; ax1 < 4; ++ax1) {
    Out[((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2))] = Out_local[(ax1 * 2)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 64)] = Out_local[((ax1 * 2) + 8)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2048)] = Out_local[((ax1 * 2) + 16)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2112)] = Out_local[((ax1 * 2) + 24)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4096)] = Out_local[((ax1 * 2) + 32)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4160)] = Out_local[((ax1 * 2) + 40)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6144)] = Out_local[((ax1 * 2) + 48)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6208)] = Out_local[((ax1 * 2) + 56)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 1)] = Out_local[((ax1 * 2) + 1)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 65)] = Out_local[((ax1 * 2) + 9)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2049)] = Out_local[((ax1 * 2) + 17)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 2113)] = Out_local[((ax1 * 2) + 25)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4097)] = Out_local[((ax1 * 2) + 33)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 4161)] = Out_local[((ax1 * 2) + 41)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6145)] = Out_local[((ax1 * 2) + 49)];
    Out[(((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (ax1 * 128)) + ((((int)threadIdx.x) & 31) * 2)) + 6209)] = Out_local[((ax1 * 2) + 57)];
  }
}

extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ QK, half* __restrict__ RowMax) {
  RowMax[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = __float2half_rn(-6.550400e+04f);
  for (int m_ax = 0; m_ax < 512; ++m_ax) {
    RowMax[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = max(RowMax[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))], QK[(((((int)blockIdx.x) * 32768) + (((int)threadIdx.x) * 512)) + m_ax)]);
  }
}

extern "C" __global__ void __launch_bounds__(32) main_kernel_1(half* __restrict__ QK, half* __restrict__ RowMax, half* __restrict__ RowSum) {
  RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __float2half_rn(0.000000e+00f);
  for (int s_ax = 0; s_ax < 512; ++s_ax) {
    RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (RowSum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + hexp((QK[(((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 512)) + s_ax)] - RowMax[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
  }
}

