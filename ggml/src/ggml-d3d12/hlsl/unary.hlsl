#include "common.hlsli"

// defines: TYPE_F32 or TYPE_F16 (+USE_16BIT) and one op define; math in f32

#if defined(TYPE_F16)
#define LOAD(buf, i)     LOAD_F16(buf, i)
#define STORE(buf, i, v) STORE_F16(buf, i, v)
#else
#define LOAD(buf, i)     LOAD_F32(buf, i)
#define STORE(buf, i, v) STORE_F32(buf, i, v)
#endif

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint ne;
    uint offset_src;
    uint offset_dst;

    uint stride_src0;
    uint stride_src1;
    uint stride_src2;
    uint stride_src3;

    uint ne0;
    uint ne1;
    uint ne2;

    float p0;   // clamp min
    float p1;   // clamp max

    uint nwg_x;
};

float erf_approx(float x) {
    const float s  = x >= 0.0f ? 1.0f : -1.0f;
    const float ax = abs(x);
    const float t  = 1.0f / (1.0f + 0.3275911f * ax);
    const float y  = 1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t) * exp(-ax * ax);
    return s * y;
}

float apply(float x) {
#if defined(ABS)
    return abs(x);
#elif defined(SGN)
    return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
#elif defined(NEG)
    return -x;
#elif defined(STEP)
    return x > 0.0f ? 1.0f : 0.0f;
#elif defined(TANH)
    return tanh(clamp(x, -9.010913f, 9.010913f));
#elif defined(RELU)
    return x > 0.0f ? x : 0.0f;
#elif defined(ELU)
    return x > 0.0f ? x : exp(x) - 1.0f;
#elif defined(HARDSIGMOID)
    return min(1.0f, max(0.0f, (x + 3.0f) / 6.0f));
#elif defined(SIGMOID)
    return 1.0f / (1.0f + exp(-x));
#elif defined(SILU)
    return x / (1.0f + exp(-x));
#elif defined(EXP)
    return exp(x);
#elif defined(LOG)
    return log(x);
#elif defined(CLAMP)
    return clamp(x, p0, p1);
#elif defined(HARDSWISH)
    return x * min(1.0f, max(0.0f, (x + 3.0f) / 6.0f));
#elif defined(GELU)
    return 0.5f * x * (1.0f + tanh(clamp(0.7978845608028654f * (x + 0.044715f * x * x * x), -9.010913f, 9.010913f)));
#elif defined(GELU_QUICK)
    return x * (1.0f / (1.0f + exp(clamp(-1.702f * x, -80.0f, 80.0f))));
#elif defined(GELU_ERF)
    return 0.5f * x * (1.0f + erf_approx(x * 0.7071067811865476f));
#elif defined(SOFTPLUS)
    return x > 20.0f ? x : log(1.0f + exp(x));
#elif defined(EXPM1)
    return exp(x) - 1.0f;
#elif defined(FLOOR)
    return floor(x);
#elif defined(CEIL)
    return ceil(x);
#elif defined(ROUND)
    return x >= 0.0f ? floor(x + 0.5f) : ceil(x - 0.5f);
#elif defined(TRUNC)
    return trunc(x);
#elif defined(SQR)
    return x * x;
#elif defined(SQRT)
    return sqrt(x);
#elif defined(SIN)
    return sin(x);
#elif defined(COS)
    return cos(x);
#endif
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint flat_i = flat_index(gid, nwg_x);
    if (flat_i >= ne) {
        return;
    }
    uint i = flat_i;
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;
    const uint src_idx = i0 * stride_src0 + i1 * stride_src1 + i2 * stride_src2 + i3 * stride_src3;
    STORE(dst, offset_dst + flat_i, apply(LOAD(src, offset_src + src_idx)));
}
