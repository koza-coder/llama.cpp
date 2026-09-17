#include "common.hlsli"

// defines: TYPE_F32 or TYPE_F16 (+USE_16BIT), OP_{REGLU,GEGLU,SWIGLU,SWIGLU_OAI,SWIGLU_CLAMP,GEGLU_ERF,GEGLU_QUICK},
// NO_SPLIT when both halves come from src0

#if defined(TYPE_F16)
#define LOAD(buf, i)     LOAD_F16(buf, i)
#define STORE(buf, i, v) STORE_F16(buf, i, v)
#else
#define LOAD(buf, i)     LOAD_F32(buf, i)
#define STORE(buf, i, v) STORE_F32(buf, i, v)
#endif

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;

    uint stride_src01;
    uint stride_src02;
    uint stride_src03;

    uint stride_src11;
    uint stride_src12;
    uint stride_src13;

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint ne;
    uint ne0;
    uint ne1;
    uint ne2;

    uint swapped;
    float alpha;
    float limit;

    uint nwg_x;
};

float op(float a, float b) {
#if defined(OP_REGLU)
    return max(a, 0.0f) * b;
#elif defined(OP_GEGLU)
    const float val = 0.79788456080286535587989211986876f * a * (1.0f + 0.044715f * a * a);
    return 0.5f * a * (2.0f - 2.0f / (exp(2.0f * val) + 1.0f)) * b;
#elif defined(OP_SWIGLU)
    return a / (1.0f + exp(-a)) * b;
#elif defined(OP_SWIGLU_OAI)
    const float xi = min(a, limit);
    const float gi = max(min(b, limit), -limit);
    return xi / (1.0f + exp(-xi * alpha)) * (1.0f + gi);
#elif defined(OP_SWIGLU_CLAMP)
    const float gate = min(a, limit);
    const float up   = clamp(b, -limit, limit);
    return gate / (1.0f + exp(-gate)) * up;
#elif defined(OP_GEGLU_ERF)
    const float x    = abs(a * 0.7071067811865476f);
    const float sgn  = a >= 0.0f ? 1.0f : -1.0f;
    const float t    = 1.0f / (1.0f + 0.3275911f * x);
    const float y    = 1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x));
    return 0.5f * a * (1.0f + sgn * y) * b;
#elif defined(OP_GEGLU_QUICK)
    return a * (1.0f / (1.0f + exp(-1.702f * a))) * b;
#endif
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;

    const uint i_a   = offset_src0 + i3 * stride_src03 + i2 * stride_src02 + i1 * stride_src01 + i0;
    const uint i_b   = offset_src1 + i3 * stride_src13 + i2 * stride_src12 + i1 * stride_src11 + i0;
    const uint i_dst = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1 + i0;

#ifdef NO_SPLIT
    const float a = LOAD(src0, i_a + (swapped != 0 ? ne0 : 0));
    const float b = LOAD(src0, i_a + (swapped != 0 ? 0 : ne0));
#else
    const float a = LOAD(src0, i_a);
    const float b = LOAD(src1, i_b);
#endif
    STORE(dst, i_dst, op(a, b));
}
