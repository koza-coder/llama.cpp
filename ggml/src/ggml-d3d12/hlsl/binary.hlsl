#include "common.hlsli"

// defines: TYPE_{F32,F16}, OP_{ADD,SUB,MUL,DIV}
// dst may alias src0 (inplace) or src1: root UAVs on the same resource are fine in D3D12,
// and every thread reads its inputs before it writes its output.

// the math runs in f32 for every type: drivers may implement f16 division as an approximation
#if defined(TYPE_F32)
#define LOAD(buf, i)     LOAD_F32(buf, i)
#define STORE(buf, i, v) STORE_F32(buf, i, v)
#elif defined(TYPE_F16)
#define LOAD(buf, i)     ((float) LOAD_F16(buf, i))
#define STORE(buf, i, v) STORE_F16(buf, i, v)
#endif

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint ne;

    uint offset_src0;
    uint offset_src1;
    uint offset_dst;

    uint stride_src0_0;
    uint stride_src0_1;
    uint stride_src0_2;
    uint stride_src0_3;

    uint stride_src1_0;
    uint stride_src1_1;
    uint stride_src1_2;
    uint stride_src1_3;

    uint a_ne0;
    uint a_ne1;
    uint a_ne2;

    uint b_ne0;
    uint b_ne1;
    uint b_ne2;
    uint b_ne3;

    uint nwg_x;
};

float op(float a, float b) {
#if defined(OP_ADD)
    return a + b;
#elif defined(OP_SUB)
    return a - b;
#elif defined(OP_MUL)
    return a * b;
#elif defined(OP_DIV)
    return a / b;
#endif
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint idx = flat_index(gid, nwg_x);
    if (idx >= ne) {
        return;
    }

    uint i = idx;
    const uint a_i3 = i / (a_ne2 * a_ne1 * a_ne0);
    i = i % (a_ne2 * a_ne1 * a_ne0);
    const uint a_i2 = i / (a_ne1 * a_ne0);
    i = i % (a_ne1 * a_ne0);
    const uint a_i1 = i / a_ne0;
    const uint a_i0 = i % a_ne0;

    // src1 is broadcast (repeated) over src0
    const uint b_i0 = a_i0 % b_ne0;
    const uint b_i1 = a_i1 % b_ne1;
    const uint b_i2 = a_i2 % b_ne2;
    const uint b_i3 = a_i3 % b_ne3;

    const uint src0_i = offset_src0 + a_i0 * stride_src0_0 + a_i1 * stride_src0_1 + a_i2 * stride_src0_2 + a_i3 * stride_src0_3;
    const uint src1_i = offset_src1 + b_i0 * stride_src1_0 + b_i1 * stride_src1_1 + b_i2 * stride_src1_2 + b_i3 * stride_src1_3;

    const float a = LOAD(src0, src0_i);
    const float b = LOAD(src1, src1_i);
    STORE(dst, offset_dst + idx, op(a, b));
}
