#include "common.hlsli"

// PAD (f32): copy src into a larger dst at offset (lp0, lp1, lp2, lp3). Outside the copied box the
// value is 0, or with CIRCULAR the source index wraps around, which is ggml's circular pad mode.
// dst is contiguous; src keeps its own strides.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src0;
    uint stride_src1;
    uint stride_src2;
    uint stride_src3;
    uint src_ne0;
    uint src_ne1;
    uint src_ne2;
    uint src_ne3;
    uint ne0;
    uint ne1;
    uint ne2;
    uint ne;
    uint lp0;
    uint lp1;
    uint lp2;
    uint lp3;
    uint nwg_x;
};

#ifdef CIRCULAR
// ggml_wrap_around: coord is never below -size, so one addition is enough before the modulo
uint wrap_around(uint coord, uint size) {
    return (coord + size) % size;
}
#endif

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint dst_idx = i;
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;

#ifdef CIRCULAR
    const uint s0 = wrap_around(i0 - lp0, src_ne0);
    const uint s1 = wrap_around(i1 - lp1, src_ne1);
    const uint s2 = wrap_around(i2 - lp2, src_ne2);
    const uint s3 = wrap_around(i3 - lp3, src_ne3);
    const uint src_idx = offset_src + s3 * stride_src3 + s2 * stride_src2 + s1 * stride_src1 + s0 * stride_src0;
    STORE_F32(dst, offset_dst + dst_idx, LOAD_F32(src, src_idx));
#else
    // unsigned subtraction wraps to a huge value when the index is left of the pad, which the
    // >= src_ne test below rejects just as the CPU's two-sided range test does
    const uint s0 = i0 - lp0;
    const uint s1 = i1 - lp1;
    const uint s2 = i2 - lp2;
    const uint s3 = i3 - lp3;
    if (s0 < src_ne0 && s1 < src_ne1 && s2 < src_ne2 && s3 < src_ne3) {
        const uint src_idx = offset_src + s3 * stride_src3 + s2 * stride_src2 + s1 * stride_src1 + s0 * stride_src0;
        STORE_F32(dst, offset_dst + dst_idx, LOAD_F32(src, src_idx));
    } else {
        STORE_F32(dst, offset_dst + dst_idx, 0.0f);
    }
#endif
}
