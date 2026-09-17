#include "common.hlsli"

// f32 only: dst = src * scale + bias (dst may alias src)

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;

    uint stride_src1;
    uint stride_src2;
    uint stride_src3;

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint ne;
    uint ne0;
    uint ne1;
    uint ne2;

    float scale;
    float bias;

    uint nwg_x;
};

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

    const uint i_src = offset_src + i3 * stride_src3 + i2 * stride_src2 + i1 * stride_src1 + i0;
    const uint i_dst = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1 + i0;

    STORE_F32(dst, i_dst, LOAD_F32(src, i_src) * scale + bias);
}
