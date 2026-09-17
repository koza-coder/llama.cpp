#include "common.hlsli"

// REPEAT (f32): dst[i0, i1, i2, i3] = src[i0 % ne00, i1 % ne01, i2 % ne02, i3 % ne03], any strides

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src0;
    uint stride_src1;
    uint stride_src2;
    uint stride_src3;
    uint stride_dst0;
    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;
    uint src_ne0;
    uint src_ne1;
    uint src_ne2;
    uint src_ne3;
    uint ne0;
    uint ne1;
    uint ne2;
    uint ne;
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
    const uint s = offset_src + (i0 % src_ne0) * stride_src0 + (i1 % src_ne1) * stride_src1 +
                   (i2 % src_ne2) * stride_src2 + (i3 % src_ne3) * stride_src3;
    STORE_F32(dst, offset_dst + i0 * stride_dst0 + i1 * stride_dst1 + i2 * stride_dst2 + i3 * stride_dst3,
              LOAD_F32(src, s));
}
