#include "common.hlsli"

// ADD_ID (f32): dst[i0, i1, i2] = src0[i0, i1, i2] + src1[i0, ids[i1, i2]]

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer ids  : register(u2);
RWByteAddressBuffer dst  : register(u3);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_ids;
    uint offset_dst;
    uint stride_src01;
    uint stride_src02;
    uint stride_src11;
    uint stride_ids1;
    uint stride_dst1;
    uint stride_dst2;
    uint ne0;
    uint ne1;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i2 = i / (ne1 * ne0);
    const uint i1 = (i / ne0) % ne1;
    const uint i0 = i % ne0;
    const int  id = LOAD_I32(ids, offset_ids + i2 * stride_ids1 + i1);
    const float a = LOAD_F32(src0, offset_src0 + i2 * stride_src02 + i1 * stride_src01 + i0);
    const float b = LOAD_F32(src1, offset_src1 + (uint) id * stride_src11 + i0);
    STORE_F32(dst, offset_dst + i2 * stride_dst2 + i1 * stride_dst1 + i0, a + b);
}
