#include "common.hlsli"

// CONCAT of two 4-byte tensors (f32 or i32, copied as raw bits) along dim; any strides

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;

    uint stride_src00;
    uint stride_src01;
    uint stride_src02;
    uint stride_src03;
    uint stride_src10;
    uint stride_src11;
    uint stride_src12;
    uint stride_src13;
    uint stride_dst0;
    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint ne;
    uint ne0;
    uint ne1;
    uint ne2;

    uint dim;
    uint src0_nedim;   // src0->ne[dim]

    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    uint idx[4];
    idx[3] = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    idx[2] = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    idx[1] = i / ne0;
    idx[0] = i % ne0;

    const uint i_dst = offset_dst + idx[0] * stride_dst0 + idx[1] * stride_dst1 + idx[2] * stride_dst2 + idx[3] * stride_dst3;
    uint bits;
    if (idx[dim] < src0_nedim) {
        bits = src0.Load((offset_src0 + idx[0] * stride_src00 + idx[1] * stride_src01 + idx[2] * stride_src02 + idx[3] * stride_src03) * 4);
    } else {
        idx[dim] -= src0_nedim;
        bits = src1.Load((offset_src1 + idx[0] * stride_src10 + idx[1] * stride_src11 + idx[2] * stride_src12 + idx[3] * stride_src13) * 4);
    }
    dst.Store(i_dst * 4, bits);
}
