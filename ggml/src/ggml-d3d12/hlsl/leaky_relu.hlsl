#include "common.hlsli"

// LEAKY_RELU (f32): dst = x > 0 ? x : x * negative_slope, one workgroup per row

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint  offset_src;
    uint  offset_dst;
    uint  stride_src1;
    uint  stride_dst1;
    uint  ne0;
    uint  n_rows;
    float negative_slope;
    uint  nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint r = gid.y * nwg_x + gid.x;
    if (r >= n_rows) {
        return;
    }
    const uint src_row = offset_src + r * stride_src1;
    const uint dst_row = offset_dst + r * stride_dst1;
    for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
        const float x = LOAD_F32(src, src_row + c);
        STORE_F32(dst, dst_row + c, x > 0.0f ? x : x * negative_slope);
    }
}
