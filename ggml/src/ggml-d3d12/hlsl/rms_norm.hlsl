#include "common.hlsli"

// one workgroup per row: dst = src * rsqrt(mean(src^2) + eps), f32 only
// L2_NORM: dst = src / max(sqrt(sum(src^2)), eps) instead
// FUSE_MUL: dst *= wgt, with wgt broadcast over dims 1..3 like ggml_mul (fused RMS_NORM + MUL)

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);
#if defined(FUSE_MUL)
RWByteAddressBuffer wgt : register(u2);
#endif

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;

    uint stride_src1;
    uint stride_src2;
    uint stride_src3;

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint ne0;
    uint ne1;
    uint ne2;
    uint n_rows;

    float eps;

    // fused multiply weight (zeros when not fused)
    uint offset_w;
    uint w_ne1;
    uint w_ne2;
    uint w_ne3;
    uint w_s1;
    uint w_s2;
    uint w_s3;

    uint nwg_x;
};

groupshared float scratch[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint i = gid.y * nwg_x + gid.x;
    if (i >= n_rows) {
        return;
    }
    const uint i3 = i / (ne2 * ne1);
    i = i % (ne2 * ne1);
    const uint i2 = i / ne1;
    const uint i1 = i % ne1;
    const uint src_row = offset_src + i3 * stride_src3 + i2 * stride_src2 + i1 * stride_src1;
    const uint dst_row = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;
#if defined(FUSE_MUL)
    const uint w_row = offset_w + (i3 % w_ne3) * w_s3 + (i2 % w_ne2) * w_s2 + (i1 % w_ne1) * w_s1;
#endif

    float sum = 0;
    for (uint col = gtid.x; col < ne0; col += WG_SIZE) {
        const float v = LOAD_F32(src, src_row + col);
        sum += v * v;
    }
    scratch[gtid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (gtid.x < off) {
            scratch[gtid.x] += scratch[gtid.x + off];
        }
        GroupMemoryBarrierWithGroupSync();
    }
#if defined(L2_NORM)
    const float scale = 1.0f / max(sqrt(scratch[0]), eps);   // L2_NORM: x / max(||x||, eps)
#else
    const float scale = 1.0f / sqrt(scratch[0] / (float) ne0 + eps);
#endif

    for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
        float v = LOAD_F32(src, src_row + c) * scale;
#if defined(FUSE_MUL)
        v *= LOAD_F32(wgt, w_row + c);
#endif
        STORE_F32(dst, dst_row + c, v);
    }
}
