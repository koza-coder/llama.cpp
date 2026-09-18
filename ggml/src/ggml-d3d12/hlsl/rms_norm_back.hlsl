#include "common.hlsli"

// RMS_NORM_BACK (f32), one workgroup per row. src0 is the incoming gradient dz, src1 is the
// forward input x, both the same shape as dst. Following ggml_compute_forward_rms_norm_back_f32:
//   sum_xx  = sum(x*x)               sum_xdz = sum(x*dz)
//   mean_eps = sum_xx/ne0 + eps      sum_eps = sum_xx + eps*ne0
//   dx = (dz + x*(-sum_xdz/sum_eps)) * rsqrt(mean_eps)

RWByteAddressBuffer dz_buf : register(u0);
RWByteAddressBuffer x_buf  : register(u1);
RWByteAddressBuffer dst    : register(u2);

cbuffer Params : register(b0) {
    uint offset_dz;
    uint offset_x;
    uint offset_dst;

    uint stride_dz1;
    uint stride_dz2;
    uint stride_dz3;

    uint stride_x1;
    uint stride_x2;
    uint stride_x3;

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint ne0;
    uint ne1;
    uint ne2;
    uint n_rows;

    float eps;

    uint nwg_x;
};

groupshared float s_xx[WG_SIZE];
groupshared float s_xdz[WG_SIZE];

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

    const uint dz_row  = offset_dz  + i3 * stride_dz3  + i2 * stride_dz2  + i1 * stride_dz1;
    const uint x_row   = offset_x   + i3 * stride_x3   + i2 * stride_x2   + i1 * stride_x1;
    const uint dst_row = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;

    float sum_xx  = 0.0f;
    float sum_xdz = 0.0f;
    for (uint col = gtid.x; col < ne0; col += WG_SIZE) {
        const float x  = LOAD_F32(x_buf, x_row + col);
        const float dz = LOAD_F32(dz_buf, dz_row + col);
        sum_xx  += x * x;
        sum_xdz += x * dz;
    }
    s_xx[gtid.x]  = sum_xx;
    s_xdz[gtid.x] = sum_xdz;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (gtid.x < off) {
            s_xx[gtid.x]  += s_xx[gtid.x + off];
            s_xdz[gtid.x] += s_xdz[gtid.x + off];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const float mean_eps = s_xx[0] / (float) ne0 + eps;
    const float sum_eps  = s_xx[0] + eps * (float) ne0;
    const float rrms     = 1.0f / sqrt(mean_eps);
    const float scale_x  = -s_xdz[0] / sum_eps;

    for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
        const float x  = LOAD_F32(x_buf, x_row + c);
        const float dz = LOAD_F32(dz_buf, dz_row + c);
        STORE_F32(dst, dst_row + c, (dz + x * scale_x) * rrms);
    }
}
