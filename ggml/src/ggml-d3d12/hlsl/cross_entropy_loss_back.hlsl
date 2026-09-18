#include "common.hlsli"

// CROSS_ENTROPY_LOSS_BACK (f32), one workgroup per row. src[0] is the scalar gradient of the
// loss, src[1] the forward logits, src[2] the forward labels. Following
// ggml_compute_forward_cross_entropy_loss_back_f32:
//   dst = (softmax(logits) - labels) * grad[0] / n_rows
// Everything but the scalar gradient is contiguous and the same shape as dst.

RWByteAddressBuffer grad_buf : register(u0);
RWByteAddressBuffer s0_buf   : register(u1);
RWByteAddressBuffer s1_buf   : register(u2);
RWByteAddressBuffer dst      : register(u3);

cbuffer Params : register(b0) {
    uint offset_grad;
    uint offset_s0;
    uint offset_s1;
    uint offset_dst;
    uint ne0;
    uint n_rows;
    uint nwg_x;
};

groupshared float scratch[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint row = gid.y * nwg_x + gid.x;
    if (row >= n_rows) {
        return;
    }
    const uint t    = gtid.x;
    const uint base = row * ne0;

    float m = -3.0e38f;
    for (uint c0 = t; c0 < ne0; c0 += WG_SIZE) {
        m = max(m, LOAD_F32(s0_buf, offset_s0 + base + c0));
    }
    scratch[t] = m;
    GroupMemoryBarrierWithGroupSync();
    for (uint o1 = WG_SIZE / 2; o1 > 0; o1 /= 2) {
        if (t < o1) {
            scratch[t] = max(scratch[t], scratch[t + o1]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float row_max = scratch[0];
    GroupMemoryBarrierWithGroupSync();

    float se = 0.0f;
    for (uint c1 = t; c1 < ne0; c1 += WG_SIZE) {
        se += exp(LOAD_F32(s0_buf, offset_s0 + base + c1) - row_max);
    }
    scratch[t] = se;
    GroupMemoryBarrierWithGroupSync();
    for (uint o2 = WG_SIZE / 2; o2 > 0; o2 /= 2) {
        if (t < o2) {
            scratch[t] += scratch[t + o2];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float inv_sum = 1.0f / scratch[0];

    const float d_by_nr = LOAD_F32(grad_buf, offset_grad) / (float) n_rows;

    for (uint c2 = t; c2 < ne0; c2 += WG_SIZE) {
        const float p = exp(LOAD_F32(s0_buf, offset_s0 + base + c2) - row_max) * inv_sum;
        STORE_F32(dst, offset_dst + base + c2, (p - LOAD_F32(s1_buf, offset_s1 + base + c2)) * d_by_nr);
    }
}
