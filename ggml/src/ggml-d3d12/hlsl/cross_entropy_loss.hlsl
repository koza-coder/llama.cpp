#include "common.hlsli"

// CROSS_ENTROPY_LOSS (f32): dst is a single scalar, so this runs as one workgroup that walks
// every row. Following ggml_compute_forward_cross_entropy_loss_f32:
//   per row: lse = max + log(sum(exp(s0 - max)));  acc += sum(s1 * (s0 - lse))
//   dst = -acc / n_rows
// src0 (logits) and src1 (labels) are contiguous and the same shape.

RWByteAddressBuffer s0_buf : register(u0);
RWByteAddressBuffer s1_buf : register(u1);
RWByteAddressBuffer dst    : register(u2);

cbuffer Params : register(b0) {
    uint offset_s0;
    uint offset_s1;
    uint offset_dst;
    uint ne0;
    uint n_rows;
    uint nwg_x;
};

groupshared float scratch[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    const uint t = gtid.x;
    float total = 0.0f;

    for (uint row = 0; row < n_rows; row++) {
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
        const float lse = row_max + log(scratch[0]);
        GroupMemoryBarrierWithGroupSync();

        float acc = 0.0f;
        for (uint c2 = t; c2 < ne0; c2 += WG_SIZE) {
            acc += LOAD_F32(s1_buf, offset_s1 + base + c2) *
                   (LOAD_F32(s0_buf, offset_s0 + base + c2) - lse);
        }
        scratch[t] = acc;
        GroupMemoryBarrierWithGroupSync();
        for (uint o3 = WG_SIZE / 2; o3 > 0; o3 /= 2) {
            if (t < o3) {
                scratch[t] += scratch[t + o3];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        total += scratch[0];
        GroupMemoryBarrierWithGroupSync();
    }

    if (t == 0) {
        STORE_F32(dst, offset_dst, -total / (float) n_rows);
    }
}
