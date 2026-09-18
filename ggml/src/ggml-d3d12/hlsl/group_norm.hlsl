#include "common.hlsli"

// GROUP_NORM (f32): channels are split into n_groups consecutive groups; every group is normalised
// over (channels in group) x ne01 x ne00 for each i03. One workgroup owns one (group, i03) pair and
// makes three passes: mean, then centre-and-variance, then scale. That is the order the CPU uses,
// including writing the centred value out before scaling it in place.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint  offset_src;
    uint  offset_dst;
    uint  stride_src1;
    uint  stride_src2;
    uint  stride_src3;
    uint  stride_dst1;
    uint  stride_dst2;
    uint  stride_dst3;
    uint  ne0;
    uint  ne1;
    uint  n_channels;
    uint  n_groups;
    uint  channels_per_group;
    uint  n_batches;
    float eps;
    uint  nwg_x;
};

groupshared float scratch[WG_SIZE];

float group_reduce(uint tid, float v) {
    scratch[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (tid < off) {
            scratch[tid] += scratch[tid + off];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float total = scratch[0];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint job = gid.y * nwg_x + gid.x;
    if (job >= n_groups * n_batches) {
        return;
    }
    const uint g   = job % n_groups;
    const uint i03 = job / n_groups;

    const uint start = g * channels_per_group;
    uint       end   = start + channels_per_group;
    if (end > n_channels) {
        end = n_channels;
    }
    const uint step  = end - start;
    const uint count = step * ne1 * ne0;

    const uint tid = gtid.x;

    float sum = 0.0f;
    for (uint i = tid; i < count; i += WG_SIZE) {
        const uint i00 = i % ne0;
        const uint i01 = (i / ne0) % ne1;
        const uint i02 = start + i / (ne0 * ne1);
        sum += LOAD_F32(src, offset_src + i03 * stride_src3 + i02 * stride_src2 + i01 * stride_src1 + i00);
    }
    const float mean = group_reduce(tid, sum) / (float) count;

    float sum2 = 0.0f;
    for (uint i = tid; i < count; i += WG_SIZE) {
        const uint i00 = i % ne0;
        const uint i01 = (i / ne0) % ne1;
        const uint i02 = start + i / (ne0 * ne1);
        const float v = LOAD_F32(src, offset_src + i03 * stride_src3 + i02 * stride_src2 + i01 * stride_src1 + i00) - mean;
        STORE_F32(dst, offset_dst + i03 * stride_dst3 + i02 * stride_dst2 + i01 * stride_dst1 + i00, v);
        sum2 += v * v;
    }
    const float variance = group_reduce(tid, sum2) / (float) count;
    const float scale    = 1.0f / sqrt(variance + eps);

    for (uint i = tid; i < count; i += WG_SIZE) {
        const uint i00 = i % ne0;
        const uint i01 = (i / ne0) % ne1;
        const uint i02 = start + i / (ne0 * ne1);
        const uint o   = offset_dst + i03 * stride_dst3 + i02 * stride_dst2 + i01 * stride_dst1 + i00;
        STORE_F32(dst, o, LOAD_F32(dst, o) * scale);
    }
}
