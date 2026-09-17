#include "common.hlsli"

// one workgroup per row; defines: HAS_MASK with MASK_F32 or MASK_F16, HAS_SINK

RWByteAddressBuffer src   : register(u0);
RWByteAddressBuffer mask  : register(u1);
RWByteAddressBuffer sinks : register(u2);
RWByteAddressBuffer dst   : register(u3);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_sinks;
    uint offset_dst;

    uint stride_src01;
    uint stride_src02;
    uint stride_src03;

    uint stride_src11;
    uint stride_src12;
    uint stride_src13;

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint ne0;
    uint ne1;
    uint ne2;

    uint ne12;
    uint ne13;

    float scale;
    float max_bias;
    float n_head_log2;
    float m0;
    float m1;

    uint n_rows;
    uint nwg_x;
};

#define CACHE_SIZE 16
groupshared float scratch[WG_SIZE];

float mask_val(uint i) {
#if defined(HAS_MASK) && defined(MASK_F16)
    uint bits;
    LOAD_U16_UNALIGNED(mask, i * 2, bits);
    return f16tof32(bits);
#elif defined(HAS_MASK)
    return LOAD_F32(mask, i);
#else
    return 0.0f;
#endif
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid = gtid.x;
    uint i = gid.y * nwg_x + gid.x;
    if (i >= n_rows) {
        return;
    }
    const uint i3 = i / (ne2 * ne1);
    i = i % (ne2 * ne1);
    const uint i2 = i / ne1;
    const uint i1 = i % ne1;
    const uint src0_row = offset_src0 + i3 * stride_src03 + i2 * stride_src02 + i1 * stride_src01;
    const uint src1_row = offset_src1 + (i3 % ne13) * stride_src13 + (i2 % ne12) * stride_src12 + i1 * stride_src11;
    const uint dst_row  = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;

    const float head  = (float) i2;
    float slope = 1.0f;
    if (max_bias > 0.0f) {
        slope = head < n_head_log2 ? pow(m0, head + 1.0f) : pow(m1, 2.0f * (head - n_head_log2) + 1.0f);
    }

    float cache[CACHE_SIZE];

#ifdef HAS_SINK
    float max_val = LOAD_F32(sinks, offset_sinks + i2);
#else
    float max_val = -1e30f;
#endif
    for (uint col = tid; col < ne0; col += WG_SIZE) {
        const float val = LOAD_F32(src, src0_row + col) * scale + slope * mask_val(src1_row + col);
        max_val = max(max_val, val);
        if (col < CACHE_SIZE) {
            cache[col] = val;
        }
    }
    scratch[tid] = max_val;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (tid < off) {
            scratch[tid] = max(scratch[tid], scratch[tid + off]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float row_max = scratch[0];
    GroupMemoryBarrierWithGroupSync();

    float sum = 0;
    for (uint c2 = tid; c2 < ne0; c2 += WG_SIZE) {
        const float val = c2 < CACHE_SIZE ? cache[c2] : LOAD_F32(src, src0_row + c2) * scale + slope * mask_val(src1_row + c2);
        const float ex  = exp(val - row_max);
        sum += ex;
        if (c2 < CACHE_SIZE) {
            cache[c2] = ex;
        } else {
            STORE_F32(dst, dst_row + c2, ex);
        }
    }
    scratch[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint off2 = WG_SIZE / 2; off2 > 0; off2 /= 2) {
        if (tid < off2) {
            scratch[tid] += scratch[tid + off2];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float row_sum = scratch[0];
#ifdef HAS_SINK
    row_sum += exp(LOAD_F32(sinks, offset_sinks + i2) - row_max);
#endif
    const float inv = 1.0f / row_sum;
    for (uint c3 = tid; c3 < ne0; c3 += WG_SIZE) {
        const float v = c3 < CACHE_SIZE ? cache[c3] : LOAD_F32(dst, dst_row + c3);
        STORE_F32(dst, dst_row + c3, v * inv);
    }
}
