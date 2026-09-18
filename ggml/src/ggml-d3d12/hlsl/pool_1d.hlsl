#include "common.hlsli"

// POOL_1D (f32 or f16 in, f32 out): window of k taps, stride s, padding p, along the row only.
// AVG divides by the number of taps that actually landed inside the row, not by k, and MAX over an
// entirely out-of-bounds window keeps -FLT_MAX. Both match ggml_compute_forward_pool_1d_ksp.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src1;
    uint stride_dst1;
    uint iw;
    uint ow;
    uint k;
    int  s;
    int  p;
    uint is_max;
    uint n_rows;
    uint nwg_x;
};

#ifdef SRC_F16
#define LOAD_SRC(i) LOAD_F16(src, (i))
#else
#define LOAD_SRC(i) LOAD_F32(src, (i))
#endif

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint r = gid.y * nwg_x + gid.x;
    if (r >= n_rows) {
        return;
    }
    const uint src_row = offset_src + r * stride_src1;
    const uint dst_row = offset_dst + r * stride_dst1;

    for (uint o = gtid.x; o < ow; o += WG_SIZE) {
        // -FLT_MAX, the CPU's MAX seed; note this is the largest finite negative, not -inf
        float res   = is_max != 0u ? -3.402823466e+38f : 0.0f;
        uint  count = 0;
        const int base = (int) o * s - p;
        for (uint ki = 0; ki < k; ki++) {
            const int j = base + (int) ki;
            if (j < 0 || j >= (int) iw) {
                continue;
            }
            const float v = LOAD_SRC(src_row + (uint) j);
            if (is_max != 0u) {
                res = max(v, res);
            } else {
                res += v;
            }
            count++;
        }
        if (is_max == 0u) {
            res = count > 0u ? res / (float) count : 0.0f;
        }
        STORE_F32(dst, dst_row + o, res);
    }
}
