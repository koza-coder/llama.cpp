#include "common.hlsli"

// TIMESTEP_EMBEDDING (f32): row i of dst holds cos/sin of src[i] scaled by a geometric frequency
// ladder. half = dim / 2; an odd dim leaves the last column zero.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint  offset_src;
    uint  offset_dst;
    uint  stride_dst1;
    uint  ne00;
    uint  half_dim;
    uint  dim;
    float neg_log_max_period;
    uint  nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint i = gid.y * nwg_x + gid.x;
    if (i >= ne00) {
        return;
    }
    const uint  row      = offset_dst + i * stride_dst1;
    const float timestep = LOAD_F32(src, offset_src + i);

    for (uint j = gtid.x; j < half_dim; j += WG_SIZE) {
        const float freq = exp(neg_log_max_period * (float) j / (float) half_dim);
        const float arg  = timestep * freq;
        STORE_F32(dst, row + j, cos(arg));
        STORE_F32(dst, row + j + half_dim, sin(arg));
    }
    if ((dim & 1u) != 0u && gtid.x == 0) {
        STORE_F32(dst, row + 2u * half_dim, 0.0f);
    }
}
