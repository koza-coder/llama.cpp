#include "common.hlsli"

// ARANGE (f32): dst[i] = start + step * i over a contiguous destination

RWByteAddressBuffer dst : register(u0);

cbuffer Params : register(b0) {
    uint  offset_dst;
    uint  ne;
    float start;
    float step;
    uint  nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    STORE_F32(dst, offset_dst + i, start + step * (float) i);
}
