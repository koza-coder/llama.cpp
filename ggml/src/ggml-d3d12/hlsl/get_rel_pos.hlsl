#include "common.hlsli"

// GET_REL_POS (f16): dst[i2, i1, i0] = src[((ne1 - i1 - 1) + i2) * src_ne0 + i0].
// Both sides are contiguous f16; this is the SAM relative position lookup.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint src_ne0;
    uint ne0;
    uint ne1;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i0 = i % ne0;
    const uint i1 = (i / ne0) % ne1;
    const uint i2 = i / (ne0 * ne1);
    const uint pos = (ne1 - i1 - 1u) + i2;
    STORE_F16(dst, offset_dst + i, LOAD_F16(src, offset_src + pos * src_ne0 + i0));
}
