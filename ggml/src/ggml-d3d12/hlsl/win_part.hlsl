#include "common.hlsli"

// WIN_PART / WIN_UNPART (f32): pure index remaps between an image and its w x w window tiling.
// Both sides are contiguous, so one thread per destination element and flat indices are enough.
// WIN_PART pads with zero where a window runs past the image; WIN_UNPART is the exact inverse and
// never reads out of range, because the padded region is simply not visited.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint src_ne0;
    uint src_ne1;
    uint src_ne2;
    uint ne0;
    uint ne1;
    uint ne2;
    uint ne;
    uint nep0;   // WIN_PART: windows across; WIN_UNPART: npx, the same count
    uint w;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint dst_idx = i;
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;

#ifdef UNPART
    const uint ip2 = i2 / w;
    const uint ip1 = i1 / w;
    const uint i02 = i2 % w;
    const uint i01 = i1 % w;
    const uint j   = (ip2 * nep0 + ip1) * src_ne2 * src_ne1 * src_ne0 + i02 * src_ne1 * src_ne0 +
                     i01 * src_ne0 + i0;
    STORE_F32(dst, offset_dst + dst_idx, LOAD_F32(src, offset_src + j));
#else
    const uint py  = i3 / nep0;
    const uint px  = i3 % nep0;
    const uint i02 = py * w + i2;
    const uint i01 = px * w + i1;
    if (i02 >= src_ne2 || i01 >= src_ne1) {
        STORE_F32(dst, offset_dst + dst_idx, 0.0f);
    } else {
        const uint j = i02 * src_ne1 * src_ne0 + i01 * src_ne0 + i0;
        STORE_F32(dst, offset_dst + dst_idx, LOAD_F32(src, offset_src + j));
    }
#endif
}
