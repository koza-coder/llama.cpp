#include "common.hlsli"

// DIAG_MASK_INF / DIAG_MASK_ZERO (f32): dst = src, with every element right of the diagonal
// shifted by n_past replaced by mask_value. Both tensors are contiguous, so one flat index
// over dst is enough and the element may be written in place.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint  offset_src;
    uint  offset_dst;
    uint  ne;
    uint  ne0;
    uint  ne1;
    uint  n_past;
    float mask_value;
    uint  nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i0 = i % ne0;
    const uint i1 = (i / ne0) % ne1;
    // the CPU walks i from n_past upward and masks i > n_past + i1; below n_past nothing matches
    const bool masked = i0 > n_past + i1;
    STORE_F32(dst, offset_dst + i, masked ? mask_value : LOAD_F32(src, offset_src + i));
}
