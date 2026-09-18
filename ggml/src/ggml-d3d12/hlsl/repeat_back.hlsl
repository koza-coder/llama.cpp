#include "common.hlsli"

// REPEAT_BACK (f32): the adjoint of REPEAT, so it sums the tiles that REPEAT would have produced.
//   dst[k0,k1,k2,k3] = sum over (i0,i1,i2,i3) of
//       src0[i0*ne0 + k0, i1*ne1 + k1, i2*ne2 + k2, i3*ne3 + k3]
// The CPU zeroes dst and accumulates tile by tile; read as a gather this is one thread per
// destination element with no zeroing pass. nr0..nr3 are the repeat counts, integral by
// construction (ggml_can_repeat). Both tensors have packed rows (nb0 == nb00 == 4).

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_s1;   // src0 nb[1] / 4
    uint stride_s2;
    uint stride_s3;
    uint stride_d1;   // dst nb[1] / 4
    uint stride_d2;
    uint stride_d3;
    uint ne0;         // dst extents
    uint ne1;
    uint ne2;
    uint ne3;
    uint nr0;         // repeat counts, src0->ne[k] / dst->ne[k]
    uint nr1;
    uint nr2;
    uint nr3;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint k3 = i / (ne2 * ne1 * ne0);
    uint rem      = i % (ne2 * ne1 * ne0);
    const uint k2 = rem / (ne1 * ne0);
    rem           = rem % (ne1 * ne0);
    const uint k1 = rem / ne0;
    const uint k0 = rem % ne0;

    float sum = 0.0f;
    for (uint i3 = 0; i3 < nr3; i3++) {
        const uint s3 = (i3 * ne3 + k3) * stride_s3;
        for (uint i2 = 0; i2 < nr2; i2++) {
            const uint s2 = s3 + (i2 * ne2 + k2) * stride_s2;
            for (uint i1 = 0; i1 < nr1; i1++) {
                const uint s1 = s2 + (i1 * ne1 + k1) * stride_s1;
                for (uint i0 = 0; i0 < nr0; i0++) {
                    sum += LOAD_F32(src, offset_src + s1 + i0 * ne0 + k0);
                }
            }
        }
    }
    STORE_F32(dst, offset_dst + k3 * stride_d3 + k2 * stride_d2 + k1 * stride_d1 + k0, sum);
}
