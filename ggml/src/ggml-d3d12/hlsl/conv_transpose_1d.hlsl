#include "common.hlsli"

// CONV_TRANSPOSE_1D (f32): dst {out_len, Cout} from a kernel {K, Cout, Cin} and an input {L, Cin}.
//
// The CPU scatters: for every input position i10 and tap i00 it adds into dst[i10*s0 + i00]. That
// needs a zeroed dst and two transposed staging copies. Turned around, a destination position o is
// hit by exactly the taps i00 for which o - i00 is a non-negative multiple of s0, so one thread can
// own one output element, gather those taps and need neither the zero pass nor the staging buffers.

RWByteAddressBuffer knl : register(u0);   // src0 {K, Cout, Cin}
RWByteAddressBuffer src : register(u1);   // src1 {L, Cin}
RWByteAddressBuffer dst : register(u2);

cbuffer Params : register(b0) {
    uint offset_knl;
    uint offset_src;
    uint offset_dst;
    uint stride_k1;   // src0 nb[1] / 4, one Cout row
    uint stride_k2;   // src0 nb[2] / 4, one Cin plane
    uint stride_s1;   // src1 nb[1] / 4, one Cin row
    uint stride_d1;   // dst  nb[1] / 4
    uint nk;          // K, taps per filter
    uint cin;
    uint len;         // L, input positions
    uint ne0;         // out_len
    uint s0;          // stride
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint oc = i / ne0;   // output channel, Cout
    const uint o  = i % ne0;

    float sum = 0.0f;
    for (uint i00 = 0; i00 < nk; i00++) {
        if (i00 > o) {
            break;
        }
        const uint rem = o - i00;
        if (rem % s0 != 0u) {
            continue;
        }
        const uint i10 = rem / s0;
        if (i10 >= len) {
            continue;
        }
        for (uint c = 0; c < cin; c++) {
            sum += LOAD_F32(src, offset_src + c * stride_s1 + i10) *
                   LOAD_F32(knl, offset_knl + c * stride_k2 + oc * stride_k1 + i00);
        }
    }
    STORE_F32(dst, offset_dst + oc * stride_d1 + o, sum);
}
