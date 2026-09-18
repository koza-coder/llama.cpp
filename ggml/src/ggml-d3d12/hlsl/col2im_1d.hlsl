#include "common.hlsli"

// COL2IM_1D (f32): scatter-add columns [K*OC, T_in] back to a signal [T_out, OC].
// The CPU already gathers rather than scatters, so this is a direct transcription: one thread owns
// one output sample and sums the at most ceil(K/s) columns that overlap it.

RWByteAddressBuffer src : register(u0);   // [K*OC, T_in]
RWByteAddressBuffer dst : register(u1);   // [T_out, OC]

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint k_oc;     // K * OC, the column height
    uint t_in;
    uint kk;       // K, taps per output channel
    uint t_out;
    int  s0;
    int  p0;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint oc = i / t_out;
    const uint t  = i % t_out;

    const int t_abs = (int) t + p0;   // position in the uncropped signal
    // the first column whose window still reaches t_abs, and the last one that starts at or before it
    int lo = (t_abs - (int) kk + 1 + s0 - 1) / s0;
    if (lo < 0) {
        lo = 0;
    }
    int hi = t_abs / s0;
    if (hi >= (int) t_in) {
        hi = (int) t_in - 1;
    }

    float sum = 0.0f;
    for (int c = lo; c <= hi; c++) {
        const int k = t_abs - c * s0;
        if (k >= 0 && k < (int) kk) {
            sum += LOAD_F32(src, offset_src + (oc * kk + (uint) k) + (uint) c * k_oc);
        }
    }
    STORE_F32(dst, offset_dst + i, sum);
}
