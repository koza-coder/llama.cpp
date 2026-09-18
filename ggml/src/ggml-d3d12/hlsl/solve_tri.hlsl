#include "common.hlsli"

// SOLVE_TRI (f32): forward substitution for a lower triangular A, one right-hand-side column per
// thread. The substitution is sequential in the row index, but every column is independent, so a
// thread owns one (batch, column) pair and keeps its own dependency chain. X is read back as it is
// written, exactly as on the CPU.

RWByteAddressBuffer srcA : register(u0);
RWByteAddressBuffer srcB : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_a;
    uint offset_b;
    uint offset_x;
    uint stride_a2;
    uint stride_a3;
    uint stride_b2;
    uint stride_b3;
    uint stride_x2;
    uint stride_x3;
    uint n;        // A is n x n
    uint k;        // right-hand-side columns
    uint ne02;
    uint n_jobs;   // ne02 * ne03 * k
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint job = flat_index(gid, nwg_x);
    if (job >= n_jobs) {
        return;
    }
    const uint i03 = job / (ne02 * k);
    const uint rem = job % (ne02 * k);
    const uint i02 = rem / k;
    const uint col = rem % k;

    const uint a_base = offset_a + i02 * stride_a2 + i03 * stride_a3;
    const uint b_base = offset_b + i02 * stride_b2 + i03 * stride_b3;
    const uint x_base = offset_x + i02 * stride_x2 + i03 * stride_x3;

    for (uint i00 = 0; i00 < n; i00++) {
        float sum = 0.0f;
        for (uint t = 0; t < i00; t++) {
            sum += LOAD_F32(srcA, a_base + i00 * n + t) * LOAD_F32(dst, x_base + t * k + col);
        }
        const float diag = LOAD_F32(srcA, a_base + i00 * n + i00);
        STORE_F32(dst, x_base + i00 * k + col, (LOAD_F32(srcB, b_base + i00 * k + col) - sum) / diag);
    }
}
