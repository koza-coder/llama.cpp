// Dequantization of one src0 row, shared by the matrix-vector kernel and by get_rows of quantized types.
// The includer defines: SRC0_<TYPE>, TPR, MAX_COLS and the ACC(value, element_index) macro, and provides the
// row length k. Values are produced strided by lane: sub-block or block number lane, lane + TPR, ...
#include "iq_tables.hlsli"

static const float KVALUES_MXFP4[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

// e8m0 scale, halved to match the doubled e2m1 table (ggml_e8m0_to_fp32_half)
float e8m0_to_f32_half(uint e) {
    return asfloat(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
}

static const float KVALUES_IQ4NL[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
// partial dot products of one src0 row against the active columns of src1, strided by lane
void dot_row(RWByteAddressBuffer src0, uint src0_base, uint lane, uint ncols, uint src1_base[MAX_COLS],
             inout float acc[MAX_COLS]) {
#if defined(SRC0_F32)
    for (uint i = lane * 4; i < k; i += TPR * 4) {
        const uint4 w = src0.Load4((src0_base + i) * 4);
        ACC(asfloat(w.x), i);
        ACC(asfloat(w.y), i + 1);
        ACC(asfloat(w.z), i + 2);
        ACC(asfloat(w.w), i + 3);
    }
#elif defined(SRC0_F16)
    for (uint i = lane * 4; i < k; i += TPR * 4) {
        uint w0, w1;
        LOAD_U32_UNALIGNED(src0, (src0_base + i) * 2, w0);
        LOAD_U32_UNALIGNED(src0, (src0_base + i) * 2 + 4, w1);
        ACC(f16tof32(w0 & 0xFFFFu), i);
        ACC(f16tof32(w0 >> 16), i + 1);
        ACC(f16tof32(w1 & 0xFFFFu), i + 2);
        ACC(f16tof32(w1 >> 16), i + 3);
    }
#elif defined(SRC0_Q4_0)
    // block: f16 d, 16 bytes of nibbles; low nibbles are elements 0..15, high nibbles 16..31
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 18;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC(((float) (byte & 0xFu) - 8.0f) * d, blk * 32 + j * 4 + b);
                ACC(((float) (byte >> 4) - 8.0f) * d, blk * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q4_1)
    // block: f16 d, f16 m, 16 bytes of nibbles; value = q * d + m
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 20;
        uint w;
        LOAD_U32_UNALIGNED(src0, base, w);
        const float d = f16tof32(w & 0xFFFFu);
        const float m = f16tof32(w >> 16);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 4 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC((float) (byte & 0xFu) * d + m, blk * 32 + j * 4 + b);
                ACC((float) (byte >> 4) * d + m, blk * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q5_0) || defined(SRC0_Q5_1)
    // Q5_0 block: f16 d, u32 qh, 16 bytes of nibbles; value = (q | bit << 4) - 16, times d
    // Q5_1 block: f16 d, f16 m, u32 qh, 16 bytes of nibbles; value = (q | bit << 4) * d + m
#if defined(SRC0_Q5_0)
    const uint bsize = 22;
    const uint hoff  = 2;
#else
    const uint bsize = 24;
    const uint hoff  = 4;
#endif
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * bsize;
        uint w, qh;
        LOAD_U32_UNALIGNED(src0, base, w);
        LOAD_U32_UNALIGNED(src0, base + hoff, qh);
        const float d = f16tof32(w & 0xFFFFu);
#if defined(SRC0_Q5_0)
        const float m = -16.0f * d;
#else
        const float m = f16tof32(w >> 16);
#endif
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + hoff + 4 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                const uint e    = j * 4 + b;
                ACC((float) ((byte & 0xFu) | (((qh >> e) & 1u) << 4)) * d + m, blk * 32 + e);
                ACC((float) ((byte >> 4) | (((qh >> (e + 16)) & 1u) << 4)) * d + m, blk * 32 + 16 + e);
            }
        }
    }
#elif defined(SRC0_BF16)
    for (uint i = lane * 4; i < k; i += TPR * 4) {
        uint w0, w1;
        LOAD_U32_UNALIGNED(src0, (src0_base + i) * 2, w0);
        LOAD_U32_UNALIGNED(src0, (src0_base + i) * 2 + 4, w1);
        ACC(asfloat(w0 << 16), i);
        ACC(asfloat(w0 & 0xFFFF0000u), i + 1);
        ACC(asfloat(w1 << 16), i + 2);
        ACC(asfloat(w1 & 0xFFFF0000u), i + 3);
    }
#elif defined(SRC0_MXFP4)
    // block: e8m0 scale byte, 16 bytes of nibbles indexing the e2m1 table
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 17;
        uint ebits;
        LOAD_U16_UNALIGNED(src0, base & ~1u, ebits);
        const float d = e8m0_to_f32_half((base & 1u) != 0 ? (ebits >> 8) : (ebits & 0xFFu));
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 1 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC(KVALUES_MXFP4[byte & 0xFu] * d, blk * 32 + j * 4 + b);
                ACC(KVALUES_MXFP4[byte >> 4] * d, blk * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_IQ4_NL)
    // block: f16 d, 16 bytes of nibbles mapped through the non-linear table
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 18;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC(KVALUES_IQ4NL[byte & 0xFu] * d, blk * 32 + j * 4 + b);
                ACC(KVALUES_IQ4NL[byte >> 4] * d, blk * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q8_0)
    // block: f16 d, 32 int8
    for (uint blk = lane; blk < k / 32; blk += TPR) {
        const uint base = (src0_base + blk) * 34;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                ACC((float) sbyte_of(q, b) * d, blk * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q4_K)
    // super-block of 256: f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles.
    // sub-block s (32 values): pair s/2 uses bytes 16 + 32*(s/2), low nibbles for even s, high for odd s.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 144;
        uint w;
        LOAD_U32_UNALIGNED(src0, base, w);
        const float d    = f16tof32(w & 0xFFFFu);
        const float dmin = f16tof32(w >> 16);
        uint sc0, sc1, sc2;
        LOAD_U32_UNALIGNED(src0, base + 4, sc0);
        LOAD_U32_UNALIGNED(src0, base + 8, sc1);
        LOAD_U32_UNALIGNED(src0, base + 12, sc2);
        uint sc, mn;
        if (s < 4) {
            sc = byte_of(sc0, s) & 63u;
            mn = byte_of(sc1, s) & 63u;
        } else {
            sc = (byte_of(sc2, s - 4) & 0xFu) | ((byte_of(sc0, s - 4) >> 6) << 4);
            mn = (byte_of(sc2, s - 4) >> 4) | ((byte_of(sc1, s - 4) >> 6) << 4);
        }
        const float dl = d * (float) sc;
        const float ml = dmin * (float) mn;
        const uint  shift = (s & 1u) * 4u;
        const uint  qbase = base + 16 + 32 * (s / 2);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const float v = dl * (float) ((byte_of(q, b) >> shift) & 0xFu) - ml;
                ACC(v, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q5_K)
    // super-block of 256 (176 bytes): f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 32 bytes qh, 128 bytes ql.
    // like Q4_K with the nibbles at 48; bit s of qh[l] adds 16 to value l of sub-block s.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 176;
        uint w;
        LOAD_U32_UNALIGNED(src0, base, w);
        const float d    = f16tof32(w & 0xFFFFu);
        const float dmin = f16tof32(w >> 16);
        uint sc0, sc1, sc2;
        LOAD_U32_UNALIGNED(src0, base + 4, sc0);
        LOAD_U32_UNALIGNED(src0, base + 8, sc1);
        LOAD_U32_UNALIGNED(src0, base + 12, sc2);
        uint sc, mn;
        if (s < 4) {
            sc = byte_of(sc0, s) & 63u;
            mn = byte_of(sc1, s) & 63u;
        } else {
            sc = (byte_of(sc2, s - 4) & 0xFu) | ((byte_of(sc0, s - 4) >> 6) << 4);
            mn = (byte_of(sc2, s - 4) >> 4) | ((byte_of(sc1, s - 4) >> 6) << 4);
        }
        const float dl = d * (float) sc;
        const float ml = dmin * (float) mn;
        const uint  shift = (s & 1u) * 4u;
        const uint  qbase = base + 48 + 32 * (s / 2);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q, h;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            LOAD_U32_UNALIGNED(src0, base + 16 + 4 * j, h);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint v5 = ((byte_of(q, b) >> shift) & 0xFu) | (((byte_of(h, b) >> s) & 1u) << 4);
                ACC(dl * (float) v5 - ml, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q2_K)
    // super-block of 256 (84 bytes): 16 bytes of 4-bit scale/min pairs, 64 bytes of 2-bit quants, f16 d, f16 dmin.
    // sub-block s: values e < 16 use scale byte 2s, e >= 16 use 2s + 1; quant byte 16 + 32 * (s / 4) + e, shift 2 * (s % 4).
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 84;
        uint w, scw;
        LOAD_U32_UNALIGNED(src0, base + 80, w);
        const float d    = f16tof32(w & 0xFFFFu);
        const float dmin = f16tof32(w >> 16);
        LOAD_U16_UNALIGNED(src0, base + 2 * s, scw);
        const float dl0 = d * (float) (scw & 0xFu);
        const float ml0 = dmin * (float) ((scw >> 4) & 0xFu);
        const float dl1 = d * (float) ((scw >> 8) & 0xFu);
        const float ml1 = dmin * (float) (scw >> 12);
        const uint  shift = 2u * (s % 4u);
        const uint  qbase = base + 16 + 32 * (s / 4);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            const float dl = j < 4 ? dl0 : dl1;
            const float ml = j < 4 ? ml0 : ml1;
            [unroll] for (uint b = 0; b < 4; b++) {
                ACC(dl * (float) ((byte_of(q, b) >> shift) & 3u) - ml, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_Q3_K)
    // super-block of 256 (110 bytes): 32 bytes hmask, 64 bytes of 2-bit quants, 12 bytes of 6-bit scales, f16 d.
    // sub-block s: scale index 2s (e < 16) or 2s + 1; quant byte 32 + 32 * (s / 4) + e, shift 2 * (s % 4);
    // value = scale * (q - (bit s of hmask[e] ? 0 : 4))
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 110;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base + 108, dbits);
        const float d = f16tof32(dbits);
        float dls[2];
        [unroll] for (uint h = 0; h < 2; h++) {
            const uint is = 2 * s + h;
            // single bytes at odd offsets: LOAD_U16_UNALIGNED only handles even addresses
            uint lo, hi;
            if (is < 8) {
                LOAD_U32_UNALIGNED(src0, base + 96 + is, lo);
                lo = lo & 0xFu;
            } else {
                LOAD_U32_UNALIGNED(src0, base + 96 + is - 8, lo);
                lo = (lo >> 4) & 0xFu;
            }
            LOAD_U32_UNALIGNED(src0, base + 104 + is % 4, hi);
            hi = (hi >> (2 * (is / 4))) & 3u;
            dls[h] = d * ((float) (lo | (hi << 4)) - 32.0f);
        }
        const uint shift = 2u * (s % 4u);
        const uint qbase = base + 32 + 32 * (s / 4);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q, hm;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            LOAD_U32_UNALIGNED(src0, base + 4 * j, hm);
            const float dl = dls[j < 4 ? 0 : 1];
            [unroll] for (uint b = 0; b < 4; b++) {
                const int qv = (int) ((byte_of(q, b) >> shift) & 3u) - (((byte_of(hm, b) >> s) & 1u) != 0 ? 0 : 4);
                ACC(dl * (float) qv, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_IQ4_XS)
    // super-block of 256 (136 bytes): f16 d, u16 scales_h, 4 bytes scales_l, 128 bytes of table nibbles.
    // sub-block s: scale = (nibble s of scales_l | 2 bits s of scales_h << 4) - 32; nibble bytes 8 + 16 s.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 136;
        uint w, sl;
        LOAD_U32_UNALIGNED(src0, base, w);
        LOAD_U32_UNALIGNED(src0, base + 4, sl);
        const float d  = f16tof32(w & 0xFFFFu);
        const uint  ls = ((byte_of(sl, s / 2) >> (4 * (s % 2))) & 0xFu) | ((((w >> 16) >> (2 * s)) & 3u) << 4);
        const float dl = d * ((float) ls - 32.0f);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 8 + 16 * s + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                ACC(KVALUES_IQ4NL[byte & 0xFu] * dl, blk * 256 + s * 32 + j * 4 + b);
                ACC(KVALUES_IQ4NL[byte >> 4] * dl, blk * 256 + s * 32 + 16 + j * 4 + b);
            }
        }
    }
#elif defined(SRC0_IQ3_S)
    // super-block of 256 (110 bytes): f16 d, 64 bytes qs, 8 bytes qh, 32 bytes signs, 4 bytes scales.
    // sub-block s: scale d * (1 + 2 * nibble s of scales), 4 groups l of 8 values; group l uses grid entries
    // qs[8s + 2l] (values 0..3) and qs[8s + 2l + 1] (values 4..7), 9th index bits 2l and 2l+1 of qh[s];
    // value j of group l is negated when bit j of signs[4s + l] is set
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 110;
        uint dbits, sc, qh;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U32_UNALIGNED(src0, base + 106 + s / 2, sc);
        LOAD_U32_UNALIGNED(src0, base + 66 + s, qh);
        const float db = f16tof32(dbits) * (float) (1u + 2u * ((sc >> (4u * (s & 1u))) & 0xFu));
        [unroll] for (uint l = 0; l < 4; l++) {
            uint q, sg;
            LOAD_U32_UNALIGNED(src0, base + 2 + 8 * s + 2 * l, q);
            LOAD_U32_UNALIGNED(src0, base + 74 + 4 * s + l, sg);
            const uint g1 = IQ3S_GRID[(q & 0xFFu) | (((qh >> (2u * l)) & 1u) << 8)];
            const uint g2 = IQ3S_GRID[((q >> 8) & 0xFFu) | (((qh >> (2u * l + 1u)) & 1u) << 8)];
            [unroll] for (uint j = 0; j < 8; j++) {
                const uint  gv  = j < 4 ? byte_of(g1, j) : byte_of(g2, j - 4);
                const float sgn = ((sg >> j) & 1u) != 0 ? -1.0f : 1.0f;
                ACC(db * (float) gv * sgn, blk * 256 + s * 32 + l * 8 + j);
            }
        }
    }
#elif defined(SRC0_IQ2_S)
    // super-block of 256 (82 bytes): f16 d, 64 bytes qs (32 grid indices, then 32 sign bytes), 8 bytes qh,
    // 8 bytes scales. sub-block s: 4 groups l of 8 values from grid entry qs[4s + l] | bits 2l..2l+1 of qh[s] << 8,
    // scale d * (0.5 + nibble) * 0.25 with the low nibble for l < 2; signs in qs[32 + 4s + l]
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint base = (src0_base + blk) * 82;
        uint dbits, sc, qh;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        LOAD_U32_UNALIGNED(src0, base + 74 + s, sc);
        LOAD_U32_UNALIGNED(src0, base + 66 + s, qh);
        const float d = f16tof32(dbits);
        [unroll] for (uint l = 0; l < 4; l++) {
            uint q, sg;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * s + l, q);
            LOAD_U32_UNALIGNED(src0, base + 34 + 4 * s + l, sg);
            const uint  gi = (q & 0xFFu) | (((qh >> (2u * l)) & 3u) << 8);
            const float dl = d * (0.5f + (float) ((sc >> (l < 2 ? 0u : 4u)) & 0xFu)) * 0.25f;
            [unroll] for (uint j = 0; j < 8; j++) {
                const uint  gv  = j < 4 ? byte_of(IQ2S_GRID_LO[gi], j) : byte_of(IQ2S_GRID_HI[gi], j - 4);
                const float sgn = ((sg >> j) & 1u) != 0 ? -1.0f : 1.0f;
                ACC(dl * (float) gv * sgn, blk * 256 + s * 32 + l * 8 + j);
            }
        }
    }
#elif defined(SRC0_Q6_K)
    // super-block of 256: 128 bytes ql, 64 bytes qh, 16 int8 scales, f16 d.
    // sub-block s: half h = s/4 (128 values), t = s%4 selects the quarter inside the half.
    for (uint sb = lane; sb < k / 32; sb += TPR) {
        const uint blk  = sb / 8;
        const uint s    = sb % 8;
        const uint h    = s / 4;
        const uint t    = s % 4;
        const uint base = (src0_base + blk) * 210;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base + 208, dbits);
        const float d = f16tof32(dbits);
        const uint  ql_base = base + 64 * h + 32 * (t & 1u);
        const uint  qh_base = base + 128 + 32 * h;
        const uint  sc_base = base + 192 + 8 * h + 2 * t;
        const uint  lshift  = (t >> 1) * 4u;
        const uint  hshift  = t * 2u;
        uint scw;
        LOAD_U32_UNALIGNED(src0, sc_base, scw);
        const float d0 = d * (float) sbyte_of(scw, 0);
        const float d1 = d * (float) sbyte_of(scw, 1);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint ql, qh;
            LOAD_U32_UNALIGNED(src0, ql_base + 4 * j, ql);
            LOAD_U32_UNALIGNED(src0, qh_base + 4 * j, qh);
            const float dsc = (j < 4) ? d0 : d1;
            [unroll] for (uint b = 0; b < 4; b++) {
                const int q = (int) (((byte_of(ql, b) >> lshift) & 0xFu) | (((byte_of(qh, b) >> hshift) & 3u) << 4)) - 32;
                ACC(dsc * (float) q, blk * 256 + s * 32 + j * 4 + b);
            }
        }
    }
#endif
}

// copy the parameters of matrix M into the working set
#define SELECT_MAT(M) { \
    mrows = m_##M; o_src0 = offset_src0_##M; o_dst = offset_dst_##M; \
    s01 = stride_01_##M; s02 = stride_02_##M; s03 = stride_03_##M; \
    add_flag = add_flag_##M; o_add = offset_add_##M; \
    add_ne1 = add_ne1_##M; add_ne2 = add_ne2_##M; add_ne3 = add_ne3_##M; \
    add_s1 = add_s1_##M; add_s2 = add_s2_##M; add_s3 = add_s3_##M; \
}

