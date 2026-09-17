#include "common.hlsli"

// defines: TYPE_F32 or TYPE_F16 (+USE_16BIT), FF_FUNC when frequency factors are present

#if defined(TYPE_F16)
#define LOAD(buf, i)     LOAD_F16(buf, i)
#define STORE(buf, i, v) STORE_F16(buf, i, v)
#else
#define LOAD(buf, i)     LOAD_F32(buf, i)
#define STORE(buf, i, v) STORE_F32(buf, i, v)
#endif

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);   // positions, i32
RWByteAddressBuffer src2 : register(u2);   // frequency factors, f32
RWByteAddressBuffer dst  : register(u3);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_src2;
    uint offset_dst;

    uint stride_src01;
    uint stride_src02;
    uint stride_src03;

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint n_threads;
    uint ne0;
    uint ne1;
    uint ne2;

    uint n_dims;
    uint mode;
    float theta_scale;
    float attn_factor;
    float freq_scale;
    float ext_factor;
    float corr_dim0;
    float corr_dim1;
    uint sections0;
    uint sections1;
    uint sections2;
    uint sections3;
    uint n_offs;

    uint nwg_x;
};

float freq_factor(uint i) {
#ifdef FF_FUNC
    return LOAD_F32(src2, offset_src2 + i / 2);
#else
    return 1.0f;
#endif
}

float rope_yarn_ramp(float low, float high, uint i) {
    const float y = ((float) (i / 2) - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

float2 rope_yarn(float theta_extrap, uint i) {
    float mscale = attn_factor;
    float theta  = freq_scale * theta_extrap;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(corr_dim0, corr_dim1, i) * ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    return float2(cos(theta) * mscale, sin(theta) * mscale);
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint t = flat_index(gid, nwg_x);
    if (t >= n_threads) {
        return;
    }
    const bool is_neox   = (mode & 2u) != 0;
    const bool is_mrope  = (mode & 8u) != 0;
    const bool is_imrope = mode == 40u;
    const bool is_vision = mode == 24u;

    uint i = t * 2;
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;

    const uint src_row = offset_src0 + i3 * stride_src03 + i2 * stride_src02 + i1 * stride_src01;
    const uint dst_row = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;

    if ((i0 < n_offs || i0 >= n_offs + n_dims) && !is_vision) {
        STORE(dst, dst_row + i0, LOAD(src0, src_row + i0));
        STORE(dst, dst_row + i0 + 1, LOAD(src0, src_row + i0 + 1));
        return;
    }

    const uint iw = i0 - n_offs;
    uint theta_base_mult = 0;
    uint theta_scale_pwr = iw / 2;
    if (is_mrope) {
        const uint sect_dims = sections0 + sections1 + sections2 + sections3;
        const uint sec_w     = sections1 + sections0;
        const uint sec_e     = sections2 + sec_w;
        const uint sector    = (iw / 2) % sect_dims;
        if (is_imrope) {
            if (sector % 3 == 1 && sector < 3 * sections1) {
                theta_base_mult = 1;
            } else if (sector % 3 == 2 && sector < 3 * sections2) {
                theta_base_mult = 2;
            } else if (sector % 3 == 0 && sector < 3 * sections0) {
                theta_base_mult = 0;
            } else {
                theta_base_mult = 3;
            }
        } else {
            if (sector >= sections0 && sector < sec_w) {
                theta_base_mult = 1;
                if (is_vision) { theta_scale_pwr = sector - sections0; }
            } else if (sector >= sec_w && sector < sec_e) {
                theta_base_mult = 2;
                if (is_vision) { theta_scale_pwr = sector - sec_w; }
            } else if (sector >= sec_e) {
                if (is_vision) { theta_scale_pwr = (iw / 2) % sec_e; }
                theta_base_mult = 3;
            } else if (is_vision) {
                theta_scale_pwr = sector;
            }
        }
    }
    const float pos        = (float) LOAD_I32(src1, offset_src1 + i2 + ne2 * theta_base_mult);
    const float theta_base = pos * pow(theta_scale, (float) theta_scale_pwr);
    const float2 thetas    = rope_yarn(theta_base / freq_factor(iw), iw);

    const bool div2 = is_neox || is_mrope || is_vision;
    const uint pair_base = div2 ? (i0 / 2 + n_offs / 2) : i0;
    const uint pair_off  = is_vision ? n_dims : ((is_neox || is_mrope) ? n_dims / 2 : 1);

    const float x0 = LOAD(src0, src_row + pair_base);
    const float x1 = LOAD(src0, src_row + pair_base + pair_off);
    STORE(dst, dst_row + pair_base, x0 * thetas.x - x1 * thetas.y);
    STORE(dst, dst_row + pair_base + pair_off, x0 * thetas.y + x1 * thetas.x);
}
