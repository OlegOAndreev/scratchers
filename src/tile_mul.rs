#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_load_ps, _mm256_setzero_ps,
    _mm256_store_ps,
};

#[inline(always)]
pub fn mul_tile_simd<const TILE_SIZE: usize>(
    m1: &[f32],
    m2: &[f32],
    md: &mut [f32],
    ib: usize,
    jb: usize,
    kb: usize,
    n: usize,
) {
    if TILE_SIZE == 8 {
        mul_tile_simd_impl_8(m1, m2, md, ib, jb, kb, n);
    } else if TILE_SIZE == 16 {
        mul_tile_simd_impl_16(m1, m2, md, ib, jb, kb, n);
    } else if TILE_SIZE == 32 {
        mul_tile_simd_impl_32(m1, m2, md, ib, jb, kb, n);
    } else {
        mul_tile_generic::<TILE_SIZE>(m1, m2, md, ib, jb, kb, n);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub fn mul_tile_simd_impl_8(
    m1: &[f32],
    m2: &[f32],
    md: &mut [f32],
    ib: usize,
    jb: usize,
    kb: usize,
    n: usize,
) {
    unsafe {
        // TODO:
        // 1. Try unrolling h loop.
        // 2. Move m2_k load up.
        // 3. Try using more accumulators and reuse m1 regs.
        // 4. Try unrolling the whole loop (instead of unrolling h loop).
        // 5. Check if packed tile works better.
        // 6. Try writing assembly?

        // Split the tile in two halves: compute the second half after the first one.
        for h in 0..2 {
            let ibh = ib + h * 4;
            let mut accum0 = _mm256_setzero_ps();
            let mut accum1 = _mm256_setzero_ps();
            let mut accum2 = _mm256_setzero_ps();
            let mut accum3 = _mm256_setzero_ps();

            for k in 0..8 {
                let m1_0k = _mm256_broadcast_ss(m1.get_unchecked(ibh * n + kb + k));
                let m1_1k = _mm256_broadcast_ss(m1.get_unchecked((ibh + 1) * n + kb + k));
                let m1_2k = _mm256_broadcast_ss(m1.get_unchecked((ibh + 2) * n + kb + k));
                let m1_3k = _mm256_broadcast_ss(m1.get_unchecked((ibh + 3) * n + kb + k));

                let m2_k = _mm256_load_ps(m2.as_ptr().add((kb + k) * n + jb));

                accum0 = _mm256_fmadd_ps(m1_0k, m2_k, accum0);
                accum1 = _mm256_fmadd_ps(m1_1k, m2_k, accum1);
                accum2 = _mm256_fmadd_ps(m1_2k, m2_k, accum2);
                accum3 = _mm256_fmadd_ps(m1_3k, m2_k, accum3);
            }

            let md_0 = _mm256_load_ps(md.as_mut_ptr().add(ibh * n + jb));
            let md_1 = _mm256_load_ps(md.as_mut_ptr().add((ibh + 1) * n + jb));
            let md_2 = _mm256_load_ps(md.as_mut_ptr().add((ibh + 2) * n + jb));
            let md_3 = _mm256_load_ps(md.as_mut_ptr().add((ibh + 3) * n + jb));
            accum0 = _mm256_add_ps(md_0, accum0);
            accum1 = _mm256_add_ps(md_1, accum1);
            accum2 = _mm256_add_ps(md_2, accum2);
            accum3 = _mm256_add_ps(md_3, accum3);
            _mm256_store_ps(md.as_mut_ptr().add(ibh * n + jb), accum0);
            _mm256_store_ps(md.as_mut_ptr().add((ibh + 1) * n + jb), accum1);
            _mm256_store_ps(md.as_mut_ptr().add((ibh + 2) * n + jb), accum2);
            _mm256_store_ps(md.as_mut_ptr().add((ibh + 3) * n + jb), accum3);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn mul_tile_simd_impl_8(
    m1: &[f32],
    m2: &[f32],
    md: &mut [f32],
    ib: usize,
    jb: usize,
    kb: usize,
    n: usize,
) {
    unsafe {
    }
}

#[inline(always)]
pub fn mul_tile_simd_impl_16(
    m1: &[f32],
    m2: &[f32],
    md: &mut [f32],
    ib: usize,
    jb: usize,
    kb: usize,
    n: usize,
) {
    mul_tile_generic::<16>(m1, m2, md, ib, jb, kb, n);
}

#[inline(always)]
pub fn mul_tile_simd_impl_32(
    m1: &[f32],
    m2: &[f32],
    md: &mut [f32],
    ib: usize,
    jb: usize,
    kb: usize,
    n: usize,
) {
    mul_tile_generic::<32>(m1, m2, md, ib, jb, kb, n);
}

#[inline(always)]
pub fn mul_tile_generic<const TILE_SIZE: usize>(
    m1: &[f32],
    m2: &[f32],
    md: &mut [f32],
    ib: usize,
    jb: usize,
    kb: usize,
    n: usize,
) {
    for i in ib..ib + TILE_SIZE {
        for j in jb..jb + TILE_SIZE {
            let mut acc = 0.0f32;
            for k in kb..kb + TILE_SIZE {
                unsafe {
                    acc += *m1.get_unchecked(i * n + k) * *m2.get_unchecked(k * n + j);
                }
            }
            md[i * n + j] += acc;
        }
    }
}
