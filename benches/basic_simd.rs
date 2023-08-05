// Benchmark basic SIMD operations.

use std::arch::asm;
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
    _mm256_zeroupper, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps,
};

use criterion::{criterion_group, criterion_main, Criterion};

pub fn basic_simd_load_benchmark(c: &mut Criterion) {
    const ROWS: &[usize] = &[100, 4000, 1000000];
    for &rows in ROWS {
        basic_simd_load_benchmark_impl::<8>(c, rows, 8, 0);
        basic_simd_load_benchmark_impl::<8>(c, rows, 8, 1);
        basic_simd_load_benchmark_impl::<8>(c, rows, 16, 0);
        basic_simd_load_benchmark_impl::<16>(c, rows, 16, 0);
        basic_simd_load_benchmark_impl::<16>(c, rows, 16, 1);
        basic_simd_load_benchmark_impl::<16>(c, rows, 32, 0);
    }
}

fn basic_simd_load_benchmark_impl<const LOAD: usize>(
    c: &mut Criterion,
    rows: usize,
    stride: usize,
    align: usize,
) {
    let data = vec![1.0f32; rows * stride + 16];
    let name = format!(
        "basic simd load x{} f32 {} rows, stride {}, alignment {}",
        LOAD, rows, stride, align
    );
    let mut group = c.benchmark_group(name);
    // Measure throughput in elements, not in Gb.
    group.throughput(criterion::Throughput::Elements(rows as u64 * LOAD as u64));

    let data_ptr = data.as_ptr();
    let aligned_ptr = unsafe {
        data_ptr
            .add(data_ptr.align_offset(8)) // Align for f32x8
            .add(align)
    };
    group.bench_function("scalar asm", |b| {
        b.iter(|| scalar_load_asm::<LOAD>(aligned_ptr, stride, rows))
    });
    group.bench_function("f32x4 asm", |b| {
        b.iter(|| f32x4_load_asm::<LOAD>(aligned_ptr, stride, rows))
    });
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        group.bench_function("f32x8 asm", |b| {
            b.iter(|| f32x8_load_asm::<LOAD>(aligned_ptr, stride, rows))
        });
    }
}

#[inline(always)]
fn scalar_load_asm<const LOAD: usize>(data: *const f32, stride: usize, rows: usize) {
    assert_eq!(rows % 2, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if cfg!(target_arch = "x86_64") {
        if LOAD == 8 {
            // Use the limited number of registers because XMM6-XMM15 are considered non-volatile
            // on Windows: https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention
            unsafe {
                asm!(
                ".p2align 4",
                "2:",
                "movss {tmp0}, dword ptr [{data}]",
                "movss {tmp1}, dword ptr [{data} + 4]",
                "movss {tmp2}, dword ptr [{data} + 8]",
                "movss {tmp3}, dword ptr [{data} + 12]",
                "movss {tmp4}, dword ptr [{data} + 16]",
                "movss {tmp5}, dword ptr [{data} + 20]",
                "movss {tmp0}, dword ptr [{data} + 24]",
                "movss {tmp1}, dword ptr [{data} + 28]",
                "add {data}, {stride}",
                "movss {tmp2}, dword ptr [{data}]",
                "movss {tmp3}, dword ptr [{data} + 4]",
                "movss {tmp4}, dword ptr [{data} + 8]",
                "movss {tmp5}, dword ptr [{data} + 12]",
                "movss {tmp0}, dword ptr [{data} + 16]",
                "movss {tmp1}, dword ptr [{data} + 20]",
                "movss {tmp2}, dword ptr [{data} + 24]",
                "movss {tmp3}, dword ptr [{data} + 28]",
                "add {data}, {stride}",
                "cmp {data}, {data_end}",
                "jb 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                stride = in(reg) stride * 4,
                tmp0 = out(xmm_reg) _,
                tmp1 = out(xmm_reg) _,
                tmp2 = out(xmm_reg) _,
                tmp3 = out(xmm_reg) _,
                tmp4 = out(xmm_reg) _,
                tmp5 = out(xmm_reg) _,
                options(nostack),
                );
            }
        } else if LOAD == 16 {
            unsafe {
                asm!(
                ".p2align 4",
                "2:",
                "movss {tmp0}, dword ptr [{data}]",
                "movss {tmp1}, dword ptr [{data} + 4]",
                "movss {tmp2}, dword ptr [{data} + 8]",
                "movss {tmp3}, dword ptr [{data} + 12]",
                "movss {tmp4}, dword ptr [{data} + 16]",
                "movss {tmp5}, dword ptr [{data} + 20]",
                "movss {tmp0}, dword ptr [{data} + 24]",
                "movss {tmp1}, dword ptr [{data} + 28]",
                "movss {tmp2}, dword ptr [{data} + 32]",
                "movss {tmp3}, dword ptr [{data} + 36]",
                "movss {tmp4}, dword ptr [{data} + 40]",
                "movss {tmp5}, dword ptr [{data} + 44]",
                "movss {tmp0}, dword ptr [{data} + 48]",
                "movss {tmp1}, dword ptr [{data} + 52]",
                "movss {tmp2}, dword ptr [{data} + 56]",
                "movss {tmp3}, dword ptr [{data} + 60]",
                "add {data}, {stride}",
                "cmp {data}, {data_end}",
                "jb 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                stride = in(reg) stride * 4,
                tmp0 = out(xmm_reg) _,
                tmp1 = out(xmm_reg) _,
                tmp2 = out(xmm_reg) _,
                tmp3 = out(xmm_reg) _,
                tmp4 = out(xmm_reg) _,
                tmp5 = out(xmm_reg) _,
                options(nostack)
                );
            }
        } else {
            unimplemented!();
        }
    } else if cfg!(target_arch = "aarch64") {
        unimplemented!();
    } else {
        unimplemented!();
    }
}

#[inline(always)]
fn f32x4_load_asm<const LOAD: usize>(data: *const f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if cfg!(target_arch = "x86_64") {
        if LOAD == 8 {
            // Use the limited number of registers because XMM6-XMM15 are considered non-volatile
            // on Windows: https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention
            if stride == 8 {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "movups {tmp0}, xmmword ptr [{data}]",
                    "movups {tmp1}, xmmword ptr [{data} + 16]",
                    "movups {tmp2}, xmmword ptr [{data} + 32]",
                    "movups {tmp3}, xmmword ptr [{data} + 48]",
                    "movups {tmp4}, xmmword ptr [{data} + 64]",
                    "movups {tmp5}, xmmword ptr [{data} + 80]",
                    "movups {tmp0}, xmmword ptr [{data} + 96]",
                    "movups {tmp1}, xmmword ptr [{data} + 112]",
                    "add {data}, 128",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = out(xmm_reg) _,
                    tmp1 = out(xmm_reg) _,
                    tmp2 = out(xmm_reg) _,
                    tmp3 = out(xmm_reg) _,
                    tmp4 = out(xmm_reg) _,
                    tmp5 = out(xmm_reg) _,
                    options(nostack)
                    );
                }
            } else {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "movups {tmp0}, xmmword ptr [{data}]",
                    "movups {tmp1}, xmmword ptr [{data} + 16]",
                    "add {data}, {stride}",
                    "movups {tmp2}, xmmword ptr [{data}]",
                    "movups {tmp3}, xmmword ptr [{data} + 16]",
                    "add {data}, {stride}",
                    "movups {tmp4}, xmmword ptr [{data}]",
                    "movups {tmp5}, xmmword ptr [{data} + 16]",
                    "add {data}, {stride}",
                    "movups {tmp0}, xmmword ptr [{data}]",
                    "movups {tmp1}, xmmword ptr [{data} + 16]",
                    "add {data}, {stride}",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = out(xmm_reg) _,
                    tmp1 = out(xmm_reg) _,
                    tmp2 = out(xmm_reg) _,
                    tmp3 = out(xmm_reg) _,
                    tmp4 = out(xmm_reg) _,
                    tmp5 = out(xmm_reg) _,
                    options(nostack)
                    );
                }
            }
        } else if LOAD == 16 {
            if stride == 16 {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "movups {tmp0}, xmmword ptr [{data}]",
                    "movups {tmp1}, xmmword ptr [{data} + 16]",
                    "movups {tmp2}, xmmword ptr [{data} + 32]",
                    "movups {tmp3}, xmmword ptr [{data} + 48]",
                    "movups {tmp4}, xmmword ptr [{data} + 64]",
                    "movups {tmp5}, xmmword ptr [{data} + 80]",
                    "movups {tmp0}, xmmword ptr [{data} + 96]",
                    "movups {tmp1}, xmmword ptr [{data} + 112]",
                    "movups {tmp2}, xmmword ptr [{data} + 128]",
                    "movups {tmp3}, xmmword ptr [{data} + 144]",
                    "movups {tmp4}, xmmword ptr [{data} + 160]",
                    "movups {tmp5}, xmmword ptr [{data} + 176]",
                    "movups {tmp0}, xmmword ptr [{data} + 192]",
                    "movups {tmp1}, xmmword ptr [{data} + 208]",
                    "movups {tmp2}, xmmword ptr [{data} + 224]",
                    "movups {tmp3}, xmmword ptr [{data} + 240]",
                    "add {data}, 256",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = out(xmm_reg) _,
                    tmp1 = out(xmm_reg) _,
                    tmp2 = out(xmm_reg) _,
                    tmp3 = out(xmm_reg) _,
                    tmp4 = out(xmm_reg) _,
                    tmp5 = out(xmm_reg) _,
                    options(nostack)
                    );
                }
            } else {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "movups {tmp0}, xmmword ptr [{data}]",
                    "movups {tmp1}, xmmword ptr [{data} + 16]",
                    "movups {tmp2}, xmmword ptr [{data} + 32]",
                    "movups {tmp3}, xmmword ptr [{data} + 48]",
                    "add {data}, {stride}",
                    "movups {tmp4}, xmmword ptr [{data}]",
                    "movups {tmp5}, xmmword ptr [{data} + 16]",
                    "movups {tmp0}, xmmword ptr [{data} + 32]",
                    "movups {tmp1}, xmmword ptr [{data} + 48]",
                    "add {data}, {stride}",
                    "movups {tmp2}, xmmword ptr [{data}]",
                    "movups {tmp3}, xmmword ptr [{data} + 16]",
                    "movups {tmp4}, xmmword ptr [{data} + 32]",
                    "movups {tmp5}, xmmword ptr [{data} + 48]",
                    "add {data}, {stride}",
                    "movups {tmp0}, xmmword ptr [{data}]",
                    "movups {tmp1}, xmmword ptr [{data} + 16]",
                    "movups {tmp2}, xmmword ptr [{data} + 32]",
                    "movups {tmp3}, xmmword ptr [{data} + 48]",
                    "add {data}, {stride}",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = out(xmm_reg) _,
                    tmp1 = out(xmm_reg) _,
                    tmp2 = out(xmm_reg) _,
                    tmp3 = out(xmm_reg) _,
                    tmp4 = out(xmm_reg) _,
                    tmp5 = out(xmm_reg) _,
                    options(nostack)
                    );
                }
            }
        } else {
            unimplemented!();
        }
    } else if cfg!(target_arch = "aarch64") {
        unimplemented!();
    } else {
        unimplemented!();
    }
}

#[inline(always)]
fn f32x8_load_asm<const LOAD: usize>(data: *const f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        if LOAD == 8 {
            if stride == 8 {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "vmovups {tmp0}, ymmword ptr [{data}]",
                    "vmovups {tmp1}, ymmword ptr [{data} + 32]",
                    "vmovups {tmp2}, ymmword ptr [{data} + 64]",
                    "vmovups {tmp3}, ymmword ptr [{data} + 96]",
                    "add {data}, 128",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = out(ymm_reg) _,
                    tmp1 = out(ymm_reg) _,
                    tmp2 = out(ymm_reg) _,
                    tmp3 = out(ymm_reg) _,
                    options(nostack)
                    );
                }
            } else {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "vmovups {tmp0}, ymmword ptr [{data}]",
                    "vmovups {tmp1}, ymmword ptr [{data} + {stride}]",
                    "lea {data}, [{data} + {stride} * 2]",
                    "vmovups {tmp2}, ymmword ptr [{data}]",
                    "vmovups {tmp3}, ymmword ptr [{data} + {stride}]",
                    "lea {data}, [{data} + {stride} * 2]",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = out(ymm_reg) _,
                    tmp1 = out(ymm_reg) _,
                    tmp2 = out(ymm_reg) _,
                    tmp3 = out(ymm_reg) _,
                    options(nostack)
                    );
                }
            }
        } else if LOAD == 16 {
            if stride == 16 {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "vmovups {tmp0}, ymmword ptr [{data}]",
                    "vmovups {tmp1}, ymmword ptr [{data} + 32]",
                    "vmovups {tmp2}, ymmword ptr [{data} + 64]",
                    "vmovups {tmp3}, ymmword ptr [{data} + 96]",
                    "vmovups {tmp4}, ymmword ptr [{data} + 128]",
                    "vmovups {tmp5}, ymmword ptr [{data} + 160]",
                    "vmovups {tmp0}, ymmword ptr [{data} + 192]",
                    "vmovups {tmp1}, ymmword ptr [{data} + 224]",
                    "add {data}, 256",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = out(ymm_reg) _,
                    tmp1 = out(ymm_reg) _,
                    tmp2 = out(ymm_reg) _,
                    tmp3 = out(ymm_reg) _,
                    tmp4 = out(ymm_reg) _,
                    tmp5 = out(ymm_reg) _,
                    options(nostack)
                    );
                }
            } else {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "vmovups {tmp0}, ymmword ptr [{data}]",
                    "vmovups {tmp1}, ymmword ptr [{data} + 32]",
                    "add {data}, {stride}",
                    "vmovups {tmp2}, ymmword ptr [{data}]",
                    "vmovups {tmp3}, ymmword ptr [{data} + 32]",
                    "add {data}, {stride}",
                    "vmovups {tmp4}, ymmword ptr [{data}]",
                    "vmovups {tmp5}, ymmword ptr [{data} + 32]",
                    "add {data}, {stride}",
                    "vmovups {tmp0}, ymmword ptr [{data}]",
                    "vmovups {tmp1}, ymmword ptr [{data} + 32]",
                    "add {data}, {stride}",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = out(ymm_reg) _,
                    tmp1 = out(ymm_reg) _,
                    tmp2 = out(ymm_reg) _,
                    tmp3 = out(ymm_reg) _,
                    tmp4 = out(ymm_reg) _,
                    tmp5 = out(ymm_reg) _,
                    options(nostack)
                    );
                }
            }
        } else {
            unimplemented!();
        }
    } else {
        unimplemented!();
    }
}

pub fn basic_simd_store_benchmark(c: &mut Criterion) {
    const ROWS: &[usize] = &[100, 4000, 1000000];
    for &rows in ROWS {
        basic_simd_store_benchmark_impl::<8>(c, rows, 8, 0);
        basic_simd_store_benchmark_impl::<8>(c, rows, 8, 1);
        basic_simd_store_benchmark_impl::<8>(c, rows, 16, 0);
        basic_simd_store_benchmark_impl::<16>(c, rows, 16, 0);
        basic_simd_store_benchmark_impl::<16>(c, rows, 16, 1);
        basic_simd_store_benchmark_impl::<16>(c, rows, 32, 0);
    }
}

fn basic_simd_store_benchmark_impl<const STORE: usize>(
    c: &mut Criterion,
    rows: usize,
    stride: usize,
    align: usize,
) {
    let mut data = vec![1.0f32; rows * stride + 16];
    let name = format!(
        "basic simd store x{} f32 {} rows, stride {}, alignment {}",
        STORE, rows, stride, align
    );
    let mut group = c.benchmark_group(name);
    // Measure throughput in elements, not in Gb.
    group.throughput(criterion::Throughput::Elements(rows as u64 * STORE as u64));

    let data_ptr = data.as_mut_ptr();
    let to_align = data_ptr.align_offset(8); // Align for f32x8
    let aligned_ptr = unsafe { data_ptr.add(to_align).add(align) };
    group.bench_function("scalar rust", |b| {
        data.fill(0.0);
        b.iter(|| scalar_store::<STORE>(aligned_ptr, stride, rows));
        assert_one(&data[to_align + align..], STORE, rows, stride);
    });
    group.bench_function("scalar asm", |b| {
        data.fill(0.0);
        b.iter(|| scalar_store_asm::<STORE>(aligned_ptr, stride, rows));
        assert_one(&data[to_align + align..], STORE, rows, stride);
    });
    group.bench_function("f32x4 rust", |b| {
        data.fill(0.0);
        b.iter(|| f32x4_store::<STORE>(aligned_ptr, stride, rows));
        assert_one(&data[to_align + align..], STORE, rows, stride);
    });
    group.bench_function("f32x4 asm", |b| {
        data.fill(0.0);
        b.iter(|| f32x4_store_asm::<STORE>(aligned_ptr, stride, rows));
        assert_one(&data[to_align + align..], STORE, rows, stride);
    });
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        group.bench_function("f32x8 rust", |b| {
            data.fill(0.0);
            b.iter(|| f32x8_store::<STORE>(aligned_ptr, stride, rows));
            assert_one(&data[to_align + align..], STORE, rows, stride);
        });
        group.bench_function("f32x8 asm", |b| {
            data.fill(0.0);
            b.iter(|| f32x8_store_asm::<STORE>(aligned_ptr, stride, rows));
            assert_one(&data[to_align + align..], STORE, rows, stride);
        });
    }
}

fn assert_one(data: &[f32], row_len: usize, rows: usize, stride: usize) {
    for i in 0..rows {
        for j in 0..row_len {
            assert_eq!(data[i * stride + j], 1.0, "row {}, idx {}", i, j);
        }
    }
}

#[inline(always)]
fn scalar_store<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 2, 0);
    if STORE == 8 {
        let mut i = 0;
        while i < rows {
            unsafe {
                *data.add(i * stride) = 1.0;
                *data.add(i * stride + 1) = 1.0;
                *data.add(i * stride + 2) = 1.0;
                *data.add(i * stride + 3) = 1.0;
                *data.add(i * stride + 4) = 1.0;
                *data.add(i * stride + 5) = 1.0;
                *data.add(i * stride + 6) = 1.0;
                *data.add(i * stride + 7) = 1.0;
                i += 1;
                *data.add(i * stride) = 1.0;
                *data.add(i * stride + 1) = 1.0;
                *data.add(i * stride + 2) = 1.0;
                *data.add(i * stride + 3) = 1.0;
                *data.add(i * stride + 4) = 1.0;
                *data.add(i * stride + 5) = 1.0;
                *data.add(i * stride + 6) = 1.0;
                *data.add(i * stride + 7) = 1.0;
                i += 1;
            }
        }
    } else if STORE == 16 {
        let mut i = 0;
        while i < rows {
            unsafe {
                *data.add(i * stride) = 1.0;
                *data.add(i * stride + 1) = 1.0;
                *data.add(i * stride + 2) = 1.0;
                *data.add(i * stride + 3) = 1.0;
                *data.add(i * stride + 4) = 1.0;
                *data.add(i * stride + 5) = 1.0;
                *data.add(i * stride + 6) = 1.0;
                *data.add(i * stride + 7) = 1.0;
                *data.add(i * stride + 8) = 1.0;
                *data.add(i * stride + 9) = 1.0;
                *data.add(i * stride + 10) = 1.0;
                *data.add(i * stride + 11) = 1.0;
                *data.add(i * stride + 12) = 1.0;
                *data.add(i * stride + 13) = 1.0;
                *data.add(i * stride + 14) = 1.0;
                *data.add(i * stride + 15) = 1.0;
                i += 1;
            }
        }
    } else {
        unimplemented!();
    }
}

#[inline(always)]
fn scalar_store_asm<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 2, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if cfg!(target_arch = "x86_64") {
        let one = unsafe { _mm_set1_ps(1.0) };
        if STORE == 8 {
            unsafe {
                asm!(
                ".p2align 4",
                "2:",
                "movss dword ptr [{data}], {tmp0}",
                "movss dword ptr [{data} + 4], {tmp0}",
                "movss dword ptr [{data} + 8], {tmp0}",
                "movss dword ptr [{data} + 12], {tmp0}",
                "movss dword ptr [{data} + 16], {tmp0}",
                "movss dword ptr [{data} + 20], {tmp0}",
                "movss dword ptr [{data} + 24], {tmp0}",
                "movss dword ptr [{data} + 28], {tmp0}",
                "add {data}, {stride}",
                "movss dword ptr [{data}], {tmp0}",
                "movss dword ptr [{data} + 4], {tmp0}",
                "movss dword ptr [{data} + 8], {tmp0}",
                "movss dword ptr [{data} + 12], {tmp0}",
                "movss dword ptr [{data} + 16], {tmp0}",
                "movss dword ptr [{data} + 20], {tmp0}",
                "movss dword ptr [{data} + 24], {tmp0}",
                "movss dword ptr [{data} + 28], {tmp0}",
                "add {data}, {stride}",
                "cmp {data}, {data_end}",
                "jb 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                stride = in(reg) stride * 4,
                tmp0 = in(xmm_reg) one,
                options(nostack),
                );
            }
        } else if STORE == 16 {
            unsafe {
                asm!(
                ".p2align 4",
                "2:",
                "movss dword ptr [{data}], {tmp0}",
                "movss dword ptr [{data} + 4], {tmp0}",
                "movss dword ptr [{data} + 8], {tmp0}",
                "movss dword ptr [{data} + 12], {tmp0}",
                "movss dword ptr [{data} + 16], {tmp0}",
                "movss dword ptr [{data} + 20], {tmp0}",
                "movss dword ptr [{data} + 24], {tmp0}",
                "movss dword ptr [{data} + 28], {tmp0}",
                "movss dword ptr [{data} + 32], {tmp0}",
                "movss dword ptr [{data} + 36], {tmp0}",
                "movss dword ptr [{data} + 40], {tmp0}",
                "movss dword ptr [{data} + 44], {tmp0}",
                "movss dword ptr [{data} + 48], {tmp0}",
                "movss dword ptr [{data} + 52], {tmp0}",
                "movss dword ptr [{data} + 56], {tmp0}",
                "movss dword ptr [{data} + 60], {tmp0}",
                "add {data}, {stride}",
                "cmp {data}, {data_end}",
                "jb 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                stride = in(reg) stride * 4,
                tmp0 = in(xmm_reg) one,
                options(nostack)
                );
            }
        } else {
            unimplemented!();
        }
    } else if cfg!(target_arch = "aarch64") {
        unimplemented!();
    } else {
        unimplemented!();
    }
}

#[inline(always)]
fn f32x4_store<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    if cfg!(target_arch = "x86_64") {
        let one = unsafe { _mm_set1_ps(1.0) };
        if STORE == 8 {
            if stride == 8 {
                let mut i = 0;
                while i < rows {
                    unsafe {
                        let ptr = data.add(i * 8);
                        _mm_storeu_ps(ptr, one);
                        _mm_storeu_ps(ptr.add(4), one);
                        _mm_storeu_ps(ptr.add(8), one);
                        _mm_storeu_ps(ptr.add(12), one);
                        _mm_storeu_ps(ptr.add(16), one);
                        _mm_storeu_ps(ptr.add(20), one);
                        _mm_storeu_ps(ptr.add(24), one);
                        _mm_storeu_ps(ptr.add(28), one);
                        i += 4;
                    }
                }
            } else {
                let mut i = 0;
                while i < rows {
                    unsafe {
                        let ptr = data.add(i * stride);
                        _mm_storeu_ps(ptr, one);
                        _mm_storeu_ps(ptr.add(4), one);
                        _mm_storeu_ps(ptr.add(stride), one);
                        _mm_storeu_ps(ptr.add(stride + 4), one);
                        _mm_storeu_ps(ptr.add(stride * 2), one);
                        _mm_storeu_ps(ptr.add(stride * 2 + 4), one);
                        _mm_storeu_ps(ptr.add(stride * 3), one);
                        _mm_storeu_ps(ptr.add(stride * 3 + 4), one);
                        i += 4;
                    }
                }
            }
        } else if STORE == 16 {
            if stride == 16 {
                let mut i = 0;
                while i < rows {
                    unsafe {
                        let ptr = data.add(i * 16);
                        _mm_storeu_ps(ptr, one);
                        _mm_storeu_ps(ptr.add(4), one);
                        _mm_storeu_ps(ptr.add(8), one);
                        _mm_storeu_ps(ptr.add(12), one);
                        _mm_storeu_ps(ptr.add(16), one);
                        _mm_storeu_ps(ptr.add(20), one);
                        _mm_storeu_ps(ptr.add(24), one);
                        _mm_storeu_ps(ptr.add(28), one);
                        _mm_storeu_ps(ptr.add(32), one);
                        _mm_storeu_ps(ptr.add(36), one);
                        _mm_storeu_ps(ptr.add(40), one);
                        _mm_storeu_ps(ptr.add(44), one);
                        _mm_storeu_ps(ptr.add(48), one);
                        _mm_storeu_ps(ptr.add(52), one);
                        _mm_storeu_ps(ptr.add(56), one);
                        _mm_storeu_ps(ptr.add(60), one);
                        i += 4;
                    }
                }
            } else {
                let mut i = 0;
                while i < rows {
                    unsafe {
                        let ptr = data.add(i * stride);
                        _mm_storeu_ps(ptr, one);
                        _mm_storeu_ps(ptr.add(4), one);
                        _mm_storeu_ps(ptr.add(8), one);
                        _mm_storeu_ps(ptr.add(12), one);
                        _mm_storeu_ps(ptr.add(stride), one);
                        _mm_storeu_ps(ptr.add(stride + 4), one);
                        _mm_storeu_ps(ptr.add(stride + 8), one);
                        _mm_storeu_ps(ptr.add(stride + 12), one);
                        _mm_storeu_ps(ptr.add(stride * 2), one);
                        _mm_storeu_ps(ptr.add(stride * 2 + 4), one);
                        _mm_storeu_ps(ptr.add(stride * 2 + 8), one);
                        _mm_storeu_ps(ptr.add(stride * 2 + 12), one);
                        _mm_storeu_ps(ptr.add(stride * 3), one);
                        _mm_storeu_ps(ptr.add(stride * 3 + 4), one);
                        _mm_storeu_ps(ptr.add(stride * 3 + 8), one);
                        _mm_storeu_ps(ptr.add(stride * 3 + 12), one);
                        i += 4;
                    }
                }
            }
        } else {
            unimplemented!();
        }
    } else if cfg!(target_arch = "aarch64") {
        unimplemented!();
    } else {
        unimplemented!();
    }
}

#[inline(always)]
fn f32x4_store_asm<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if cfg!(target_arch = "x86_64") {
        let one = unsafe { _mm_set1_ps(1.0) };
        if STORE == 8 {
            if stride == 8 {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "movups xmmword ptr [{data} + 32], {tmp0}",
                    "movups xmmword ptr [{data} + 48], {tmp0}",
                    "movups xmmword ptr [{data} + 64], {tmp0}",
                    "movups xmmword ptr [{data} + 80], {tmp0}",
                    "movups xmmword ptr [{data} + 96], {tmp0}",
                    "movups xmmword ptr [{data} + 112], {tmp0}",
                    "add {data}, 128",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = in(xmm_reg) one,
                    options(nostack),
                    );
                }
            } else {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "add {data}, {stride}",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "add {data}, {stride}",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "add {data}, {stride}",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "add {data}, {stride}",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = in(xmm_reg) one,
                    options(nostack),
                    );
                }
            }
        } else if STORE == 16 {
            if stride == 16 {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "movups xmmword ptr [{data} + 32], {tmp0}",
                    "movups xmmword ptr [{data} + 48], {tmp0}",
                    "movups xmmword ptr [{data} + 64], {tmp0}",
                    "movups xmmword ptr [{data} + 80], {tmp0}",
                    "movups xmmword ptr [{data} + 96], {tmp0}",
                    "movups xmmword ptr [{data} + 112], {tmp0}",
                    "movups xmmword ptr [{data} + 128], {tmp0}",
                    "movups xmmword ptr [{data} + 144], {tmp0}",
                    "movups xmmword ptr [{data} + 160], {tmp0}",
                    "movups xmmword ptr [{data} + 176], {tmp0}",
                    "movups xmmword ptr [{data} + 192], {tmp0}",
                    "movups xmmword ptr [{data} + 208], {tmp0}",
                    "movups xmmword ptr [{data} + 224], {tmp0}",
                    "movups xmmword ptr [{data} + 240], {tmp0}",
                    "add {data}, 256",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = in(xmm_reg) one,
                    options(nostack),
                    );
                }
            } else {
                unsafe {
                    asm!(
                    ".p2align 4",
                    "2:",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "movups xmmword ptr [{data} + 32], {tmp0}",
                    "movups xmmword ptr [{data} + 48], {tmp0}",
                    "add {data}, {stride}",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "movups xmmword ptr [{data} + 32], {tmp0}",
                    "movups xmmword ptr [{data} + 48], {tmp0}",
                    "add {data}, {stride}",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "movups xmmword ptr [{data} + 32], {tmp0}",
                    "movups xmmword ptr [{data} + 48], {tmp0}",
                    "add {data}, {stride}",
                    "movups xmmword ptr [{data}], {tmp0}",
                    "movups xmmword ptr [{data} + 16], {tmp0}",
                    "movups xmmword ptr [{data} + 32], {tmp0}",
                    "movups xmmword ptr [{data} + 48], {tmp0}",
                    "add {data}, {stride}",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = in(xmm_reg) one,
                    options(nostack),
                    );
                }
            }
        } else {
            unimplemented!();
        }
    } else if cfg!(target_arch = "aarch64") {
        unimplemented!();
    } else {
        unimplemented!();
    }
}

#[inline(always)]
fn f32x8_store<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        let one = unsafe { _mm256_set1_ps(1.0) };
        if STORE == 8 {
            if stride == 8 {
                let mut i = 0;
                while i < rows {
                    unsafe {
                        let ptr = data.add(i * 8);
                        _mm256_storeu_ps(ptr, one);
                        _mm256_storeu_ps(ptr.add(8), one);
                        _mm256_storeu_ps(ptr.add(16), one);
                        _mm256_storeu_ps(ptr.add(24), one);
                        i += 4;
                    }
                }
            } else {
                let mut i = 0;
                while i < rows {
                    unsafe {
                        let ptr = data.add(i * stride);
                        _mm256_storeu_ps(ptr, one);
                        _mm256_storeu_ps(ptr.add(stride), one);
                        _mm256_storeu_ps(ptr.add(stride * 2), one);
                        _mm256_storeu_ps(ptr.add(stride * 3), one);
                        i += 4;
                    }
                }
            }
        } else if STORE == 16 {
            if stride == 16 {
                let mut i = 0;
                while i < rows {
                    unsafe {
                        let ptr = data.add(i * 16);
                        _mm256_storeu_ps(ptr, one);
                        _mm256_storeu_ps(ptr.add(8), one);
                        _mm256_storeu_ps(ptr.add(16), one);
                        _mm256_storeu_ps(ptr.add(24), one);
                        _mm256_storeu_ps(ptr.add(32), one);
                        _mm256_storeu_ps(ptr.add(40), one);
                        _mm256_storeu_ps(ptr.add(48), one);
                        _mm256_storeu_ps(ptr.add(56), one);
                        i += 4;
                    }
                }
            } else {
                let mut i = 0;
                while i < rows {
                    unsafe {
                        let ptr = data.add(i * stride);
                        _mm256_storeu_ps(ptr, one);
                        _mm256_storeu_ps(ptr.add(8), one);
                        _mm256_storeu_ps(ptr.add(stride), one);
                        _mm256_storeu_ps(ptr.add(stride + 8), one);
                        _mm256_storeu_ps(ptr.add(stride * 2), one);
                        _mm256_storeu_ps(ptr.add(stride * 2 + 8), one);
                        _mm256_storeu_ps(ptr.add(stride * 3), one);
                        _mm256_storeu_ps(ptr.add(stride * 3 + 8), one);
                        i += 4;
                    }
                }
            }
        } else {
            unimplemented!();
        }
        unsafe {
            _mm256_zeroupper();
        }
    } else {
        unimplemented!();
    }
}

#[inline(always)]
fn f32x8_store_asm<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        // Can't use _mm256_set1_ps due to vzeroupper.
        let one = &[1.0f32];
        if STORE == 8 {
            if stride == 8 {
                unsafe {
                    asm!(
                    "vbroadcastss {tmp0}, [{one}]",
                    ".p2align 4",
                    "2:",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "vmovups ymmword ptr [{data} + 64], {tmp0}",
                    "vmovups ymmword ptr [{data} + 96], {tmp0}",
                    "add {data}, 128",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    one = in(reg) one.as_ptr(),
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = out(ymm_reg) _,
                    options(nostack),
                    );
                }
            } else {
                unsafe {
                    asm!(
                    "vbroadcastss {tmp0}, [{one}]",
                    ".p2align 4",
                    "2:",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + {stride}], {tmp0}",
                    "lea {data}, [{data} + {stride} * 2]",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + {stride}], {tmp0}",
                    "lea {data}, [{data} + {stride} * 2]",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    one = in(reg) one.as_ptr(),
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = out(ymm_reg) _,
                    options(nostack),
                    );
                }
            }
        } else if STORE == 16 {
            if stride == 16 {
                unsafe {
                    asm!(
                    "vbroadcastss {tmp0}, [{one}]",
                    ".p2align 4",
                    "2:",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "vmovups ymmword ptr [{data} + 64], {tmp0}",
                    "vmovups ymmword ptr [{data} + 96], {tmp0}",
                    "vmovups ymmword ptr [{data} + 128], {tmp0}",
                    "vmovups ymmword ptr [{data} + 160], {tmp0}",
                    "vmovups ymmword ptr [{data} + 192], {tmp0}",
                    "vmovups ymmword ptr [{data} + 224], {tmp0}",
                    "add {data}, 256",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    one = in(reg) one.as_ptr(),
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    tmp0 = out(ymm_reg) _,
                    options(nostack),
                    );
                }
            } else {
                unsafe {
                    asm!(
                    "vbroadcastss {tmp0}, [{one}]",
                    ".p2align 4",
                    "2:",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "add {data}, {stride}",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "add {data}, {stride}",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "add {data}, {stride}",
                    "vmovups ymmword ptr [{data}], {tmp0}",
                    "vmovups ymmword ptr [{data} + 32], {tmp0}",
                    "add {data}, {stride}",
                    "cmp {data}, {data_end}",
                    "jb 2b",
                    "vzeroupper",
                    one = in(reg) one.as_ptr(),
                    data = inout(reg) data => _,
                    data_end = in(reg) data_end,
                    stride = in(reg) stride * 4,
                    tmp0 = out(ymm_reg) _,
                    options(nostack),
                    );
                }
            }
        } else {
            unimplemented!();
        }
    } else {
        unimplemented!();
    }
}

// Do in-register adds with single target to check latency.
pub fn basic_simd_add_latency_benchmark(c: &mut Criterion) {
    const ITERATIONS: usize = 1000000;
    let mut group = c.benchmark_group("basic simd add-latency f32");
    // Measure throughput in flops, not in Gb.
    group.throughput(criterion::Throughput::Elements(ITERATIONS as u64 * 8));
    let arr = [1.0f32; 8];
    let brr = [3.0f32; 8];
    let mut golden_result = [0.0f32; 8];
    scalar_add_latency(ITERATIONS, &arr, &brr, &mut golden_result);

    let mut result = [0.0f32; 8];
    group.bench_function("scalar rust", |b| {
        result.fill(0.0);
        b.iter(|| scalar_add_latency(ITERATIONS, &arr, &brr, &mut result));
        assert_eq!(result, golden_result);
    });
    // Skip scalar asm code and intrinsic code.
    group.bench_function("f32x4 asm", |b| {
        result.fill(0.0);
        b.iter(|| f32x4_add_latency_asm(ITERATIONS, &arr, &brr, &mut result));
        assert_eq!(result, golden_result);
    });
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        group.bench_function("f32x8 asm", |b| {
            result.fill(0.0);
            b.iter(|| f32x8_add_latency_asm(ITERATIONS, &arr, &brr, &mut result));
            assert_eq!(result, golden_result);
        });
    }
}

#[inline(always)]
fn scalar_add_latency(iterations: usize, a: &[f32; 8], b: &[f32; 8], result: &mut [f32; 8]) {
    for j in 0..8 {
        result[j] = a[j];
    }
    for _ in 0..iterations {
        for j in 0..8 {
            unsafe {
                *result.get_unchecked_mut(j) += *b.get_unchecked(j);
            }
        }
    }
}

#[inline(always)]
fn f32x4_add_latency_asm(iterations: usize, a: &[f32; 8], b: &[f32; 8], result: &mut [f32; 8]) {
    assert_eq!(iterations % 4, 0);
    unsafe {
        asm!(
        "movups {a0}, xmmword ptr [{a}]",
        "movups {a1}, xmmword ptr [{a} + 16]",
        "movups {tmp0}, xmmword ptr [{b}]",
        "movups {tmp1}, xmmword ptr [{b} + 16]",
        ".p2align 4",
        "2:",
        "addps {a0}, {tmp0}",
        "addps {a1}, {tmp1}",
        "addps {a0}, {tmp0}",
        "addps {a1}, {tmp1}",
        "addps {a0}, {tmp0}",
        "addps {a1}, {tmp1}",
        "addps {a0}, {tmp0}",
        "addps {a1}, {tmp1}",
        "dec {iterations}",
        "jnz 2b",
        "movups xmmword ptr [{result}], {a0}",
        "movups xmmword ptr [{result} + 16], {a1}",
        a = in(reg) a.as_ptr(),
        b = in(reg) b.as_ptr(),
        result = in(reg) result.as_ptr(),
        iterations = inout(reg) iterations / 4 => _,
        a0 = out(xmm_reg) _,
        a1 = out(xmm_reg) _,
        tmp0 = out(xmm_reg) _,
        tmp1 = out(xmm_reg) _,
        options(nostack),
        );
    }
}

#[inline(always)]
fn f32x8_add_latency_asm(iterations: usize, a: &[f32; 8], b: &[f32; 8], result: &mut [f32; 8]) {
    assert_eq!(iterations % 4, 0);
    unsafe {
        asm!(
        "vmovups {a0}, ymmword ptr [{a}]",
        "vmovups {b0}, ymmword ptr [{b}]",
        ".p2align 4",
        "2:",
        "vaddps {a0}, {a0}, {b0}",
        "vaddps {a0}, {a0}, {b0}",
        "vaddps {a0}, {a0}, {b0}",
        "vaddps {a0}, {a0}, {b0}",
        "dec {iterations}",
        "jnz 2b",
        "vmovups ymmword ptr [{result}], {a0}",
        "vzeroupper",
        a = in(reg) a.as_ptr(),
        b = in(reg) b.as_ptr(),
        result = in(reg) result.as_ptr(),
        iterations = inout(reg) iterations / 4 => _,
        a0 = out(ymm_reg) _,
        b0 = out(ymm_reg) _,
        options(nostack),
        );
    }
}

// Do in-register (or at least try to in SSE2 case) fma with multiple target regs to avoid being
// latency-bound.
pub fn basic_simd_fma_benchmark(c: &mut Criterion) {
    const ITERATIONS: usize = 1000000;
    let mut group = c.benchmark_group("basic simd fma f32");
    // Measure throughput in flops, not in Gb. We count one fma as two flops.
    group.throughput(criterion::Throughput::Elements(ITERATIONS as u64 * 64));
    let arr = [1.0f32; 16];
    let brr = [3.0f32; 16];
    let crr = [0.1f32; 16];
    let drr = [4.0f32; 16];
    let err = [0.3f32; 16];
    let mut golden_result = [0.0f32; 32];
    scalar_fma(ITERATIONS, &arr, &brr, &crr, &drr, &err, &mut golden_result);

    let mut result = [0.0f32; 32];
    group.bench_function("scalar rust", |b| {
        result.fill(0.0);
        b.iter(|| scalar_fma(ITERATIONS, &arr, &brr, &crr, &drr, &err, &mut result));
        assert_eq!(result, golden_result);
    });
    // Skip scalar asm code.
    group.bench_function("f32x4 rust", |b| {
        result.fill(0.0);
        b.iter(|| f32x4_fma(ITERATIONS, &arr, &brr, &crr, &drr, &err, &mut result));
        assert_eq!(result, golden_result);
    });
    group.bench_function("f32x4 asm", |b| {
        result.fill(0.0);
        b.iter(|| f32x4_fma_asm(ITERATIONS, &arr, &brr, &crr, &drr, &err, &mut result));
        assert_eq!(result, golden_result);
    });
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        group.bench_function("f32x8 rust", |b| {
            result.fill(0.0);
            b.iter(|| f32x8_fma(ITERATIONS, &arr, &brr, &crr, &drr, &err, &mut result));
            assert_eq!(result, golden_result);
        });
        group.bench_function("f32x8 asm", |b| {
            result.fill(0.0);
            b.iter(|| f32x8_fma_asm(ITERATIONS, &arr, &brr, &crr, &drr, &err, &mut result));
            assert_eq!(result, golden_result);
        });
    }
}

#[inline(always)]
fn scalar_fma(
    iterations: usize,
    a: &[f32; 16],
    b: &[f32; 16],
    c: &[f32; 16],
    d: &[f32; 16],
    e: &[f32; 16],
    result: &mut [f32; 32],
) {
    for j in 0..16 {
        result[j] = a[j];
        result[j + 16] = a[j];
    }
    for _ in 0..iterations {
        for j in 0..16 {
            unsafe {
                *result.get_unchecked_mut(j) += *b.get_unchecked(j) * *c.get_unchecked(j);
                *result.get_unchecked_mut(j + 16) += *d.get_unchecked(j) * *e.get_unchecked(j);
            }
        }
    }
}

#[inline(always)]
fn f32x4_fma(
    iterations: usize,
    a: &[f32; 16],
    b: &[f32; 16],
    c: &[f32; 16],
    d: &[f32; 16],
    e: &[f32; 16],
    result: &mut [f32; 32],
) {
    assert_eq!(iterations % 2, 0);
    let ap = a.as_ptr();
    let mut a0 = unsafe { _mm_loadu_ps(ap) };
    let mut a1 = unsafe { _mm_loadu_ps(ap.add(4)) };
    let mut a2 = unsafe { _mm_loadu_ps(ap.add(8)) };
    let mut a3 = unsafe { _mm_loadu_ps(ap.add(12)) };
    let mut a4 = unsafe { _mm_loadu_ps(ap) };
    let mut a5 = unsafe { _mm_loadu_ps(ap.add(4)) };
    let mut a6 = unsafe { _mm_loadu_ps(ap.add(8)) };
    let mut a7 = unsafe { _mm_loadu_ps(ap.add(12)) };
    let bp = b.as_ptr();
    let b0 = unsafe { _mm_loadu_ps(bp) };
    let b1 = unsafe { _mm_loadu_ps(bp.add(4)) };
    let b2 = unsafe { _mm_loadu_ps(bp.add(8)) };
    let b3 = unsafe { _mm_loadu_ps(bp.add(12)) };
    let cp = c.as_ptr();
    let c0 = unsafe { _mm_loadu_ps(cp) };
    let c1 = unsafe { _mm_loadu_ps(cp.add(4)) };
    let c2 = unsafe { _mm_loadu_ps(cp.add(8)) };
    let c3 = unsafe { _mm_loadu_ps(cp.add(12)) };
    let dp = d.as_ptr();
    let d0 = unsafe { _mm_loadu_ps(dp) };
    let d1 = unsafe { _mm_loadu_ps(dp.add(4)) };
    let d2 = unsafe { _mm_loadu_ps(dp.add(8)) };
    let d3 = unsafe { _mm_loadu_ps(dp.add(12)) };
    let ep = e.as_ptr();
    let e0 = unsafe { _mm_loadu_ps(ep) };
    let e1 = unsafe { _mm_loadu_ps(ep.add(4)) };
    let e2 = unsafe { _mm_loadu_ps(ep.add(8)) };
    let e3 = unsafe { _mm_loadu_ps(ep.add(12)) };
    for _ in 0..iterations {
        unsafe {
            a0 = _mm_add_ps(a0, _mm_mul_ps(b0, c0));
            a1 = _mm_add_ps(a1, _mm_mul_ps(b1, c1));
            a2 = _mm_add_ps(a2, _mm_mul_ps(b2, c2));
            a3 = _mm_add_ps(a3, _mm_mul_ps(b3, c3));
            a4 = _mm_add_ps(a4, _mm_mul_ps(d0, e0));
            a5 = _mm_add_ps(a5, _mm_mul_ps(d1, e1));
            a6 = _mm_add_ps(a6, _mm_mul_ps(d2, e2));
            a7 = _mm_add_ps(a7, _mm_mul_ps(d3, e3));
        }
    }
    unsafe {
        let rp = result.as_mut_ptr();
        _mm_storeu_ps(rp, a0);
        _mm_storeu_ps(rp.add(4), a1);
        _mm_storeu_ps(rp.add(8), a2);
        _mm_storeu_ps(rp.add(12), a3);
        _mm_storeu_ps(rp.add(16), a4);
        _mm_storeu_ps(rp.add(20), a5);
        _mm_storeu_ps(rp.add(24), a6);
        _mm_storeu_ps(rp.add(28), a7);
    }
}

#[inline(always)]
fn f32x4_fma_asm(
    iterations: usize,
    a: &[f32; 16],
    b: &[f32; 16],
    c: &[f32; 16],
    d: &[f32; 16],
    e: &[f32; 16],
    result: &mut [f32; 32],
) {
    unsafe {
        asm!(
        "movups {a0}, xmmword ptr [{a}]",
        "movups {a1}, xmmword ptr [{a} + 16]",
        "movups {a2}, xmmword ptr [{a} + 32]",
        "movups {a3}, xmmword ptr [{a} + 48]",
        "movups {a4}, xmmword ptr [{a}]",
        "movups {a5}, xmmword ptr [{a} + 16]",
        "movups {a6}, xmmword ptr [{a} + 32]",
        "movups {a7}, xmmword ptr [{a} + 48]",
        ".p2align 4",
        "2:",
        "movups {tmp0}, xmmword ptr [{b}]",
        "movups {tmp1}, xmmword ptr [{b} + 16]",
        "movups {tmp2}, xmmword ptr [{b} + 32]",
        "movups {tmp3}, xmmword ptr [{b} + 48]",
        "movups {tmp4}, xmmword ptr [{d}]",
        "movups {tmp5}, xmmword ptr [{d} + 16]",
        "movups {tmp6}, xmmword ptr [{d} + 32]",
        "movups {tmp7}, xmmword ptr [{d} + 48]",
        "mulps {tmp0}, xmmword ptr [{c}]",
        "mulps {tmp1}, xmmword ptr [{c} + 16]",
        "mulps {tmp2}, xmmword ptr [{c} + 32]",
        "mulps {tmp3}, xmmword ptr [{c} + 48]",
        "mulps {tmp4}, xmmword ptr [{e}]",
        "mulps {tmp5}, xmmword ptr [{e} + 16]",
        "mulps {tmp6}, xmmword ptr [{e} + 32]",
        "mulps {tmp7}, xmmword ptr [{e} + 48]",
        "addps {a0}, {tmp0}",
        "addps {a1}, {tmp1}",
        "addps {a2}, {tmp2}",
        "addps {a3}, {tmp3}",
        "addps {a4}, {tmp4}",
        "addps {a5}, {tmp5}",
        "addps {a6}, {tmp6}",
        "addps {a7}, {tmp7}",
        "dec {iterations}",
        "jnz 2b",
        "movups xmmword ptr [{result}], {a0}",
        "movups xmmword ptr [{result} + 16], {a1}",
        "movups xmmword ptr [{result} + 32], {a2}",
        "movups xmmword ptr [{result} + 48], {a3}",
        "movups xmmword ptr [{result} + 64], {a4}",
        "movups xmmword ptr [{result} + 80], {a5}",
        "movups xmmword ptr [{result} + 96], {a6}",
        "movups xmmword ptr [{result} + 112], {a7}",
        a = in(reg) a.as_ptr(),
        b = in(reg) b.as_ptr(),
        c = in(reg) c.as_ptr(),
        d = in(reg) d.as_ptr(),
        e = in(reg) e.as_ptr(),
        result = in(reg) result.as_ptr(),
        iterations = inout(reg) iterations => _,
        a0 = out(xmm_reg) _,
        a1 = out(xmm_reg) _,
        a2 = out(xmm_reg) _,
        a3 = out(xmm_reg) _,
        a4 = out(xmm_reg) _,
        a5 = out(xmm_reg) _,
        a6 = out(xmm_reg) _,
        a7 = out(xmm_reg) _,
        tmp0 = out(xmm_reg) _,
        tmp1 = out(xmm_reg) _,
        tmp2 = out(xmm_reg) _,
        tmp3 = out(xmm_reg) _,
        tmp4 = out(xmm_reg) _,
        tmp5 = out(xmm_reg) _,
        tmp6 = out(xmm_reg) _,
        tmp7 = out(xmm_reg) _,
        options(nostack),
        );
    }
}

#[inline(always)]
fn f32x8_fma(
    iterations: usize,
    a: &[f32; 16],
    b: &[f32; 16],
    c: &[f32; 16],
    d: &[f32; 16],
    e: &[f32; 16],
    result: &mut [f32; 32],
) {
    assert_eq!(iterations % 2, 0);
    let ap = a.as_ptr();
    let mut a0 = unsafe { _mm256_loadu_ps(ap) };
    let mut a1 = unsafe { _mm256_loadu_ps(ap.add(8)) };
    let mut a2 = unsafe { _mm256_loadu_ps(ap) };
    let mut a3 = unsafe { _mm256_loadu_ps(ap.add(8)) };
    let bp = b.as_ptr();
    let b0 = unsafe { _mm256_loadu_ps(bp) };
    let b1 = unsafe { _mm256_loadu_ps(bp.add(8)) };
    let cp = c.as_ptr();
    let c0 = unsafe { _mm256_loadu_ps(cp) };
    let c1 = unsafe { _mm256_loadu_ps(cp.add(8)) };
    let dp = d.as_ptr();
    let d0 = unsafe { _mm256_loadu_ps(dp) };
    let d1 = unsafe { _mm256_loadu_ps(dp.add(8)) };
    let ep = e.as_ptr();
    let e0 = unsafe { _mm256_loadu_ps(ep) };
    let e1 = unsafe { _mm256_loadu_ps(ep.add(8)) };
    for _ in 0..iterations {
        unsafe {
            a0 = _mm256_fmadd_ps(b0, c0, a0);
            a1 = _mm256_fmadd_ps(b1, c1, a1);
            a2 = _mm256_fmadd_ps(d0, e0, a2);
            a3 = _mm256_fmadd_ps(d1, e1, a3);
        }
    }
    unsafe {
        let rp = result.as_mut_ptr();
        _mm256_storeu_ps(rp, a0);
        _mm256_storeu_ps(rp.add(8), a1);
        _mm256_storeu_ps(rp.add(16), a2);
        _mm256_storeu_ps(rp.add(24), a3);
        _mm256_zeroupper();
    }
}

#[inline(always)]
fn f32x8_fma_asm(
    iterations: usize,
    a: &[f32; 16],
    b: &[f32; 16],
    c: &[f32; 16],
    d: &[f32; 16],
    e: &[f32; 16],
    result: &mut [f32; 32],
) {
    assert_eq!(iterations % 2, 0);
    unsafe {
        asm!(
        "vmovups {a0}, ymmword ptr [{a}]",
        "vmovups {a1}, ymmword ptr [{a} + 32]",
        "vmovups {a2}, ymmword ptr [{a}]",
        "vmovups {a3}, ymmword ptr [{a} + 32]",
        "vmovups {b0}, ymmword ptr [{b}]",
        "vmovups {b1}, ymmword ptr [{b} + 32]",
        "vmovups {c0}, ymmword ptr [{c}]",
        "vmovups {c1}, ymmword ptr [{c} + 32]",
        "vmovups {d0}, ymmword ptr [{d}]",
        "vmovups {d1}, ymmword ptr [{d} + 32]",
        "vmovups {e0}, ymmword ptr [{e}]",
        "vmovups {e1}, ymmword ptr [{e} + 32]",
        ".p2align 4",
        "2:",
        "vfmadd231ps {a0}, {b0}, {c0}",
        "vfmadd231ps {a1}, {b1}, {c1}",
        "vfmadd231ps {a2}, {d0}, {e0}",
        "vfmadd231ps {a3}, {d1}, {e1}",
        "vfmadd231ps {a0}, {b0}, {c0}",
        "vfmadd231ps {a1}, {b1}, {c1}",
        "vfmadd231ps {a2}, {d0}, {e0}",
        "vfmadd231ps {a3}, {d1}, {e1}",
        "dec {iterations}",
        "jnz 2b",
        "vmovups ymmword ptr [{result}], {a0}",
        "vmovups ymmword ptr [{result} + 32], {a1}",
        "vmovups ymmword ptr [{result} + 64], {a2}",
        "vmovups ymmword ptr [{result} + 96], {a3}",
        "vzeroupper",
        a = in(reg) a.as_ptr(),
        b = in(reg) b.as_ptr(),
        c = in(reg) c.as_ptr(),
        d = in(reg) d.as_ptr(),
        e = in(reg) e.as_ptr(),
        result = in(reg) result.as_ptr(),
        iterations = inout(reg) iterations / 2 => _,
        a0 = out(ymm_reg) _,
        a1 = out(ymm_reg) _,
        a2 = out(ymm_reg) _,
        a3 = out(ymm_reg) _,
        b0 = out(ymm_reg) _,
        b1 = out(ymm_reg) _,
        c0 = out(ymm_reg) _,
        c1 = out(ymm_reg) _,
        d0 = out(ymm_reg) _,
        d1 = out(ymm_reg) _,
        e0 = out(ymm_reg) _,
        e1 = out(ymm_reg) _,
        options(nostack),
        );
    }
}

// Do load+fma with multiple target regs to avoid being latency-bound.
pub fn basic_simd_load_fma_benchmark(c: &mut Criterion) {
    const ROWS: &[usize] = &[100, 4000, 1000000];
    for &rows in ROWS {
        let data = vec![1.0f32; rows * 16];
        let data_ptr = data.as_ptr();
        let mut group = c.benchmark_group(format!("basic simd load-fma f32 {} rows", rows));
        // Measure throughput in flops, not in Gb. We count one fma as two flops.
        group.throughput(criterion::Throughput::Elements(rows as u64 * 16));
        let mut golden_result = [0.0f32; 32];
        scalar_load_fma(data_ptr, rows, &mut golden_result);

        let mut result = [0.0f32; 32];
        group.bench_function("scalar rust", |b| {
            result.fill(0.0);
            b.iter(|| scalar_load_fma(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });
        // Skip scalar asm code.
        group.bench_function("f32x4 rust", |b| {
            result.fill(0.0);
            b.iter(|| f32x4_load_fma(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });
        group.bench_function("f32x4 asm", |b| {
            result.fill(0.0);
            b.iter(|| f32x4_load_fma_asm(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });
        if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
            group.bench_function("f32x8 rust", |b| {
                result.fill(0.0);
                b.iter(|| f32x8_load_fma(data_ptr, rows, &mut result));
                assert_eq!(result, golden_result);
            });
            group.bench_function("f32x8 asm", |b| {
                result.fill(0.0);
                b.iter(|| f32x8_load_fma_asm(data_ptr, rows, &mut result));
                assert_eq!(result, golden_result);
            });
        }
    }
}

#[inline(always)]
fn scalar_load_fma(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    let mut acc = [0.0f32; 32];
    for j in 0..rows / 4 {
        for k in 0..16 {
            unsafe {
                *acc.get_unchecked_mut(k) +=
                    *data.add((j * 4) * 16 + k) * *data.add((j * 4 + 1) * 16 + k);
                *acc.get_unchecked_mut(k + 16) +=
                    *data.add((j * 4 + 2) * 16 + k) * *data.add((j * 4 + 3) * 16 + k);
            }
        }
    }
    *result = acc;
}

#[inline(always)]
fn f32x4_load_fma(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    assert_eq!(rows % 4, 0);
    let mut a0 = unsafe { _mm_set1_ps(0.0) };
    let mut a1 = unsafe { _mm_set1_ps(0.0) };
    let mut a2 = unsafe { _mm_set1_ps(0.0) };
    let mut a3 = unsafe { _mm_set1_ps(0.0) };
    let mut a4 = unsafe { _mm_set1_ps(0.0) };
    let mut a5 = unsafe { _mm_set1_ps(0.0) };
    let mut a6 = unsafe { _mm_set1_ps(0.0) };
    let mut a7 = unsafe { _mm_set1_ps(0.0) };
    for j in 0..rows / 4 {
        unsafe {
            let r00 = _mm_loadu_ps(data.add(j * 4 * 16));
            let r01 = _mm_loadu_ps(data.add(j * 4 * 16 + 4));
            let r02 = _mm_loadu_ps(data.add(j * 4 * 16 + 8));
            let r03 = _mm_loadu_ps(data.add(j * 4 * 16 + 12));
            let r10 = _mm_loadu_ps(data.add((j * 4 + 1) * 16));
            let r11 = _mm_loadu_ps(data.add((j * 4 + 1) * 16 + 4));
            let r12 = _mm_loadu_ps(data.add((j * 4 + 1) * 16 + 8));
            let r13 = _mm_loadu_ps(data.add((j * 4 + 1) * 16 + 12));
            a0 = _mm_add_ps(a0, _mm_mul_ps(r00, r10));
            a1 = _mm_add_ps(a1, _mm_mul_ps(r01, r11));
            a2 = _mm_add_ps(a2, _mm_mul_ps(r02, r12));
            a3 = _mm_add_ps(a3, _mm_mul_ps(r03, r13));
            let r00 = _mm_loadu_ps(data.add((j * 4 + 2) * 16));
            let r01 = _mm_loadu_ps(data.add((j * 4 + 2) * 16 + 4));
            let r02 = _mm_loadu_ps(data.add((j * 4 + 2) * 16 + 8));
            let r03 = _mm_loadu_ps(data.add((j * 4 + 2) * 16 + 12));
            let r10 = _mm_loadu_ps(data.add((j * 4 + 3) * 16));
            let r11 = _mm_loadu_ps(data.add((j * 4 + 3) * 16 + 4));
            let r12 = _mm_loadu_ps(data.add((j * 4 + 3) * 16 + 8));
            let r13 = _mm_loadu_ps(data.add((j * 4 + 3) * 16 + 12));
            a4 = _mm_add_ps(a4, _mm_mul_ps(r00, r10));
            a5 = _mm_add_ps(a5, _mm_mul_ps(r01, r11));
            a6 = _mm_add_ps(a6, _mm_mul_ps(r02, r12));
            a7 = _mm_add_ps(a7, _mm_mul_ps(r03, r13));
        }
    }
    unsafe {
        let rp = result.as_mut_ptr();
        _mm_storeu_ps(rp, a0);
        _mm_storeu_ps(rp.add(4), a1);
        _mm_storeu_ps(rp.add(8), a2);
        _mm_storeu_ps(rp.add(12), a3);
        _mm_storeu_ps(rp.add(16), a4);
        _mm_storeu_ps(rp.add(20), a5);
        _mm_storeu_ps(rp.add(24), a6);
        _mm_storeu_ps(rp.add(28), a7);
    }
}

#[inline(always)]
fn f32x4_load_fma_asm(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    assert_eq!(rows % 4, 0);
    unsafe {
        asm!(
        "xorps {a0}, {a0}",
        "xorps {a1}, {a1}",
        "xorps {a2}, {a2}",
        "xorps {a3}, {a3}",
        "xorps {a4}, {a4}",
        "xorps {a5}, {a5}",
        "xorps {a6}, {a6}",
        "xorps {a7}, {a7}",
        ".p2align 4",
        "2:",
        "movups {tmp0}, xmmword ptr [{data}]",
        "movups {tmp1}, xmmword ptr [{data} + 16]",
        "movups {tmp2}, xmmword ptr [{data} + 32]",
        "movups {tmp3}, xmmword ptr [{data} + 48]",
        "movups {tmp4}, xmmword ptr [{data} + 128]",
        "movups {tmp5}, xmmword ptr [{data} + 144]",
        "movups {tmp6}, xmmword ptr [{data} + 160]",
        "movups {tmp7}, xmmword ptr [{data} + 176]",
        "mulps {tmp0}, xmmword ptr [{data} + 64]",
        "mulps {tmp1}, xmmword ptr [{data} + 80]",
        "mulps {tmp2}, xmmword ptr [{data} + 96]",
        "mulps {tmp3}, xmmword ptr [{data} + 112]",
        "mulps {tmp0}, xmmword ptr [{data} + 192]",
        "mulps {tmp1}, xmmword ptr [{data} + 208]",
        "mulps {tmp2}, xmmword ptr [{data} + 224]",
        "mulps {tmp3}, xmmword ptr [{data} + 240]",
        "addps {a0}, {tmp0}",
        "addps {a1}, {tmp1}",
        "addps {a2}, {tmp2}",
        "addps {a3}, {tmp3}",
        "addps {a4}, {tmp4}",
        "addps {a5}, {tmp5}",
        "addps {a6}, {tmp6}",
        "addps {a7}, {tmp7}",
        "add {data}, 256",
        "cmp {data}, {data_end}",
        "jb 2b",
        "movups xmmword ptr [{result}], {a0}",
        "movups xmmword ptr [{result} + 16], {a1}",
        "movups xmmword ptr [{result} + 32], {a2}",
        "movups xmmword ptr [{result} + 48], {a3}",
        "movups xmmword ptr [{result} + 64], {a4}",
        "movups xmmword ptr [{result} + 80], {a5}",
        "movups xmmword ptr [{result} + 96], {a6}",
        "movups xmmword ptr [{result} + 112], {a7}",
        data = in(reg) data,
        data_end = in(reg) data.add(rows * 16),
        result = in(reg) result.as_ptr(),
        a0 = out(xmm_reg) _,
        a1 = out(xmm_reg) _,
        a2 = out(xmm_reg) _,
        a3 = out(xmm_reg) _,
        a4 = out(xmm_reg) _,
        a5 = out(xmm_reg) _,
        a6 = out(xmm_reg) _,
        a7 = out(xmm_reg) _,
        tmp0 = out(xmm_reg) _,
        tmp1 = out(xmm_reg) _,
        tmp2 = out(xmm_reg) _,
        tmp3 = out(xmm_reg) _,
        tmp4 = out(xmm_reg) _,
        tmp5 = out(xmm_reg) _,
        tmp6 = out(xmm_reg) _,
        tmp7 = out(xmm_reg) _,
        options(nostack),
        );
    }
}

#[inline(always)]
fn f32x8_load_fma(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    let mut a0 = unsafe { _mm256_set1_ps(0.0) };
    let mut a1 = unsafe { _mm256_set1_ps(0.0) };
    let mut a2 = unsafe { _mm256_set1_ps(0.0) };
    let mut a3 = unsafe { _mm256_set1_ps(0.0) };
    for j in 0..rows / 4 {
        unsafe {
            let r00 = _mm256_loadu_ps(data.add(j * 4 * 16));
            let r01 = _mm256_loadu_ps(data.add(j * 4 * 16 + 8));
            let r10 = _mm256_loadu_ps(data.add((j * 4 + 1) * 16));
            let r11 = _mm256_loadu_ps(data.add((j * 4 + 1) * 16 + 8));
            let r20 = _mm256_loadu_ps(data.add((j * 4 + 2) * 16));
            let r21 = _mm256_loadu_ps(data.add((j * 4 + 2) * 16 + 8));
            let r30 = _mm256_loadu_ps(data.add((j * 4 + 3) * 16));
            let r31 = _mm256_loadu_ps(data.add((j * 4 + 3) * 16 + 8));
            a0 = _mm256_fmadd_ps(r00, r10, a0);
            a1 = _mm256_fmadd_ps(r01, r11, a1);
            a2 = _mm256_fmadd_ps(r20, r30, a2);
            a3 = _mm256_fmadd_ps(r21, r31, a3);
        }
    }
    unsafe {
        let rp = result.as_mut_ptr();
        _mm256_storeu_ps(rp, a0);
        _mm256_storeu_ps(rp.add(8), a1);
        _mm256_storeu_ps(rp.add(16), a2);
        _mm256_storeu_ps(rp.add(24), a3);
        _mm256_zeroupper();
    }
}

#[inline(always)]
fn f32x8_load_fma_asm(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    assert_eq!(rows % 4, 0);
    unsafe {
        asm!(
        "vxorps {a0}, {a0}, {a0}",
        "vxorps {a1}, {a1}, {a1}",
        "vxorps {a2}, {a2}, {a2}",
        "vxorps {a3}, {a3}, {a3}",
        ".p2align 4",
        "2:",
        "vmovups {tmp0}, ymmword ptr [{data}]",
        "vmovups {tmp1}, ymmword ptr [{data} + 32]",
        "vmovups {tmp2}, ymmword ptr [{data} + 128]",
        "vmovups {tmp3}, ymmword ptr [{data} + 160]",
        "vfmadd231ps {a0}, {tmp0}, ymmword ptr [{data} + 64]",
        "vfmadd231ps {a1}, {tmp1}, ymmword ptr [{data} + 96]",
        "vfmadd231ps {a2}, {tmp2}, ymmword ptr [{data} + 192]",
        "vfmadd231ps {a3}, {tmp3}, ymmword ptr [{data} + 224]",
        "add {data}, 256",
        "cmp {data}, {data_end}",
        "jb 2b",
        "vmovups ymmword ptr [{result}], {a0}",
        "vmovups ymmword ptr [{result} + 32], {a1}",
        "vmovups ymmword ptr [{result} + 64], {a2}",
        "vmovups ymmword ptr [{result} + 96], {a3}",
        "vzeroupper",
        data = in(reg) data,
        data_end = in(reg) data.add(rows * 16),
        result = in(reg) result.as_ptr(),
        a0 = out(ymm_reg) _,
        a1 = out(ymm_reg) _,
        a2 = out(ymm_reg) _,
        a3 = out(ymm_reg) _,
        tmp0 = out(ymm_reg) _,
        tmp1 = out(ymm_reg) _,
        tmp2 = out(ymm_reg) _,
        tmp3 = out(ymm_reg) _,
        options(nostack),
        );
    }
}

// Do load+add with multiple target regs to avoid being latency-bound.
pub fn basic_simd_load_add_benchmark(c: &mut Criterion) {
    const ROWS: &[usize] = &[100, 4000, 1000000];
    for &rows in ROWS {
        let data = vec![1.0f32; rows * 16];
        let data_ptr = data.as_ptr();
        let mut group = c.benchmark_group(format!("basic simd load-add f32 {} rows", rows));
        // Measure throughput in flops, not in Gb.
        group.throughput(criterion::Throughput::Elements(rows as u64 * 16));
        let mut golden_result = [0.0f32; 32];
        scalar_load_add(data_ptr, rows, &mut golden_result);

        let mut result = [0.0f32; 32];
        group.bench_function("scalar rust", |b| {
            result.fill(0.0);
            b.iter(|| scalar_load_add(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });
        // Skip scalar asm code.
        group.bench_function("f32x4 rust", |b| {
            result.fill(0.0);
            b.iter(|| f32x4_load_add(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });
        group.bench_function("f32x4 asm", |b| {
            result.fill(0.0);
            b.iter(|| f32x4_load_add_asm(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });
        if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
            group.bench_function("f32x8 rust", |b| {
                result.fill(0.0);
                b.iter(|| f32x8_load_add(data_ptr, rows, &mut result));
                assert_eq!(result, golden_result);
            });
            group.bench_function("f32x8 asm", |b| {
                result.fill(0.0);
                b.iter(|| f32x8_load_add_asm(data_ptr, rows, &mut result));
                assert_eq!(result, golden_result);
            });
        }
    }
}

#[inline(always)]
fn scalar_load_add(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    let mut acc = [0.0f32; 32];
    for j in 0..rows / 4 {
        for k in 0..16 {
            unsafe {
                *acc.get_unchecked_mut(k) +=
                    *data.add((j * 4) * 16 + k) + *data.add((j * 4 + 1) * 16 + k);
                *acc.get_unchecked_mut(k + 16) +=
                    *data.add((j * 4 + 2) * 16 + k) + *data.add((j * 4 + 3) * 16 + k);
            }
        }
    }
    *result = acc;
}

#[inline(always)]
fn f32x4_load_add(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    assert_eq!(rows % 4, 0);
    let mut a0 = unsafe { _mm_set1_ps(0.0) };
    let mut a1 = unsafe { _mm_set1_ps(0.0) };
    let mut a2 = unsafe { _mm_set1_ps(0.0) };
    let mut a3 = unsafe { _mm_set1_ps(0.0) };
    let mut a4 = unsafe { _mm_set1_ps(0.0) };
    let mut a5 = unsafe { _mm_set1_ps(0.0) };
    let mut a6 = unsafe { _mm_set1_ps(0.0) };
    let mut a7 = unsafe { _mm_set1_ps(0.0) };
    for j in 0..rows / 4 {
        unsafe {
            let r00 = _mm_loadu_ps(data.add(j * 4 * 16));
            let r01 = _mm_loadu_ps(data.add(j * 4 * 16 + 4));
            let r02 = _mm_loadu_ps(data.add(j * 4 * 16 + 8));
            let r03 = _mm_loadu_ps(data.add(j * 4 * 16 + 12));
            let r10 = _mm_loadu_ps(data.add((j * 4 + 1) * 16));
            let r11 = _mm_loadu_ps(data.add((j * 4 + 1) * 16 + 4));
            let r12 = _mm_loadu_ps(data.add((j * 4 + 1) * 16 + 8));
            let r13 = _mm_loadu_ps(data.add((j * 4 + 1) * 16 + 12));
            a0 = _mm_add_ps(a0, _mm_add_ps(r00, r10));
            a1 = _mm_add_ps(a1, _mm_add_ps(r01, r11));
            a2 = _mm_add_ps(a2, _mm_add_ps(r02, r12));
            a3 = _mm_add_ps(a3, _mm_add_ps(r03, r13));
            let r00 = _mm_loadu_ps(data.add((j * 4 + 2) * 16));
            let r01 = _mm_loadu_ps(data.add((j * 4 + 2) * 16 + 4));
            let r02 = _mm_loadu_ps(data.add((j * 4 + 2) * 16 + 8));
            let r03 = _mm_loadu_ps(data.add((j * 4 + 2) * 16 + 12));
            let r10 = _mm_loadu_ps(data.add((j * 4 + 3) * 16));
            let r11 = _mm_loadu_ps(data.add((j * 4 + 3) * 16 + 4));
            let r12 = _mm_loadu_ps(data.add((j * 4 + 3) * 16 + 8));
            let r13 = _mm_loadu_ps(data.add((j * 4 + 3) * 16 + 12));
            a4 = _mm_add_ps(a4, _mm_add_ps(r00, r10));
            a5 = _mm_add_ps(a5, _mm_add_ps(r01, r11));
            a6 = _mm_add_ps(a6, _mm_add_ps(r02, r12));
            a7 = _mm_add_ps(a7, _mm_add_ps(r03, r13));
        }
    }
    unsafe {
        let rp = result.as_mut_ptr();
        _mm_storeu_ps(rp, a0);
        _mm_storeu_ps(rp.add(4), a1);
        _mm_storeu_ps(rp.add(8), a2);
        _mm_storeu_ps(rp.add(12), a3);
        _mm_storeu_ps(rp.add(16), a4);
        _mm_storeu_ps(rp.add(20), a5);
        _mm_storeu_ps(rp.add(24), a6);
        _mm_storeu_ps(rp.add(28), a7);
    }
}

#[inline(always)]
fn f32x4_load_add_asm(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    assert_eq!(rows % 4, 0);
    unsafe {
        asm!(
        "xorps {a0}, {a0}",
        "xorps {a1}, {a1}",
        "xorps {a2}, {a2}",
        "xorps {a3}, {a3}",
        "xorps {a4}, {a4}",
        "xorps {a5}, {a5}",
        "xorps {a6}, {a6}",
        "xorps {a7}, {a7}",
        ".p2align 4",
        "2:",
        "movups {tmp0}, xmmword ptr [{data}]",
        "movups {tmp1}, xmmword ptr [{data} + 16]",
        "movups {tmp2}, xmmword ptr [{data} + 32]",
        "movups {tmp3}, xmmword ptr [{data} + 48]",
        "movups {tmp4}, xmmword ptr [{data} + 128]",
        "movups {tmp5}, xmmword ptr [{data} + 144]",
        "movups {tmp6}, xmmword ptr [{data} + 160]",
        "movups {tmp7}, xmmword ptr [{data} + 176]",
        "addps {tmp0}, xmmword ptr [{data} + 64]",
        "addps {tmp1}, xmmword ptr [{data} + 80]",
        "addps {tmp2}, xmmword ptr [{data} + 96]",
        "addps {tmp3}, xmmword ptr [{data} + 112]",
        "addps {tmp4}, xmmword ptr [{data} + 192]",
        "addps {tmp5}, xmmword ptr [{data} + 208]",
        "addps {tmp6}, xmmword ptr [{data} + 224]",
        "addps {tmp7}, xmmword ptr [{data} + 240]",
        "addps {a0}, {tmp0}",
        "addps {a1}, {tmp1}",
        "addps {a2}, {tmp2}",
        "addps {a3}, {tmp3}",
        "addps {a4}, {tmp4}",
        "addps {a5}, {tmp5}",
        "addps {a6}, {tmp6}",
        "addps {a7}, {tmp7}",
        "add {data}, 256",
        "cmp {data}, {data_end}",
        "jb 2b",
        "movups xmmword ptr [{result}], {a0}",
        "movups xmmword ptr [{result} + 16], {a1}",
        "movups xmmword ptr [{result} + 32], {a2}",
        "movups xmmword ptr [{result} + 48], {a3}",
        "movups xmmword ptr [{result} + 64], {a4}",
        "movups xmmword ptr [{result} + 80], {a5}",
        "movups xmmword ptr [{result} + 96], {a6}",
        "movups xmmword ptr [{result} + 112], {a7}",
        data = in(reg) data,
        data_end = in(reg) data.add(rows * 16),
        result = in(reg) result.as_ptr(),
        a0 = out(xmm_reg) _,
        a1 = out(xmm_reg) _,
        a2 = out(xmm_reg) _,
        a3 = out(xmm_reg) _,
        a4 = out(xmm_reg) _,
        a5 = out(xmm_reg) _,
        a6 = out(xmm_reg) _,
        a7 = out(xmm_reg) _,
        tmp0 = out(xmm_reg) _,
        tmp1 = out(xmm_reg) _,
        tmp2 = out(xmm_reg) _,
        tmp3 = out(xmm_reg) _,
        tmp4 = out(xmm_reg) _,
        tmp5 = out(xmm_reg) _,
        tmp6 = out(xmm_reg) _,
        tmp7 = out(xmm_reg) _,
        options(nostack),
        );
    }
}

#[inline(always)]
fn f32x8_load_add(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    let mut a0 = unsafe { _mm256_set1_ps(0.0) };
    let mut a1 = unsafe { _mm256_set1_ps(0.0) };
    let mut a2 = unsafe { _mm256_set1_ps(0.0) };
    let mut a3 = unsafe { _mm256_set1_ps(0.0) };
    for j in 0..rows / 4 {
        unsafe {
            let r00 = _mm256_loadu_ps(data.add(j * 4 * 16));
            let r01 = _mm256_loadu_ps(data.add(j * 4 * 16 + 8));
            let r10 = _mm256_loadu_ps(data.add((j * 4 + 1) * 16));
            let r11 = _mm256_loadu_ps(data.add((j * 4 + 1) * 16 + 8));
            let r20 = _mm256_loadu_ps(data.add((j * 4 + 2) * 16));
            let r21 = _mm256_loadu_ps(data.add((j * 4 + 2) * 16 + 8));
            let r30 = _mm256_loadu_ps(data.add((j * 4 + 3) * 16));
            let r31 = _mm256_loadu_ps(data.add((j * 4 + 3) * 16 + 8));
            a0 = _mm256_add_ps(a0, _mm256_add_ps(r00, r10));
            a1 = _mm256_add_ps(a1, _mm256_add_ps(r01, r11));
            a2 = _mm256_add_ps(a2, _mm256_add_ps(r20, r30));
            a3 = _mm256_add_ps(a3, _mm256_add_ps(r21, r31));
        }
    }
    unsafe {
        let rp = result.as_mut_ptr();
        _mm256_storeu_ps(rp, a0);
        _mm256_storeu_ps(rp.add(8), a1);
        _mm256_storeu_ps(rp.add(16), a2);
        _mm256_storeu_ps(rp.add(24), a3);
        _mm256_zeroupper();
    }
}

#[inline(always)]
fn f32x8_load_add_asm(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    assert_eq!(rows % 4, 0);
    unsafe {
        asm!(
        "vxorps {a0}, {a0}, {a0}",
        "vxorps {a1}, {a1}, {a1}",
        "vxorps {a2}, {a2}, {a2}",
        "vxorps {a3}, {a3}, {a3}",
        ".p2align 4",
        "2:",
        "vmovups {tmp0}, ymmword ptr [{data}]",
        "vmovups {tmp1}, ymmword ptr [{data} + 32]",
        "vmovups {tmp2}, ymmword ptr [{data} + 128]",
        "vmovups {tmp3}, ymmword ptr [{data} + 160]",
        "vaddps {tmp0}, {tmp0}, ymmword ptr [{data} + 64]",
        "vaddps {tmp1}, {tmp1}, ymmword ptr [{data} + 96]",
        "vaddps {tmp2}, {tmp2}, ymmword ptr [{data} + 192]",
        "vaddps {tmp3}, {tmp3}, ymmword ptr [{data} + 224]",
        "vaddps {a0}, {a0}, {tmp0}",
        "vaddps {a1}, {a1}, {tmp1}",
        "vaddps {a2}, {a2}, {tmp2}",
        "vaddps {a3}, {a3}, {tmp3}",
        "add {data}, 256",
        "cmp {data}, {data_end}",
        "jb 2b",
        "vmovups ymmword ptr [{result}], {a0}",
        "vmovups ymmword ptr [{result} + 32], {a1}",
        "vmovups ymmword ptr [{result} + 64], {a2}",
        "vmovups ymmword ptr [{result} + 96], {a3}",
        "vzeroupper",
        data = in(reg) data,
        data_end = in(reg) data.add(rows * 16),
        result = in(reg) result.as_ptr(),
        a0 = out(ymm_reg) _,
        a1 = out(ymm_reg) _,
        a2 = out(ymm_reg) _,
        a3 = out(ymm_reg) _,
        tmp0 = out(ymm_reg) _,
        tmp1 = out(ymm_reg) _,
        tmp2 = out(ymm_reg) _,
        tmp3 = out(ymm_reg) _,
        options(nostack),
        );
    }
}

criterion_group!(
    basic_simd_benches,
    basic_simd_load_benchmark,
    basic_simd_store_benchmark,
    basic_simd_add_latency_benchmark,
    basic_simd_fma_benchmark,
    basic_simd_load_add_benchmark,
    basic_simd_load_fma_benchmark
);
criterion_main!(basic_simd_benches);
