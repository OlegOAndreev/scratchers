// Benchmark basic SIMD operations.

use std::arch::asm;

use criterion::{Criterion, criterion_group, criterion_main};

pub fn basic_simd_load_benchmark(c: &mut Criterion) {
    basic_simd_load_benchmark_impl::<8>(c);
    basic_simd_load_benchmark_impl::<16>(c);
}

fn basic_simd_load_benchmark_impl<const LOAD: usize>(c: &mut Criterion) {
    const ROWS: &[usize] = &[100, 4000, 1000000];
    for &rows in ROWS {
        for &stride in &[LOAD, LOAD + 8] {
            let data = vec![1.0f32; rows * stride];
            let data_ptr = data.as_ptr();
            let name = format!("basic simd load x{} f32 {} rows, {} stride", LOAD, rows, stride);
            let mut group = c.benchmark_group(name);
            // Measure throughput in elements, not in Gb.
            group.throughput(criterion::Throughput::Elements(rows as u64 * LOAD as u64));
            group.bench_function("scalar asm", |b| b.iter(|| scalar_load_asm::<LOAD>(data_ptr, stride, rows)));
            group.bench_function("f32x4 asm", |b| b.iter(|| f32x4_load_asm::<LOAD>(data_ptr, stride, rows)));
            if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
                group.bench_function("f32x8 asm", |b| b.iter(|| f32x8_load_asm::<LOAD>(data_ptr, stride, rows)));
            }
        }
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
                data = in(reg) data,
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
                data = in(reg) data,
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
                    data = in(reg) data,
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
                    data = in(reg) data,
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
                    data = in(reg) data,
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
                    data = in(reg) data,
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
                    data = in(reg) data,
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
                    data = in(reg) data,
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
                    data = in(reg) data,
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
                    data = in(reg) data,
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
        }
    } else {
        unimplemented!();
    }
}

pub fn basic_simd_store_benchmark(c: &mut Criterion) {
    basic_simd_store_benchmark_impl::<8>(c);
    basic_simd_store_benchmark_impl::<16>(c);
}

fn basic_simd_store_benchmark_impl<const STORE: usize>(_c: &mut Criterion) {
    // const ROWS: &[usize] = &[100, 4000, 1000000];
    // for &rows in ROWS {
    //     for &stride in &[STORE, STORE + 8] {
    //         let data = vec![1.0f32; rows * stride];
    //         let data_ptr = data.as_ptr();
    //         let name = format!("basic simd store x{} f32 {} rows, {} stride", STORE, rows, stride);
    //         let mut group = c.benchmark_group(name);
    //         // Measure throughput in elements, not in Gb.
    //         group.throughput(criterion::Throughput::Elements(rows as u64 * STORE as u64));
    //         group.bench_function("scalar", |b| b.iter(|| scalar_store::<STORE>(data_ptr, stride, rows)));
    //         group.bench_function("scalar asm", |b| b.iter(|| scalar_store_asm::<STORE>(data_ptr, stride, rows)));
    //         group.bench_function("f32x4", |b| b.iter(|| f32x4_store::<STORE>(data_ptr, stride, rows)));
    //         group.bench_function("f32x4 asm", |b| b.iter(|| f32x4_store_asm::<STORE>(data_ptr, stride, rows)));
    //         if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
    //             group.bench_function("f32x8", |b| b.iter(|| f32x8_store::<STORE>(data_ptr, stride, rows)));
    //             group.bench_function("f32x8 asm", |b| b.iter(|| f32x8_store_asm::<STORE>(data_ptr, stride, rows)));
    //         }
    //     }
    // }
}

criterion_group!(basic_simd_benches, basic_simd_load_benchmark, basic_simd_store_benchmark);
criterion_main!(basic_simd_benches);
