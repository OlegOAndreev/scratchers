// Benchmark basic SIMD operations.

use std::arch::asm;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
    _mm256_zeroupper,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps};

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
    group.throughput(criterion::Throughput::Bytes(rows as u64 * LOAD as u64 * 4));

    let data_ptr = data.as_ptr();
    let aligned_ptr = unsafe {
        data_ptr
            .add(data_ptr.align_offset(8)) // Align for f32x8
            .add(align)
    };

    group.bench_function("scalar asm", |b| {
        b.iter(|| scalar_load_asm::<LOAD>(aligned_ptr, stride, rows))
    });

    #[cfg(target_arch = "aarch64")]
    group.bench_function("scalar asm ldr", |b| {
        b.iter(|| scalar_load_asm_ldr::<LOAD>(aligned_ptr, stride, rows))
    });

    group.bench_function("f32x4 asm", |b| {
        b.iter(|| f32x4_load_asm::<LOAD>(aligned_ptr, stride, rows))
    });

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    group.bench_function("f32x8 asm", |b| {
        b.iter(|| f32x8_load_asm::<LOAD>(aligned_ptr, stride, rows))
    });
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn scalar_load_asm<const LOAD: usize>(data: *const f32, stride: usize, rows: usize) {
    assert_eq!(rows % 2, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if LOAD == 8 {
        unsafe {
            asm!(
            ".p2align 5",
            "2:",
            "ldp {tmp0:s}, {tmp1:s}, [{data}]",
            "ldp {tmp2:s}, {tmp3:s}, [{data}, #8]",
            "ldp {tmp4:s}, {tmp5:s}, [{data}, #16]",
            "ldp {tmp6:s}, {tmp7:s}, [{data}, #24]",
            "add {data}, {data}, {stride}",
            "ldp {tmp0:s}, {tmp1:s}, [{data}]",
            "ldp {tmp2:s}, {tmp3:s}, [{data}, #8]",
            "ldp {tmp4:s}, {tmp5:s}, [{data}, #16]",
            "ldp {tmp6:s}, {tmp7:s}, [{data}, #24]",
            "add {data}, {data}, {stride}",
            "cmp {data}, {data_end}",
            "b.lo 2b",
            data = inout(reg) data => _,
            data_end = in(reg) data_end,
            stride = in(reg) stride * 4,
            tmp0 = out(vreg) _,
            tmp1 = out(vreg) _,
            tmp2 = out(vreg) _,
            tmp3 = out(vreg) _,
            tmp4 = out(vreg) _,
            tmp5 = out(vreg) _,
            tmp6 = out(vreg) _,
            tmp7 = out(vreg) _,
            options(nostack),
            );
        }
    } else if LOAD == 16 {
        unsafe {
            asm!(
            ".p2align 5",
            "2:",
            "ldp {tmp0:s}, {tmp1:s}, [{data}]",
            "ldp {tmp2:s}, {tmp3:s}, [{data}, #8]",
            "ldp {tmp4:s}, {tmp5:s}, [{data}, #16]",
            "ldp {tmp6:s}, {tmp7:s}, [{data}, #24]",
            "ldp {tmp8:s}, {tmp9:s}, [{data}, #32]",
            "ldp {tmp10:s}, {tmp11:s}, [{data}, #40]",
            "ldp {tmp12:s}, {tmp13:s}, [{data}, #48]",
            "ldp {tmp14:s}, {tmp15:s}, [{data}, #56]",
            "add {data}, {data}, {stride}",
            "cmp {data}, {data_end}",
            "b.lo 2b",
            data = inout(reg) data => _,
            data_end = in(reg) data_end,
            stride = in(reg) stride * 4,
            tmp0 = out(vreg) _,
            tmp1 = out(vreg) _,
            tmp2 = out(vreg) _,
            tmp3 = out(vreg) _,
            tmp4 = out(vreg) _,
            tmp5 = out(vreg) _,
            tmp6 = out(vreg) _,
            tmp7 = out(vreg) _,
            tmp8 = out(vreg) _,
            tmp9 = out(vreg) _,
            tmp10 = out(vreg) _,
            tmp11 = out(vreg) _,
            tmp12 = out(vreg) _,
            tmp13 = out(vreg) _,
            tmp14 = out(vreg) _,
            tmp15 = out(vreg) _,
            options(nostack),
            );
        }
    } else {
        unimplemented!();
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn scalar_load_asm_ldr<const LOAD: usize>(data: *const f32, stride: usize, rows: usize) {
    assert_eq!(rows % 2, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if LOAD == 8 {
        unsafe {
            asm!(
            ".p2align 5",
            "2:",
            "ldr {tmp0:s}, [{data}]",
            "ldr {tmp1:s}, [{data}, #4]",
            "ldr {tmp2:s}, [{data}, #8]",
            "ldr {tmp3:s}, [{data}, #12]",
            "ldr {tmp4:s}, [{data}, #16]",
            "ldr {tmp5:s}, [{data}, #20]",
            "ldr {tmp6:s}, [{data}, #24]",
            "ldr {tmp7:s}, [{data}, #28]",
            "add {data}, {data}, {stride}",
            "ldr {tmp0:s}, [{data}]",
            "ldr {tmp1:s}, [{data}, #4]",
            "ldr {tmp2:s}, [{data}, #8]",
            "ldr {tmp3:s}, [{data}, #12]",
            "ldr {tmp4:s}, [{data}, #16]",
            "ldr {tmp5:s}, [{data}, #20]",
            "ldr {tmp6:s}, [{data}, #24]",
            "ldr {tmp7:s}, [{data}, #28]",
            "add {data}, {data}, {stride}",
            "cmp {data}, {data_end}",
            "b.lo 2b",
            data = inout(reg) data => _,
            data_end = in(reg) data_end,
            stride = in(reg) stride * 4,
            tmp0 = out(vreg) _,
            tmp1 = out(vreg) _,
            tmp2 = out(vreg) _,
            tmp3 = out(vreg) _,
            tmp4 = out(vreg) _,
            tmp5 = out(vreg) _,
            tmp6 = out(vreg) _,
            tmp7 = out(vreg) _,
            options(nostack),
            );
        }
    } else if LOAD == 16 {
        unsafe {
            asm!(
            ".p2align 5",
            "2:",
            "ldr {tmp0:s}, [{data}]",
            "ldr {tmp1:s}, [{data}, #4]",
            "ldr {tmp2:s}, [{data}, #8]",
            "ldr {tmp3:s}, [{data}, #12]",
            "ldr {tmp4:s}, [{data}, #16]",
            "ldr {tmp5:s}, [{data}, #20]",
            "ldr {tmp6:s}, [{data}, #24]",
            "ldr {tmp7:s}, [{data}, #28]",
            "ldr {tmp8:s}, [{data}, #32]",
            "ldr {tmp9:s}, [{data}, #36]",
            "ldr {tmp10:s}, [{data}, #40]",
            "ldr {tmp11:s}, [{data}, #44]",
            "ldr {tmp12:s}, [{data}, #48]",
            "ldr {tmp13:s}, [{data}, #52]",
            "ldr {tmp14:s}, [{data}, #56]",
            "ldr {tmp15:s}, [{data}, #60]",
            "add {data}, {data}, {stride}",
            "cmp {data}, {data_end}",
            "b.lo 2b",
            data = inout(reg) data => _,
            data_end = in(reg) data_end,
            stride = in(reg) stride * 4,
            tmp0 = out(vreg) _,
            tmp1 = out(vreg) _,
            tmp2 = out(vreg) _,
            tmp3 = out(vreg) _,
            tmp4 = out(vreg) _,
            tmp5 = out(vreg) _,
            tmp6 = out(vreg) _,
            tmp7 = out(vreg) _,
            tmp8 = out(vreg) _,
            tmp9 = out(vreg) _,
            tmp10 = out(vreg) _,
            tmp11 = out(vreg) _,
            tmp12 = out(vreg) _,
            tmp13 = out(vreg) _,
            tmp14 = out(vreg) _,
            tmp15 = out(vreg) _,
            options(nostack),
            );
        }
    } else {
        unimplemented!();
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn scalar_load_asm<const LOAD: usize>(data: *const f32, stride: usize, rows: usize) {
    assert_eq!(rows % 2, 0);
    let data_end = unsafe { data.add(stride * rows) };
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
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f32x4_load_asm<const LOAD: usize>(data: *const f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
    if LOAD == 8 {
        if stride == 8 {
            unsafe {
                asm!(
                ".p2align 5",
                "2:",
                "ld1 {{ {tmp0}.4s, {tmp1}.4s, {tmp2}.4s, {tmp3}.4s }}, [{data}], #64",
                "ld1 {{ {tmp4}.4s, {tmp5}.4s, {tmp6}.4s, {tmp7}.4s }}, [{data}], #64",
                "cmp {data}, {data_end}",
                "b.lo 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                tmp0 = out(vreg) _,
                tmp1 = out(vreg) _,
                tmp2 = out(vreg) _,
                tmp3 = out(vreg) _,
                tmp4 = out(vreg) _,
                tmp5 = out(vreg) _,
                tmp6 = out(vreg) _,
                tmp7 = out(vreg) _,
                options(nostack)
                );
            }
        } else {
            unsafe {
                asm!(
                ".p2align 5",
                "2:",
                "ld1 {{ {tmp0}.4s, {tmp1}.4s }}, [{data}], {stride}",
                "ld1 {{ {tmp2}.4s, {tmp3}.4s }}, [{data}], {stride}",
                "ld1 {{ {tmp4}.4s, {tmp5}.4s }}, [{data}], {stride}",
                "ld1 {{ {tmp6}.4s, {tmp7}.4s }}, [{data}], {stride}",
                "cmp {data}, {data_end}",
                "b.lo 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                stride = in(reg) stride * 4,
                tmp0 = out(vreg) _,
                tmp1 = out(vreg) _,
                tmp2 = out(vreg) _,
                tmp3 = out(vreg) _,
                tmp4 = out(vreg) _,
                tmp5 = out(vreg) _,
                tmp6 = out(vreg) _,
                tmp7 = out(vreg) _,
                options(nostack)
                );
            }
        }
    } else if LOAD == 16 {
        if stride == 16 {
            unsafe {
                asm!(
                ".p2align 5",
                "2:",
                "ld1 {{ {tmp0}.4s, {tmp1}.4s, {tmp2}.4s, {tmp3}.4s }}, [{data}], #64",
                "ld1 {{ {tmp4}.4s, {tmp5}.4s, {tmp6}.4s, {tmp7}.4s }}, [{data}], #64",
                "ld1 {{ {tmp0}.4s, {tmp1}.4s, {tmp2}.4s, {tmp3}.4s }}, [{data}], #64",
                "ld1 {{ {tmp4}.4s, {tmp5}.4s, {tmp6}.4s, {tmp7}.4s }}, [{data}], #64",
                "cmp {data}, {data_end}",
                "b.lo 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                tmp0 = out(vreg) _,
                tmp1 = out(vreg) _,
                tmp2 = out(vreg) _,
                tmp3 = out(vreg) _,
                tmp4 = out(vreg) _,
                tmp5 = out(vreg) _,
                tmp6 = out(vreg) _,
                tmp7 = out(vreg) _,
                options(nostack)
                );
            }
        } else {
            unsafe {
                asm!(
                ".p2align 5",
                "2:",
                "ld1 {{ {tmp0}.4s, {tmp1}.4s, {tmp2}.4s, {tmp3}.4s }}, [{data}], {stride}",
                "ld1 {{ {tmp4}.4s, {tmp5}.4s, {tmp6}.4s, {tmp7}.4s }}, [{data}], {stride}",
                "ld1 {{ {tmp0}.4s, {tmp1}.4s, {tmp2}.4s, {tmp3}.4s }}, [{data}], {stride}",
                "ld1 {{ {tmp4}.4s, {tmp5}.4s, {tmp6}.4s, {tmp7}.4s }}, [{data}], {stride}",
                "cmp {data}, {data_end}",
                "b.lo 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                stride = in(reg) stride * 4,
                tmp0 = out(vreg) _,
                tmp1 = out(vreg) _,
                tmp2 = out(vreg) _,
                tmp3 = out(vreg) _,
                tmp4 = out(vreg) _,
                tmp5 = out(vreg) _,
                tmp6 = out(vreg) _,
                tmp7 = out(vreg) _,
                options(nostack)
                );
            }
        }
    } else {
        unimplemented!();
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn f32x4_load_asm<const LOAD: usize>(data: *const f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
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
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn f32x8_load_asm<const LOAD: usize>(data: *const f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
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
    group.throughput(criterion::Throughput::Bytes(rows as u64 * STORE as u64 * 4));

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

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    group.bench_function("f32x8 rust", |b| {
        data.fill(0.0);
        b.iter(|| f32x8_store::<STORE>(aligned_ptr, stride, rows));
        assert_one(&data[to_align + align..], STORE, rows, stride);
    });

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    group.bench_function("f32x8 asm", |b| {
        data.fill(0.0);
        b.iter(|| f32x8_store_asm::<STORE>(aligned_ptr, stride, rows));
        assert_one(&data[to_align + align..], STORE, rows, stride);
    });
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn scalar_store_asm<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 2, 0);
    let data_end = unsafe { data.add(stride * rows) };
    let one = unsafe { vdupq_n_f32(1.0) };
    if STORE == 8 {
        unsafe {
            asm!(
            ".p2align 5",
            "2:",
            "stp {one:s}, {one:s}, [{data}]",
            "stp {one:s}, {one:s}, [{data}, #8]",
            "stp {one:s}, {one:s}, [{data}, #16]",
            "stp {one:s}, {one:s}, [{data}, #24]",
            "add {data}, {data}, {stride}",
            "stp {one:s}, {one:s}, [{data}]",
            "stp {one:s}, {one:s}, [{data}, #8]",
            "stp {one:s}, {one:s}, [{data}, #16]",
            "stp {one:s}, {one:s}, [{data}, #24]",
            "add {data}, {data}, {stride}",
            "cmp {data}, {data_end}",
            "b.lo 2b",
            data = inout(reg) data => _,
            data_end = in(reg) data_end,
            stride = in(reg) stride * 4,
            one = in(vreg) one,
            options(nostack),
            );
        }
    } else if STORE == 16 {
        unsafe {
            asm!(
            ".p2align 4",
            "2:",
            "stp {one:s}, {one:s}, [{data}]",
            "stp {one:s}, {one:s}, [{data}, #8]",
            "stp {one:s}, {one:s}, [{data}, #16]",
            "stp {one:s}, {one:s}, [{data}, #24]",
            "stp {one:s}, {one:s}, [{data}, #32]",
            "stp {one:s}, {one:s}, [{data}, #40]",
            "stp {one:s}, {one:s}, [{data}, #48]",
            "stp {one:s}, {one:s}, [{data}, #56]",
            "add {data}, {data}, {stride}",
            "cmp {data}, {data_end}",
            "b.lo 2b",
            data = inout(reg) data => _,
            data_end = in(reg) data_end,
            stride = in(reg) stride * 4,
            one = in(vreg) one,
            options(nostack),
            );
        }
    } else {
        unimplemented!();
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn scalar_store_asm<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 2, 0);
    let data_end = unsafe { data.add(stride * rows) };
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
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f32x4_store<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let one = unsafe { vdupq_n_f32(1.0) };
    if STORE == 8 {
        if stride == 8 {
            let mut i = 0;
            while i < rows {
                unsafe {
                    let ptr = data.add(i * 8);
                    vst1q_f32(ptr, one);
                    vst1q_f32(ptr.add(4), one);
                    vst1q_f32(ptr.add(8), one);
                    vst1q_f32(ptr.add(12), one);
                    vst1q_f32(ptr.add(16), one);
                    vst1q_f32(ptr.add(20), one);
                    vst1q_f32(ptr.add(24), one);
                    vst1q_f32(ptr.add(28), one);
                    i += 4;
                }
            }
        } else {
            let mut i = 0;
            while i < rows {
                unsafe {
                    let ptr = data.add(i * stride);
                    vst1q_f32(ptr, one);
                    vst1q_f32(ptr.add(4), one);
                    vst1q_f32(ptr.add(stride), one);
                    vst1q_f32(ptr.add(stride + 4), one);
                    vst1q_f32(ptr.add(stride * 2), one);
                    vst1q_f32(ptr.add(stride * 2 + 4), one);
                    vst1q_f32(ptr.add(stride * 3), one);
                    vst1q_f32(ptr.add(stride * 3 + 4), one);
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
                    vst1q_f32(ptr, one);
                    vst1q_f32(ptr.add(4), one);
                    vst1q_f32(ptr.add(8), one);
                    vst1q_f32(ptr.add(12), one);
                    vst1q_f32(ptr.add(16), one);
                    vst1q_f32(ptr.add(20), one);
                    vst1q_f32(ptr.add(24), one);
                    vst1q_f32(ptr.add(28), one);
                    vst1q_f32(ptr.add(32), one);
                    vst1q_f32(ptr.add(36), one);
                    vst1q_f32(ptr.add(40), one);
                    vst1q_f32(ptr.add(44), one);
                    vst1q_f32(ptr.add(48), one);
                    vst1q_f32(ptr.add(52), one);
                    vst1q_f32(ptr.add(56), one);
                    vst1q_f32(ptr.add(60), one);
                    i += 4;
                }
            }
        } else {
            let mut i = 0;
            while i < rows {
                unsafe {
                    let ptr = data.add(i * stride);
                    vst1q_f32(ptr, one);
                    vst1q_f32(ptr.add(4), one);
                    vst1q_f32(ptr.add(8), one);
                    vst1q_f32(ptr.add(12), one);
                    vst1q_f32(ptr.add(stride), one);
                    vst1q_f32(ptr.add(stride + 4), one);
                    vst1q_f32(ptr.add(stride + 8), one);
                    vst1q_f32(ptr.add(stride + 12), one);
                    vst1q_f32(ptr.add(stride * 2), one);
                    vst1q_f32(ptr.add(stride * 2 + 4), one);
                    vst1q_f32(ptr.add(stride * 2 + 8), one);
                    vst1q_f32(ptr.add(stride * 2 + 12), one);
                    vst1q_f32(ptr.add(stride * 3), one);
                    vst1q_f32(ptr.add(stride * 3 + 4), one);
                    vst1q_f32(ptr.add(stride * 3 + 8), one);
                    vst1q_f32(ptr.add(stride * 3 + 12), one);
                    i += 4;
                }
            }
        }
    } else {
        unimplemented!();
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn f32x4_store<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
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
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f32x4_store_asm<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };

    let one = unsafe { vdupq_n_f32(1.0) };
    if STORE == 8 {
        if stride == 8 {
            unsafe {
                asm!(
                ".p2align 5",
                "2:",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], #64",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], #64",
                "cmp {data}, {data_end}",
                "b.lo 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                // Rust inline assembly does not allow allocating sequential registers here and LLVM does not seem
                // to allow non-sequential registers for st1 one way or another. That's why we have to use explicitly
                // chosen registers instead of register classes.
                in("v4") one,
                in("v5") one,
                in("v6") one,
                in("v7") one,
                options(nostack)
                );
            }
        } else {
            unsafe {
                asm!(
                ".p2align 5",
                "2:",
                "stp q4, q5, [{data}]",
                "add {data}, {data}, {stride}",
                "stp q4, q5, [{data}]",
                "add {data}, {data}, {stride}",
                "stp q4, q5, [{data}]",
                "add {data}, {data}, {stride}",
                "stp q4, q5, [{data}]",
                "add {data}, {data}, {stride}",
                "cmp {data}, {data_end}",
                "b.lo 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                stride = in(reg) stride * 4,
                in("v4") one,
                in("v5") one,
                options(nostack)
                );
            }
        }
    } else if STORE == 16 {
        if stride == 16 {
            unsafe {
                asm!(
                ".p2align 5",
                "2:",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], #64",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], #64",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], #64",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], #64",
                "cmp {data}, {data_end}",
                "b.lo 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                in("v4") one,
                in("v5") one,
                in("v6") one,
                in("v7") one,
                options(nostack)
                );
            }
        } else {
            unsafe {
                asm!(
                ".p2align 5",
                "2:",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], {stride}",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], {stride}",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], {stride}",
                "st1 {{ v4.4s, v5.4s, v6.4s, v7.4s }}, [{data}], {stride}",
                "cmp {data}, {data_end}",
                "b.lo 2b",
                data = inout(reg) data => _,
                data_end = in(reg) data_end,
                stride = in(reg) stride * 4,
                in("v4") one,
                in("v5") one,
                in("v6") one,
                in("v7") one,
                options(nostack)
                );
            }
        }
    } else {
        unimplemented!();
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn f32x4_store_asm<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
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
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn f32x8_store<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
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
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn f32x8_store_asm<const STORE: usize>(data: *mut f32, stride: usize, rows: usize) {
    assert_eq!(rows % 4, 0);
    let data_end = unsafe { data.add(stride * rows) };
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

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    group.bench_function("f32x8 asm", |b| {
        result.fill(0.0);
        b.iter(|| f32x8_add_latency_asm(ITERATIONS, &arr, &brr, &mut result));
        assert_eq!(result, golden_result);
    });
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f32x4_add_latency_asm(iterations: usize, a: &[f32; 8], b: &[f32; 8], result: &mut [f32; 8]) {
    assert_eq!(iterations % 4, 0);
    unsafe {
        asm!(
        "ldp {a0:q}, {a1:q}, [{a}]",
        "ldp {tmp0:q}, {tmp1:q}, [{b}]",
        ".p2align 5",
        "2:",
        "fadd {a0}.4s, {a0}.4s, {tmp0}.4s",
        "fadd {a1}.4s, {a1}.4s, {tmp1}.4s",
        "fadd {a0}.4s, {a0}.4s, {tmp0}.4s",
        "fadd {a1}.4s, {a1}.4s, {tmp1}.4s",
        "fadd {a0}.4s, {a0}.4s, {tmp0}.4s",
        "fadd {a1}.4s, {a1}.4s, {tmp1}.4s",
        "fadd {a0}.4s, {a0}.4s, {tmp0}.4s",
        "fadd {a1}.4s, {a1}.4s, {tmp1}.4s",
        "subs {iterations}, {iterations}, #1",
        "b.ne 2b",
        "stp {a0:q}, {a1:q}, [{result}]",
        a = in(reg) a.as_ptr(),
        b = in(reg) b.as_ptr(),
        result = in(reg) result.as_ptr(),
        iterations = inout(reg) iterations / 4 => _,
        a0 = out(vreg) _,
        a1 = out(vreg) _,
        tmp0 = out(vreg) _,
        tmp1 = out(vreg) _,
        options(nostack),
        );
    }
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    group.bench_function("f32x8 rust", |b| {
        result.fill(0.0);
        b.iter(|| f32x8_fma(ITERATIONS, &arr, &brr, &crr, &drr, &err, &mut result));
        assert_eq!(result, golden_result);
    });

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    group.bench_function("f32x8 asm", |b| {
        result.fill(0.0);
        b.iter(|| f32x8_fma_asm(ITERATIONS, &arr, &brr, &crr, &drr, &err, &mut result));
        assert_eq!(result, golden_result);
    });
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

#[cfg(target_arch = "aarch64")]
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
    let ap = a.as_ptr();
    let mut a0 = unsafe { vld1q_f32(ap) };
    let mut a1 = unsafe { vld1q_f32(ap.add(4)) };
    let mut a2 = unsafe { vld1q_f32(ap.add(8)) };
    let mut a3 = unsafe { vld1q_f32(ap.add(12)) };
    let mut a4 = unsafe { vld1q_f32(ap) };
    let mut a5 = unsafe { vld1q_f32(ap.add(4)) };
    let mut a6 = unsafe { vld1q_f32(ap.add(8)) };
    let mut a7 = unsafe { vld1q_f32(ap.add(12)) };
    let bp = b.as_ptr();
    let b0 = unsafe { vld1q_f32(bp) };
    let b1 = unsafe { vld1q_f32(bp.add(4)) };
    let b2 = unsafe { vld1q_f32(bp.add(8)) };
    let b3 = unsafe { vld1q_f32(bp.add(12)) };
    let cp = c.as_ptr();
    let c0 = unsafe { vld1q_f32(cp) };
    let c1 = unsafe { vld1q_f32(cp.add(4)) };
    let c2 = unsafe { vld1q_f32(cp.add(8)) };
    let c3 = unsafe { vld1q_f32(cp.add(12)) };
    let dp = d.as_ptr();
    let d0 = unsafe { vld1q_f32(dp) };
    let d1 = unsafe { vld1q_f32(dp.add(4)) };
    let d2 = unsafe { vld1q_f32(dp.add(8)) };
    let d3 = unsafe { vld1q_f32(dp.add(12)) };
    let ep = e.as_ptr();
    let e0 = unsafe { vld1q_f32(ep) };
    let e1 = unsafe { vld1q_f32(ep.add(4)) };
    let e2 = unsafe { vld1q_f32(ep.add(8)) };
    let e3 = unsafe { vld1q_f32(ep.add(12)) };
    for _ in 0..iterations {
        unsafe {
            a0 = vfmaq_f32(a0, b0, c0);
            a1 = vfmaq_f32(a1, b1, c1);
            a2 = vfmaq_f32(a2, b2, c2);
            a3 = vfmaq_f32(a3, b3, c3);
            a4 = vfmaq_f32(a4, d0, e0);
            a5 = vfmaq_f32(a5, d1, e1);
            a6 = vfmaq_f32(a6, d2, e2);
            a7 = vfmaq_f32(a7, d3, e3);
        }
    }
    unsafe {
        let rp = result.as_mut_ptr();
        vst1q_f32(rp, a0);
        vst1q_f32(rp.add(4), a1);
        vst1q_f32(rp.add(8), a2);
        vst1q_f32(rp.add(12), a3);
        vst1q_f32(rp.add(16), a4);
        vst1q_f32(rp.add(20), a5);
        vst1q_f32(rp.add(24), a6);
        vst1q_f32(rp.add(28), a7);
    }
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "aarch64")]
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
        "ld1 {{ {a0}.4s, {a1}.4s, {a2}.4s, {a3}.4s }}, [{a}]",
        "ld1 {{ {a4}.4s, {a5}.4s, {a6}.4s, {a7}.4s }}, [{a}]",
        "ld1 {{ {b0}.4s, {b1}.4s, {b2}.4s, {b3}.4s }}, [{b}]",
        "ld1 {{ {c0}.4s, {c1}.4s, {c2}.4s, {c3}.4s }}, [{c}]",
        "ld1 {{ {d0}.4s, {d1}.4s, {d2}.4s, {d3}.4s }}, [{d}]",
        "ld1 {{ {e0}.4s, {e1}.4s, {e2}.4s, {e3}.4s }}, [{e}]",
        ".p2align 5",
        "2:",
        "fmla {a0}.4s, {b0}.4s, {c0}.4s",
        "fmla {a1}.4s, {b1}.4s, {c1}.4s",
        "fmla {a2}.4s, {b2}.4s, {c2}.4s",
        "fmla {a3}.4s, {b3}.4s, {c3}.4s",
        "fmla {a4}.4s, {d0}.4s, {e0}.4s",
        "fmla {a5}.4s, {d1}.4s, {e1}.4s",
        "fmla {a6}.4s, {d2}.4s, {e2}.4s",
        "fmla {a7}.4s, {d3}.4s, {e3}.4s",
        "subs {iterations}, {iterations}, #1",
        "b.ne 2b",
        "stp {a0:q}, {a1:q}, [{result}]",
        "stp {a2:q}, {a3:q}, [{result}, #32]",
        "stp {a4:q}, {a5:q}, [{result}, #64]",
        "stp {a6:q}, {a7:q}, [{result}, #96]",
        a = in(reg) a.as_ptr(),
        b = in(reg) b.as_ptr(),
        c = in(reg) c.as_ptr(),
        d = in(reg) d.as_ptr(),
        e = in(reg) e.as_ptr(),
        result = in(reg) result.as_ptr(),
        iterations = inout(reg) iterations => _,
        a0 = out(vreg) _,
        a1 = out(vreg) _,
        a2 = out(vreg) _,
        a3 = out(vreg) _,
        a4 = out(vreg) _,
        a5 = out(vreg) _,
        a6 = out(vreg) _,
        a7 = out(vreg) _,
        b0 = out(vreg) _,
        b1 = out(vreg) _,
        b2 = out(vreg) _,
        b3 = out(vreg) _,
        c0 = out(vreg) _,
        c1 = out(vreg) _,
        c2 = out(vreg) _,
        c3 = out(vreg) _,
        d0 = out(vreg) _,
        d1 = out(vreg) _,
        d2 = out(vreg) _,
        d3 = out(vreg) _,
        e0 = out(vreg) _,
        e1 = out(vreg) _,
        e2 = out(vreg) _,
        e3 = out(vreg) _,
        options(nostack),
        );
    }
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        group.bench_function("f32x8 rust", |b| {
            result.fill(0.0);
            b.iter(|| f32x8_load_fma(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        group.bench_function("f32x8 asm", |b| {
            result.fill(0.0);
            b.iter(|| f32x8_load_fma_asm(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f32x4_load_fma(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    assert_eq!(rows % 4, 0);
    let mut a0 = unsafe { vdupq_n_f32(0.0) };
    let mut a1 = unsafe { vdupq_n_f32(0.0) };
    let mut a2 = unsafe { vdupq_n_f32(0.0) };
    let mut a3 = unsafe { vdupq_n_f32(0.0) };
    let mut a4 = unsafe { vdupq_n_f32(0.0) };
    let mut a5 = unsafe { vdupq_n_f32(0.0) };
    let mut a6 = unsafe { vdupq_n_f32(0.0) };
    let mut a7 = unsafe { vdupq_n_f32(0.0) };
    for j in 0..rows / 4 {
        unsafe {
            let r00 = vld1q_f32(data.add(j * 4 * 16));
            let r01 = vld1q_f32(data.add(j * 4 * 16 + 4));
            let r02 = vld1q_f32(data.add(j * 4 * 16 + 8));
            let r03 = vld1q_f32(data.add(j * 4 * 16 + 12));
            let r10 = vld1q_f32(data.add((j * 4 + 1) * 16));
            let r11 = vld1q_f32(data.add((j * 4 + 1) * 16 + 4));
            let r12 = vld1q_f32(data.add((j * 4 + 1) * 16 + 8));
            let r13 = vld1q_f32(data.add((j * 4 + 1) * 16 + 12));
            a0 = vfmaq_f32(a0, r00, r10);
            a1 = vfmaq_f32(a1, r01, r11);
            a2 = vfmaq_f32(a2, r02, r12);
            a3 = vfmaq_f32(a3, r03, r13);
            let r00 = vld1q_f32(data.add((j * 4 + 2) * 16));
            let r01 = vld1q_f32(data.add((j * 4 + 2) * 16 + 4));
            let r02 = vld1q_f32(data.add((j * 4 + 2) * 16 + 8));
            let r03 = vld1q_f32(data.add((j * 4 + 2) * 16 + 12));
            let r10 = vld1q_f32(data.add((j * 4 + 3) * 16));
            let r11 = vld1q_f32(data.add((j * 4 + 3) * 16 + 4));
            let r12 = vld1q_f32(data.add((j * 4 + 3) * 16 + 8));
            let r13 = vld1q_f32(data.add((j * 4 + 3) * 16 + 12));
            a4 = vfmaq_f32(a4, r00, r10);
            a5 = vfmaq_f32(a5, r01, r11);
            a6 = vfmaq_f32(a6, r02, r12);
            a7 = vfmaq_f32(a7, r03, r13);
        }
    }
    unsafe {
        let rp = result.as_mut_ptr();
        vst1q_f32(rp, a0);
        vst1q_f32(rp.add(4), a1);
        vst1q_f32(rp.add(8), a2);
        vst1q_f32(rp.add(12), a3);
        vst1q_f32(rp.add(16), a4);
        vst1q_f32(rp.add(20), a5);
        vst1q_f32(rp.add(24), a6);
        vst1q_f32(rp.add(28), a7);
    }
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "aarch64")]
fn f32x4_load_fma_asm(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    unsafe {
        asm!(
        "eor {a0}.16b, {a0}.16b, {a0}.16b",
        "eor {a1}.16b, {a1}.16b, {a1}.16b",
        "eor {a2}.16b, {a2}.16b, {a2}.16b",
        "eor {a3}.16b, {a3}.16b, {a3}.16b",
        "eor {a4}.16b, {a4}.16b, {a4}.16b",
        "eor {a5}.16b, {a5}.16b, {a5}.16b",
        "eor {a6}.16b, {a6}.16b, {a6}.16b",
        "eor {a7}.16b, {a7}.16b, {a7}.16b",
        ".p2align 5",
        "2:",
        "ld1 {{ {tmp0}.4s, {tmp1}.4s, {tmp2}.4s, {tmp3}.4s }}, [{data}], #64",
        "ld1 {{ {tmp4}.4s, {tmp5}.4s, {tmp6}.4s, {tmp7}.4s }}, [{data}], #64",
        "fmla {a0}.4s, {tmp0}.4s, {tmp4}.4s",
        "fmla {a1}.4s, {tmp1}.4s, {tmp5}.4s",
        "fmla {a2}.4s, {tmp2}.4s, {tmp6}.4s",
        "fmla {a3}.4s, {tmp3}.4s, {tmp7}.4s",
        "ld1 {{ {tmp8}.4s, {tmp9}.4s, {tmp10}.4s, {tmp11}.4s }}, [{data}], #64",
        "ld1 {{ {tmp12}.4s, {tmp13}.4s, {tmp14}.4s, {tmp15}.4s }}, [{data}], #64",
        "fmla {a4}.4s, {tmp8}.4s, {tmp12}.4s",
        "fmla {a5}.4s, {tmp9}.4s, {tmp13}.4s",
        "fmla {a6}.4s, {tmp10}.4s, {tmp14}.4s",
        "fmla {a7}.4s, {tmp11}.4s, {tmp15}.4s",
        "cmp {data}, {data_end}",
        "b.lo 2b",
        "stp {a0:q}, {a1:q}, [{result}]",
        "stp {a2:q}, {a3:q}, [{result}, #32]",
        "stp {a4:q}, {a5:q}, [{result}, #64]",
        "stp {a6:q}, {a7:q}, [{result}, #96]",
        data = in(reg) data,
        data_end = in(reg) data.add(rows * 16),
        result = in(reg) result.as_ptr(),
        a0 = out(vreg) _,
        a1 = out(vreg) _,
        a2 = out(vreg) _,
        a3 = out(vreg) _,
        a4 = out(vreg) _,
        a5 = out(vreg) _,
        a6 = out(vreg) _,
        a7 = out(vreg) _,
        tmp0 = out(vreg) _,
        tmp1 = out(vreg) _,
        tmp2 = out(vreg) _,
        tmp3 = out(vreg) _,
        tmp4 = out(vreg) _,
        tmp5 = out(vreg) _,
        tmp6 = out(vreg) _,
        tmp7 = out(vreg) _,
        tmp8 = out(vreg) _,
        tmp9 = out(vreg) _,
        tmp10 = out(vreg) _,
        tmp11 = out(vreg) _,
        tmp12 = out(vreg) _,
        tmp13 = out(vreg) _,
        tmp14 = out(vreg) _,
        tmp15 = out(vreg) _,
        options(nostack),
        );
    }
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        group.bench_function("f32x8 rust", |b| {
            result.fill(0.0);
            b.iter(|| f32x8_load_add(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        group.bench_function("f32x8 asm", |b| {
            result.fill(0.0);
            b.iter(|| f32x8_load_add_asm(data_ptr, rows, &mut result));
            assert_eq!(result, golden_result);
        });
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f32x4_load_add(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    assert_eq!(rows % 4, 0);
    let mut a0 = unsafe { vdupq_n_f32(0.0) };
    let mut a1 = unsafe { vdupq_n_f32(0.0) };
    let mut a2 = unsafe { vdupq_n_f32(0.0) };
    let mut a3 = unsafe { vdupq_n_f32(0.0) };
    let mut a4 = unsafe { vdupq_n_f32(0.0) };
    let mut a5 = unsafe { vdupq_n_f32(0.0) };
    let mut a6 = unsafe { vdupq_n_f32(0.0) };
    let mut a7 = unsafe { vdupq_n_f32(0.0) };
    for j in 0..rows / 4 {
        unsafe {
            let r00 = vld1q_f32(data.add(j * 4 * 16));
            let r01 = vld1q_f32(data.add(j * 4 * 16 + 4));
            let r02 = vld1q_f32(data.add(j * 4 * 16 + 8));
            let r03 = vld1q_f32(data.add(j * 4 * 16 + 12));
            let r10 = vld1q_f32(data.add((j * 4 + 1) * 16));
            let r11 = vld1q_f32(data.add((j * 4 + 1) * 16 + 4));
            let r12 = vld1q_f32(data.add((j * 4 + 1) * 16 + 8));
            let r13 = vld1q_f32(data.add((j * 4 + 1) * 16 + 12));
            a0 = vaddq_f32(a0, vaddq_f32(r00, r10));
            a1 = vaddq_f32(a1, vaddq_f32(r01, r11));
            a2 = vaddq_f32(a2, vaddq_f32(r02, r12));
            a3 = vaddq_f32(a3, vaddq_f32(r03, r13));
            let r00 = vld1q_f32(data.add((j * 4 + 2) * 16));
            let r01 = vld1q_f32(data.add((j * 4 + 2) * 16 + 4));
            let r02 = vld1q_f32(data.add((j * 4 + 2) * 16 + 8));
            let r03 = vld1q_f32(data.add((j * 4 + 2) * 16 + 12));
            let r10 = vld1q_f32(data.add((j * 4 + 3) * 16));
            let r11 = vld1q_f32(data.add((j * 4 + 3) * 16 + 4));
            let r12 = vld1q_f32(data.add((j * 4 + 3) * 16 + 8));
            let r13 = vld1q_f32(data.add((j * 4 + 3) * 16 + 12));
            a4 = vaddq_f32(a4, vaddq_f32(r00, r10));
            a5 = vaddq_f32(a5, vaddq_f32(r01, r11));
            a6 = vaddq_f32(a6, vaddq_f32(r02, r12));
            a7 = vaddq_f32(a7, vaddq_f32(r03, r13));
        }
    }
    unsafe {
        let rp = result.as_mut_ptr();
        vst1q_f32(rp, a0);
        vst1q_f32(rp.add(4), a1);
        vst1q_f32(rp.add(8), a2);
        vst1q_f32(rp.add(12), a3);
        vst1q_f32(rp.add(16), a4);
        vst1q_f32(rp.add(20), a5);
        vst1q_f32(rp.add(24), a6);
        vst1q_f32(rp.add(28), a7);
    }
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f32x4_load_add_asm(data: *const f32, rows: usize, result: &mut [f32; 32]) {
    unsafe {
        asm!(
        "eor {a0}.16b, {a0}.16b, {a0}.16b",
        "eor {a1}.16b, {a1}.16b, {a1}.16b",
        "eor {a2}.16b, {a2}.16b, {a2}.16b",
        "eor {a3}.16b, {a3}.16b, {a3}.16b",
        "eor {a4}.16b, {a4}.16b, {a4}.16b",
        "eor {a5}.16b, {a5}.16b, {a5}.16b",
        "eor {a6}.16b, {a6}.16b, {a6}.16b",
        "eor {a7}.16b, {a7}.16b, {a7}.16b",
        ".p2align 5",
        "2:",
        "ld1 {{ {tmp0}.4s, {tmp1}.4s, {tmp2}.4s, {tmp3}.4s }}, [{data}], #64",
        "ld1 {{ {tmp4}.4s, {tmp5}.4s, {tmp6}.4s, {tmp7}.4s }}, [{data}], #64",
        "fadd {tmp0}.4s, {tmp0}.4s, {tmp4}.4s",
        "fadd {tmp1}.4s, {tmp1}.4s, {tmp5}.4s",
        "fadd {tmp2}.4s, {tmp2}.4s, {tmp6}.4s",
        "fadd {tmp3}.4s, {tmp3}.4s, {tmp7}.4s",
        "fadd {a0}.4s, {a0}.4s, {tmp0}.4s",
        "fadd {a1}.4s, {a1}.4s, {tmp1}.4s",
        "fadd {a2}.4s, {a2}.4s, {tmp2}.4s",
        "fadd {a3}.4s, {a3}.4s, {tmp3}.4s",
        "ld1 {{ {tmp8}.4s, {tmp9}.4s, {tmp10}.4s, {tmp11}.4s }}, [{data}], #64",
        "ld1 {{ {tmp12}.4s, {tmp13}.4s, {tmp14}.4s, {tmp15}.4s }}, [{data}], #64",
        "fadd {tmp8}.4s, {tmp8}.4s, {tmp12}.4s",
        "fadd {tmp9}.4s, {tmp9}.4s, {tmp13}.4s",
        "fadd {tmp10}.4s, {tmp10}.4s, {tmp14}.4s",
        "fadd {tmp11}.4s, {tmp11}.4s, {tmp15}.4s",
        "fadd {a4}.4s, {a4}.4s, {tmp8}.4s",
        "fadd {a5}.4s, {a5}.4s, {tmp9}.4s",
        "fadd {a6}.4s, {a6}.4s, {tmp10}.4s",
        "fadd {a7}.4s, {a7}.4s, {tmp11}.4s",
        "cmp {data}, {data_end}",
        "b.lo 2b",
        "stp {a0:q}, {a1:q}, [{result}]",
        "stp {a2:q}, {a3:q}, [{result}, #32]",
        "stp {a4:q}, {a5:q}, [{result}, #64]",
        "stp {a6:q}, {a7:q}, [{result}, #96]",
        data = in(reg) data,
        data_end = in(reg) data.add(rows * 16),
        result = in(reg) result.as_ptr(),
        a0 = out(vreg) _,
        a1 = out(vreg) _,
        a2 = out(vreg) _,
        a3 = out(vreg) _,
        a4 = out(vreg) _,
        a5 = out(vreg) _,
        a6 = out(vreg) _,
        a7 = out(vreg) _,
        tmp0 = out(vreg) _,
        tmp1 = out(vreg) _,
        tmp2 = out(vreg) _,
        tmp3 = out(vreg) _,
        tmp4 = out(vreg) _,
        tmp5 = out(vreg) _,
        tmp6 = out(vreg) _,
        tmp7 = out(vreg) _,
        tmp8 = out(vreg) _,
        tmp9 = out(vreg) _,
        tmp10 = out(vreg) _,
        tmp11 = out(vreg) _,
        tmp12 = out(vreg) _,
        tmp13 = out(vreg) _,
        tmp14 = out(vreg) _,
        tmp15 = out(vreg) _,
        options(nostack),
        );
    }
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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
