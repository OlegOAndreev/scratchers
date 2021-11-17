#![allow(non_snake_case)]

use std::cell::RefCell;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

use criterion::{Criterion, criterion_group, criterion_main};
use glam::{Mat4, Vec4};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

use scratchers::aligned_vec::{AlignedMatrix, AlignedVec};

static RAYON_GLOBAL_INIT: AtomicBool = AtomicBool::new(false);


// struct GpuInput {
//     instance: wgpu::Instance,
//     device: wgpu::Device,
// }


// Matrix-vector multiplication.

struct MatrixVecMultiplyInput {
    K: usize,
    L: usize,
    // Matrix sized KxL with stride mat_stride.
    src_mat: AlignedMatrix,
    // M vectors, each sized K
    src_vecs: Vec<AlignedVec>,
    // M vectors, each sized L. RefCell is used because criterion requires inputs to be passed
    // by immutable reference.
    dst_vecs: RefCell<Vec<AlignedVec>>,
    golden_dst_vecs: Vec<AlignedVec>,
}

impl MatrixVecMultiplyInput {
    fn reset_dst(&mut self) {
        for dst_vec in self.dst_vecs.borrow_mut().iter_mut() {
            dst_vec.fill(0.0);
        }
    }

    fn store_golden_dst(&mut self) {
        self.golden_dst_vecs = self.dst_vecs.borrow().clone();
        self.reset_dst();
    }

    fn compare_golden_dst(&mut self) {
        // See vec4_matrix_vec_mul_v2 for comments why the tolerances are higher than epsilon. The
        // amount of sums is self.K, so we use it in the tolerance multiplier.
        let tolerance = self.K as f32 * f32::EPSILON / 4.0;
        for (golden_dst_vec, dst_vec) in self.golden_dst_vecs.iter()
            .zip(self.dst_vecs.borrow().iter()) {
            assert_eq!(golden_dst_vec.len(), dst_vec.len());
            for (g, d) in golden_dst_vec.iter().zip(dst_vec.iter()) {
                let diff = (g - d).abs();
                let max = g.abs().max(d.abs());
                if diff > f32::EPSILON && diff / max >= tolerance {
                    assert!(false, "Different values: {} vs {}", g, d);
                }
            }
        }
        self.reset_dst();
    }
}

fn prepare_matrix_vec_multiply_input(K: usize, L: usize, M: usize) -> MatrixVecMultiplyInput {
    let mut src_mat = AlignedMatrix::new(K, L);
    for l in 0..L {
        for k in 0..K {
            src_mat[l][k] = (l + k) as f32;
        }
    }
    let mut src_vecs = Vec::with_capacity(M);
    for m in 0..M {
        src_vecs.push(AlignedVec::new(K));
        for k in 0..K {
            src_vecs[m][k] = 1.0 / (m * 3 + k * 2 + 1) as f32;
        }
    }
    let mut dst_vecs = Vec::with_capacity(M);
    for _ in 0..M {
        dst_vecs.push(AlignedVec::new(L));
    }

    MatrixVecMultiplyInput {
        K,
        L,
        src_mat,
        src_vecs,
        dst_vecs: RefCell::new(dst_vecs),
        golden_dst_vecs: vec![],
    }
}

// Fill dst_range part of dst_vec.
#[inline(always)]
fn standard_matrix_vec_mul(
    src_mat: &AlignedMatrix,
    src_vec: &AlignedVec,
    dst_vec: &mut AlignedVec,
    dst_range: Range<usize>,
) {
    for l in dst_range {
        dst_vec[l] = src_vec.iter()
            .zip(&src_mat[l])
            .map(|(v1, v2)| v1 * v2)
            .sum();
    }
}

// Basic SIMD algorithm: load Vec4 from src vector and multiply it by 4 matrix rows.
// TODO: Add a second version that would multiply several src vectors at once by chunks of 8
// to improve cache utilization.
fn vec4_matrix_vec_mul(
    src_mat: &AlignedMatrix,
    src_vec: &AlignedVec,
    dst_vec: &mut AlignedVec,
) {
    let dst_len = dst_vec.len();
    let src_slice = src_vec.as_vec4();
    let dst_slice = dst_vec.as_vec4_mut();
    let last_l = (dst_len / 4) * 4;
    for l in (0..last_l).step_by(4) {
        let mut dst = Vec4::ZERO;
        let mat_row0 = src_mat.as_vec4(l);
        let mat_row1 = src_mat.as_vec4(l + 1);
        let mat_row2 = src_mat.as_vec4(l + 2);
        let mat_row3 = src_mat.as_vec4(l + 3);
        // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and only
        // if at least one of the margins (either matrix or vec) is filled with zeroes.
        for ((((&src, &m0), &m1), &m2), &m3) in src_slice.iter()
            .zip(mat_row0)
            .zip(mat_row1)
            .zip(mat_row2)
            .zip(mat_row3) {
            let mul = Mat4::from_cols(src * m0, src * m1, src * m2, src * m3)
                .transpose();
            dst += mul.x_axis;
            dst += mul.y_axis;
            dst += mul.z_axis;
            dst += mul.w_axis;
        }
        dst_slice[l / 4] = dst;
    }
    // Compute the remaining 0-3 elements.
    match dst_len - last_l {
        3 => {
            let mut dst = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(last_l);
            let mat_row1 = src_mat.as_vec4(last_l + 1);
            let mat_row2 = src_mat.as_vec4(last_l + 2);
            for (((&src, &m0), &m1), &m2) in src_slice.iter()
                .zip(mat_row0)
                .zip(mat_row1)
                .zip(mat_row2) {
                let mul = Mat4::from_cols(src * m0, src * m1, src * m2, Vec4::ZERO)
                    .transpose();
                dst += mul.x_axis;
                dst += mul.y_axis;
                dst += mul.z_axis;
                dst += mul.w_axis;
            }
            dst_slice[last_l / 4] = dst;
        }
        2 => {
            let mut dst = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(last_l);
            let mat_row1 = src_mat.as_vec4(last_l + 1);
            for ((&src, &m0), &m1) in src_slice.iter()
                .zip(mat_row0)
                .zip(mat_row1) {
                let mul = Mat4::from_cols(src * m0, src * m1, Vec4::ZERO, Vec4::ZERO)
                    .transpose();
                dst += mul.x_axis;
                dst += mul.y_axis;
                dst += mul.z_axis;
                dst += mul.w_axis;
            }
            dst_slice[last_l / 4] = dst;
        }
        1 => {
            let mut dst = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(last_l);
            for (&src, &m0) in src_slice.iter()
                .zip(mat_row0) {
                let mul = Mat4::from_cols(src * m0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO)
                    .transpose();
                dst += mul.x_axis;
                dst += mul.y_axis;
                dst += mul.z_axis;
                dst += mul.w_axis;
            }
            dst_slice[last_l / 4] = dst;
        }
        0 => {}
        _ => unreachable!("Bad remainder: {}, {}", last_l, dst_len)
    }
}

// A variation of vec4_matrix_vec_mul which keeps 4 horizontal sums and sums them only at the end.
// It reorders the operations (s[0] + s[4] + s[8] + ... + s[1] + s[5] + ...) and requires higher
// tolerances when comparing against golden results.
// TODO: Add a second version that would multiply several src vectors at once by chunks of 8
// to improve cache utilization.
fn vec4_matrix_vec_mul_v2(
    src_mat: &AlignedMatrix,
    src_vec: &AlignedVec,
    dst_vec: &mut AlignedVec,
) {
    let dst_len = dst_vec.len();
    let src_slice = src_vec.as_vec4();
    let dst_slice = dst_vec.as_vec4_mut();
    let last_l = (dst_len / 4) * 4;
    for l in (0..last_l).step_by(4) {
        let mut sum0 = Vec4::ZERO;
        let mut sum1 = Vec4::ZERO;
        let mut sum2 = Vec4::ZERO;
        let mut sum3 = Vec4::ZERO;
        let mat_row0 = src_mat.as_vec4(l);
        let mat_row1 = src_mat.as_vec4(l + 1);
        let mat_row2 = src_mat.as_vec4(l + 2);
        let mat_row3 = src_mat.as_vec4(l + 3);
        // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and only
        // if at least one of the margins (either matrix or vec) is filled with zeroes.
        for ((((&src, &m0), &m1), &m2), &m3) in src_slice.iter()
            .zip(mat_row0)
            .zip(mat_row1)
            .zip(mat_row2)
            .zip(mat_row3) {
            sum0 += src * m0;
            sum1 += src * m1;
            sum2 += src * m2;
            sum3 += src * m3;
        }
        let sum = Mat4::from_cols(sum0, sum1, sum2, sum3).transpose();
        dst_slice[l / 4] = sum.x_axis + sum.y_axis + sum.z_axis + sum.w_axis;
    }
    // Compute the remaining 0-3 elements.
    match dst_len - last_l {
        3 => {
            let mut sum0 = Vec4::ZERO;
            let mut sum1 = Vec4::ZERO;
            let mut sum2 = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(last_l);
            let mat_row1 = src_mat.as_vec4(last_l + 1);
            let mat_row2 = src_mat.as_vec4(last_l + 2);
            // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and only
            // if at least one of the margins (either matrix or vec) is filled with zeroes.
            for (((&src, &m0), &m1), &m2) in src_slice.iter()
                .zip(mat_row0)
                .zip(mat_row1)
                .zip(mat_row2) {
                sum0 += src * m0;
                sum1 += src * m1;
                sum2 += src * m2;
            }
            let sum = Mat4::from_cols(sum0, sum1, sum2, Vec4::ZERO).transpose();
            dst_slice[last_l / 4] = sum.x_axis + sum.y_axis + sum.z_axis + sum.w_axis;
        }
        2 => {
            let mut sum0 = Vec4::ZERO;
            let mut sum1 = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(last_l);
            let mat_row1 = src_mat.as_vec4(last_l + 1);
            // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and only
            // if at least one of the margins (either matrix or vec) is filled with zeroes.
            for ((&src, &m0), &m1) in src_slice.iter()
                .zip(mat_row0)
                .zip(mat_row1) {
                sum0 += src * m0;
                sum1 += src * m1;
            }
            let sum = Mat4::from_cols(sum0, sum1, Vec4::ZERO, Vec4::ZERO).transpose();
            dst_slice[last_l / 4] = sum.x_axis + sum.y_axis + sum.z_axis + sum.w_axis;
        }
        1 => {
            let mut sum0 = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(last_l);
            // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and only
            // if at least one of the margins (either matrix or vec) is filled with zeroes.
            for (&src, &m0) in src_slice.iter()
                .zip(mat_row0) {
                sum0 += src * m0;
            }
            let sum = Mat4::from_cols(sum0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
            dst_slice[last_l / 4] = sum.x_axis + sum.y_axis + sum.z_axis + sum.w_axis;
        }
        0 => {}
        _ => unreachable!("Bad remainder: {}, {}", last_l, dst_len)
    }
}

fn bench_single_thread_matrix_vec_multiply(input: &MatrixVecMultiplyInput) {
    for (dst_vec, src_vec) in input.dst_vecs.borrow_mut()
        .iter_mut().zip(&input.src_vecs) {
        standard_matrix_vec_mul(&input.src_mat, src_vec, dst_vec, 0..input.L);
    }
}

fn bench_vec4_matrix_vec_multiply(input: &MatrixVecMultiplyInput) {
    for (dst_vec, src_vec) in input.dst_vecs.borrow_mut()
        .iter_mut().zip(&input.src_vecs) {
        vec4_matrix_vec_mul(&input.src_mat, src_vec, dst_vec);
    }
}

fn bench_vec4_matrix_vec_multiply_v2(input: &MatrixVecMultiplyInput) {
    for (dst_vec, src_vec) in input.dst_vecs.borrow_mut()
        .iter_mut().zip(&input.src_vecs) {
        vec4_matrix_vec_mul_v2(&input.src_mat, src_vec, dst_vec);
    }
}

fn bench_rayon_matrix_vec_multiply(input: &MatrixVecMultiplyInput) {
    let K = input.K;
    let L = input.L;
    let src_mat = &input.src_mat;
    let chunk_size = if K >= 64 {
        1
    } else if K >= 32 {
        4
    } else {
        8
    };
    input.dst_vecs.borrow_mut().par_chunks_mut(chunk_size)
        .zip_eq(input.src_vecs.par_chunks(chunk_size))
        .for_each(|(dst_vec_chunk, src_vec_chunk)| {
            for (dst_vec, src_vec) in dst_vec_chunk.iter_mut().zip(src_vec_chunk) {
                standard_matrix_vec_mul(src_mat, src_vec, dst_vec, 0..L);
            }
        });
}

fn bench_rayon_vec4_matrix_vec_multiply(input: &MatrixVecMultiplyInput) {
    let K = input.K;
    let src_mat = &input.src_mat;
    let chunk_size = if K >= 64 {
        1
    } else if K >= 32 {
        4
    } else {
        8
    };
    input.dst_vecs.borrow_mut().par_chunks_mut(chunk_size)
        .zip_eq(input.src_vecs.par_chunks(chunk_size))
        .for_each(|(dst_vec_chunk, src_vec_chunk)| {
            for (dst_vec, src_vec) in dst_vec_chunk.iter_mut().zip(src_vec_chunk) {
                vec4_matrix_vec_mul(src_mat, src_vec, dst_vec);
            }
        });
}

fn bench_rayon_vec4_matrix_vec_multiply_v2(input: &MatrixVecMultiplyInput) {
    let K = input.K;
    let src_mat = &input.src_mat;
    let chunk_size = if K >= 64 {
        1
    } else if K >= 32 {
        4
    } else {
        8
    };
    input.dst_vecs.borrow_mut().par_chunks_mut(chunk_size)
        .zip_eq(input.src_vecs.par_chunks(chunk_size))
        .for_each(|(dst_vec_chunk, src_vec_chunk)| {
            for (dst_vec, src_vec) in dst_vec_chunk.iter_mut().zip(src_vec_chunk) {
                vec4_matrix_vec_mul_v2(src_mat, src_vec, dst_vec);
            }
        });
}

fn matrix_vec_multiply(c: &mut Criterion) {
    let ncpu = init_rayon();
    for K in [16usize, 100usize, 128usize, 1000usize] {
        for L in [10usize, 128usize] {
            for M in [1usize, 64usize, 500usize] {
                let mut group = c.benchmark_group(
                    format!("matrix_vec_multiply/size {}x{}, {} vecs", K, L, M));

                // Compute ~throughput in Gflops.
                group.throughput(criterion::Throughput::Elements(
                    K as u64 * L as u64 * M as u64 * 2));

                let mut input = prepare_matrix_vec_multiply_input(K, L, M);
                bench_single_thread_matrix_vec_multiply(&input);
                input.store_golden_dst();

                group.bench_function("single thread", |b| {
                    b.iter(|| bench_single_thread_matrix_vec_multiply(&input));
                    input.compare_golden_dst();
                });

                group.bench_function("vec4 single thread", |b| {
                    b.iter(|| bench_vec4_matrix_vec_multiply(&input));
                    input.compare_golden_dst();
                });

                group.bench_function("v2 vec4 single thread", |b| {
                    b.iter(|| bench_vec4_matrix_vec_multiply_v2(&input));
                    input.compare_golden_dst();
                });

                if M > 1 {
                    group.bench_function(format!("{} threads", ncpu), |b| {
                        b.iter(|| bench_rayon_matrix_vec_multiply(&input));
                        input.compare_golden_dst();
                    });

                    group.bench_function(format!("vec4 {} threads", ncpu), |b| {
                        b.iter(|| bench_rayon_vec4_matrix_vec_multiply(&input));
                        input.compare_golden_dst();
                    });

                    group.bench_function(format!("v2 vec4 {} threads", ncpu), |b| {
                        b.iter(|| bench_rayon_vec4_matrix_vec_multiply_v2(&input));
                        input.compare_golden_dst();
                    });
                }

                group.finish();
            }
        }
    }
}

fn init_rayon() -> usize {
    let ncpu = if let Ok(v) = std::env::var("NUM_CPUS") {
        v.parse::<usize>().unwrap()
    } else {
        num_cpus::get_physical()
    };
    if !RAYON_GLOBAL_INIT.swap(true, Ordering::SeqCst) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(ncpu)
            .build_global()
            .unwrap();
    }
    ncpu
}

criterion_group!(benches, matrix_vec_multiply);
criterion_main!(benches);
