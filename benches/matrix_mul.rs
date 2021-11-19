#![allow(non_snake_case)]

use std::cell::RefCell;
use std::ops::{DerefMut, Range};
use std::sync::atomic::{AtomicBool, Ordering};

use criterion::{Criterion, criterion_group, criterion_main};
// use futures::executor;
use glam::{Mat4, Vec4};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

use scratchers::aligned_vec::{AlignedMatrix, AlignedVec};

static RAYON_GLOBAL_INIT: AtomicBool = AtomicBool::new(false);


// struct GpuInput {
//     instance: wgpu::Instance,
//     device: wgpu::Device,
//     queue: wgpu::Queue,
// }
//
// impl GpuInput {
//     const BACKENDS: wgpu::Backends = wgpu::Backends::PRIMARY;
//
//     fn new(discrete: bool) -> Option<Self> {
//         let instance = wgpu::Instance::new(Self::BACKENDS);
//         let adapter = Self::find_adapter(&instance, discrete)?;
//         let adapter_info = adapter.get_info();
//         println!("Using adapter {:?}", adapter_info);
//         println!("Adapter limits: {:?}", adapter.limits());
//         println!("Adapter features: {:?}", adapter.features());
//
//         let (device, queue) = executor::block_on(Self::request_device(&adapter, discrete));
//         Some(Self {
//             instance,
//             device,
//             queue,
//         })
//     }
//
//     fn find_adapter(instance: &wgpu::Instance, discrete: bool) -> Option<wgpu::Adapter> {
//         instance.enumerate_adapters(Self::BACKENDS)
//             .find(|a| if discrete {
//                 a.get_info().device_type == wgpu::DeviceType::DiscreteGpu
//             } else {
//                 a.get_info().device_type == wgpu::DeviceType::IntegratedGpu
//             })
//     }
//
//     async fn request_device(
//         adapter: &wgpu::Adapter,
//         discrete: bool,
//     ) -> (wgpu::Device, wgpu::Queue) {
//         let trace_dir = std::env::var("WGPU_TRACE");
//         let features = Self::desired_features(adapter.features(), discrete);
//         match adapter.request_device(
//             &wgpu::DeviceDescriptor {
//                 features,
//                 ..Default::default()
//             },
//             trace_dir.ok().as_ref().map(std::path::Path::new),
//         ).await {
//             Ok((device, queue)) => (device, queue),
//             Err(e) => panic!("Failed to request device: {}", e),
//         }
//     }
//
//     fn desired_features(features: wgpu::Features, discrete: bool) -> wgpu::Features {
//         let features = if discrete {
//             features & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
//         } else {
//             features
//         };
//         features & wgpu::Features::PIPELINE_STATISTICS_QUERY
//     }
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
        let mut accum0 = Vec4::ZERO;
        let mut accum1 = Vec4::ZERO;
        let mut accum2 = Vec4::ZERO;
        let mut accum3 = Vec4::ZERO;
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
            accum0 += src * m0;
            accum1 += src * m1;
            accum2 += src * m2;
            accum3 += src * m3;
        }
        let sum = Mat4::from_cols(accum0, accum1, accum2, accum3).transpose();
        dst_slice[l / 4] = sum.x_axis + sum.y_axis + sum.z_axis + sum.w_axis;
    }
    // Compute the remaining 0-3 elements.
    match dst_len - last_l {
        3 => {
            let mut accum0 = Vec4::ZERO;
            let mut accum1 = Vec4::ZERO;
            let mut accum2 = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(last_l);
            let mat_row1 = src_mat.as_vec4(last_l + 1);
            let mat_row2 = src_mat.as_vec4(last_l + 2);
            // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and only
            // if at least one of the margins (either matrix or vec) is filled with zeroes.
            for (((&src, &m0), &m1), &m2) in src_slice.iter()
                .zip(mat_row0)
                .zip(mat_row1)
                .zip(mat_row2) {
                accum0 += src * m0;
                accum1 += src * m1;
                accum2 += src * m2;
            }
            let sum = Mat4::from_cols(accum0, accum1, accum2, Vec4::ZERO).transpose();
            dst_slice[last_l / 4] = sum.x_axis + sum.y_axis + sum.z_axis + sum.w_axis;
        }
        2 => {
            let mut accum0 = Vec4::ZERO;
            let mut accum1 = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(last_l);
            let mat_row1 = src_mat.as_vec4(last_l + 1);
            // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and only
            // if at least one of the margins (either matrix or vec) is filled with zeroes.
            for ((&src, &m0), &m1) in src_slice.iter()
                .zip(mat_row0)
                .zip(mat_row1) {
                accum0 += src * m0;
                accum1 += src * m1;
            }
            let sum = Mat4::from_cols(accum0, accum1, Vec4::ZERO, Vec4::ZERO).transpose();
            dst_slice[last_l / 4] = sum.x_axis + sum.y_axis + sum.z_axis + sum.w_axis;
        }
        1 => {
            let mut accum0 = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(last_l);
            // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and only
            // if at least one of the margins (either matrix or vec) is filled with zeroes.
            for (&src, &m0) in src_slice.iter()
                .zip(mat_row0) {
                accum0 += src * m0;
            }
            let sum = Mat4::from_cols(accum0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
            dst_slice[last_l / 4] = sum.x_axis + sum.y_axis + sum.z_axis + sum.w_axis;
        }
        0 => {}
        _ => unreachable!("Bad remainder: {}, {}", last_l, dst_len)
    }
}

// An improved version of vec4_matrix_vec_mul_v2: process multiple src_vecs at once. We manually
// unroll the inner loop to not depend on Rust unroller.
fn vec4_matrix_vec_mul_v3(
    src_mat: &AlignedMatrix,
    src_vecs: &[AlignedVec],
    dst_vecs: &mut [AlignedVec],
) {
    let last_src_idx = (src_vecs.len() / 4) * 4;
    for src_idx in (0..last_src_idx).step_by(4) {
        let dst_len = dst_vecs[src_idx].len();
        let src_slice0 = src_vecs[src_idx].as_vec4();
        let src_slice1 = src_vecs[src_idx + 1].as_vec4();
        let src_slice2 = src_vecs[src_idx + 2].as_vec4();
        let src_slice3 = src_vecs[src_idx + 3].as_vec4();
        let last_l = (dst_len / 4) * 4;
        for l in (0..last_l).step_by(4) {
            let mut accum0x0 = Vec4::ZERO;
            let mut accum1x0 = Vec4::ZERO;
            let mut accum2x0 = Vec4::ZERO;
            let mut accum3x0 = Vec4::ZERO;
            let mut accum0x1 = Vec4::ZERO;
            let mut accum1x1 = Vec4::ZERO;
            let mut accum2x1 = Vec4::ZERO;
            let mut accum3x1 = Vec4::ZERO;
            let mut accum0x2 = Vec4::ZERO;
            let mut accum1x2 = Vec4::ZERO;
            let mut accum2x2 = Vec4::ZERO;
            let mut accum3x2 = Vec4::ZERO;
            let mut accum0x3 = Vec4::ZERO;
            let mut accum1x3 = Vec4::ZERO;
            let mut accum2x3 = Vec4::ZERO;
            let mut accum3x3 = Vec4::ZERO;
            let mat_row0 = src_mat.as_vec4(l);
            let mat_row1 = src_mat.as_vec4(l + 1);
            let mat_row2 = src_mat.as_vec4(l + 2);
            let mat_row3 = src_mat.as_vec4(l + 3);
            // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and
            // only if at least one of the margins (either matrix or vec) is filled with zeroes.
            for (((((((&src0, &src1), &src2), &src3), &m0), &m1), &m2), &m3) in src_slice0.iter()
                .zip(src_slice1)
                .zip(src_slice2)
                .zip(src_slice3)
                .zip(mat_row0)
                .zip(mat_row1)
                .zip(mat_row2)
                .zip(mat_row3) {
                accum0x0 += src0 * m0;
                accum0x1 += src0 * m1;
                accum0x2 += src0 * m2;
                accum0x3 += src0 * m3;
                accum1x0 += src1 * m0;
                accum1x1 += src1 * m1;
                accum1x2 += src1 * m2;
                accum1x3 += src1 * m3;
                accum2x0 += src2 * m0;
                accum2x1 += src2 * m1;
                accum2x2 += src2 * m2;
                accum2x3 += src2 * m3;
                accum3x0 += src3 * m0;
                accum3x1 += src3 * m1;
                accum3x2 += src3 * m2;
                accum3x3 += src3 * m3;
            }
            let sum0 = Mat4::from_cols(accum0x0, accum0x1, accum0x2, accum0x3).transpose();
            dst_vecs[src_idx].put_vec4(l / 4, sum0.x_axis + sum0.y_axis + sum0.z_axis
                + sum0.w_axis);
            let sum1 = Mat4::from_cols(accum1x0, accum1x1, accum1x2, accum1x3).transpose();
            dst_vecs[src_idx + 1].put_vec4(l / 4, sum1.x_axis + sum1.y_axis + sum1.z_axis
                + sum1.w_axis);
            let sum2 = Mat4::from_cols(accum2x0, accum2x1, accum2x2, accum2x3).transpose();
            dst_vecs[src_idx + 2].put_vec4(l / 4, sum2.x_axis + sum2.y_axis + sum2.z_axis
                + sum2.w_axis);
            let sum3 = Mat4::from_cols(accum3x0, accum3x1, accum3x2, accum3x3).transpose();
            dst_vecs[src_idx + 3].put_vec4(l / 4, sum3.x_axis + sum3.y_axis + sum3.z_axis
                + sum3.w_axis);
        }
        // Compute the remaining 0-3 elements.
        match dst_len - last_l {
            3 => {
                let mut accum0x0 = Vec4::ZERO;
                let mut accum0x1 = Vec4::ZERO;
                let mut accum0x2 = Vec4::ZERO;
                let mut accum1x0 = Vec4::ZERO;
                let mut accum1x1 = Vec4::ZERO;
                let mut accum1x2 = Vec4::ZERO;
                let mut accum2x0 = Vec4::ZERO;
                let mut accum2x1 = Vec4::ZERO;
                let mut accum2x2 = Vec4::ZERO;
                let mut accum3x0 = Vec4::ZERO;
                let mut accum3x1 = Vec4::ZERO;
                let mut accum3x2 = Vec4::ZERO;
                let mat_row0 = src_mat.as_vec4(last_l);
                let mat_row1 = src_mat.as_vec4(last_l + 1);
                let mat_row2 = src_mat.as_vec4(last_l + 2);
                // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if
                // and only if at least one of the margins (either matrix or vec) is filled with
                // zeroes.
                for ((((((&src0, &src1), &src2), &src3), &m0), &m1), &m2) in src_slice0.iter()
                    .zip(src_slice1)
                    .zip(src_slice2)
                    .zip(src_slice3)
                    .zip(mat_row0)
                    .zip(mat_row1)
                    .zip(mat_row2) {
                    accum0x0 += src0 * m0;
                    accum0x1 += src0 * m1;
                    accum0x2 += src0 * m2;
                    accum1x0 += src1 * m0;
                    accum1x1 += src1 * m1;
                    accum1x2 += src1 * m2;
                    accum2x0 += src2 * m0;
                    accum2x1 += src2 * m1;
                    accum2x2 += src2 * m2;
                    accum3x0 += src3 * m0;
                    accum3x1 += src3 * m1;
                    accum3x2 += src3 * m2;
                }
                let sum0 = Mat4::from_cols(accum0x0, accum0x1, accum0x2, Vec4::ZERO).transpose();
                dst_vecs[src_idx].put_vec4(last_l / 4, sum0.x_axis + sum0.y_axis + sum0.z_axis
                    + sum0.w_axis);
                let sum1 = Mat4::from_cols(accum1x0, accum1x1, accum1x2, Vec4::ZERO).transpose();
                dst_vecs[src_idx + 1].put_vec4(last_l / 4, sum1.x_axis + sum1.y_axis + sum1.z_axis
                    + sum1.w_axis);
                let sum2 = Mat4::from_cols(accum2x0, accum2x1, accum2x2, Vec4::ZERO).transpose();
                dst_vecs[src_idx + 2].put_vec4(last_l / 4, sum2.x_axis + sum2.y_axis + sum2.z_axis
                    + sum2.w_axis);
                let sum3 = Mat4::from_cols(accum3x0, accum3x1, accum3x2, Vec4::ZERO).transpose();
                dst_vecs[src_idx + 3].put_vec4(last_l / 4, sum3.x_axis + sum3.y_axis + sum3.z_axis
                    + sum3.w_axis);
            }
            2 => {
                let mut accum0x0 = Vec4::ZERO;
                let mut accum0x1 = Vec4::ZERO;
                let mut accum1x0 = Vec4::ZERO;
                let mut accum1x1 = Vec4::ZERO;
                let mut accum2x0 = Vec4::ZERO;
                let mut accum2x1 = Vec4::ZERO;
                let mut accum3x0 = Vec4::ZERO;
                let mut accum3x1 = Vec4::ZERO;
                let mat_row0 = src_mat.as_vec4(last_l);
                let mat_row1 = src_mat.as_vec4(last_l + 1);
                // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if
                // and only if at least one of the margins (either matrix or vec) is filled with
                // zeroes.
                for (((((&src0, &src1), &src2), &src3), &m0), &m1) in src_slice0.iter()
                    .zip(src_slice1)
                    .zip(src_slice2)
                    .zip(src_slice3)
                    .zip(mat_row0)
                    .zip(mat_row1) {
                    accum0x0 += src0 * m0;
                    accum0x1 += src0 * m1;
                    accum1x0 += src1 * m0;
                    accum1x1 += src1 * m1;
                    accum2x0 += src2 * m0;
                    accum2x1 += src2 * m1;
                    accum3x0 += src3 * m0;
                    accum3x1 += src3 * m1;
                }
                let sum0 = Mat4::from_cols(accum0x0, accum0x1, Vec4::ZERO, Vec4::ZERO).transpose();
                dst_vecs[src_idx].put_vec4(last_l / 4, sum0.x_axis + sum0.y_axis + sum0.z_axis
                    + sum0.w_axis);
                let sum1 = Mat4::from_cols(accum1x0, accum1x1, Vec4::ZERO, Vec4::ZERO).transpose();
                dst_vecs[src_idx + 1].put_vec4(last_l / 4, sum1.x_axis + sum1.y_axis + sum1.z_axis
                    + sum1.w_axis);
                let sum2 = Mat4::from_cols(accum2x0, accum2x1, Vec4::ZERO, Vec4::ZERO).transpose();
                dst_vecs[src_idx + 2].put_vec4(last_l / 4, sum2.x_axis + sum2.y_axis + sum2.z_axis
                    + sum2.w_axis);
                let sum3 = Mat4::from_cols(accum3x0, accum3x1, Vec4::ZERO, Vec4::ZERO).transpose();
                dst_vecs[src_idx + 3].put_vec4(last_l / 4, sum3.x_axis + sum3.y_axis + sum3.z_axis
                    + sum3.w_axis);
            }
            1 => {
                let mut accum0x0 = Vec4::ZERO;
                let mut accum1x0 = Vec4::ZERO;
                let mut accum2x0 = Vec4::ZERO;
                let mut accum3x0 = Vec4::ZERO;
                let mat_row0 = src_mat.as_vec4(last_l);
                // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if
                // and only if at least one of the margins (either matrix or vec) is filled with
                // zeroes.
                for ((((&src0, &src1), &src2), &src3), &m0) in src_slice0.iter()
                    .zip(src_slice1)
                    .zip(src_slice2)
                    .zip(src_slice3)
                    .zip(mat_row0) {
                    accum0x0 += src0 * m0;
                    accum1x0 += src1 * m0;
                    accum2x0 += src2 * m0;
                    accum3x0 += src3 * m0;
                }
                let sum0 = Mat4::from_cols(accum0x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
                dst_vecs[src_idx].put_vec4(last_l / 4, sum0.x_axis + sum0.y_axis + sum0.z_axis
                    + sum0.w_axis);
                let sum1 = Mat4::from_cols(accum1x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
                dst_vecs[src_idx + 1].put_vec4(last_l / 4, sum1.x_axis + sum1.y_axis + sum1.z_axis
                    + sum1.w_axis);
                let sum2 = Mat4::from_cols(accum2x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
                dst_vecs[src_idx + 2].put_vec4(last_l / 4, sum2.x_axis + sum2.y_axis + sum2.z_axis
                    + sum2.w_axis);
                let sum3 = Mat4::from_cols(accum3x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
                dst_vecs[src_idx + 3].put_vec4(last_l / 4, sum3.x_axis + sum3.y_axis + sum3.z_axis
                    + sum3.w_axis);
            }
            0 => {}
            _ => unreachable!("Bad remainder: {}, {}", last_l, dst_len)
        }
    }

    for src_idx in last_src_idx..src_vecs.len() {
        vec4_matrix_vec_mul_v2(src_mat, &src_vecs[src_idx], &mut dst_vecs[src_idx]);
    }
}

// fn vec4_matrix_vec_mul_v3(
//     src_mat: &AlignedMatrix,
//     src_vecs: &[AlignedVec],
//     dst_vecs: &mut [AlignedVec],
// ) {
//     let last_src_idx = (src_vecs.len() / 8) * 8;
//     for src_idx in (0..last_src_idx).step_by(8) {
//         let dst_len = dst_vecs[src_idx].len();
//         let src_slice0 = src_vecs[src_idx].as_vec4();
//         let src_slice1 = src_vecs[src_idx + 1].as_vec4();
//         let src_slice2 = src_vecs[src_idx + 2].as_vec4();
//         let src_slice3 = src_vecs[src_idx + 3].as_vec4();
//         let src_slice4 = src_vecs[src_idx + 4].as_vec4();
//         let src_slice5 = src_vecs[src_idx + 5].as_vec4();
//         let src_slice6 = src_vecs[src_idx + 6].as_vec4();
//         let src_slice7 = src_vecs[src_idx + 7].as_vec4();
//         let last_l = (dst_len / 4) * 4;
//         for l in (0..last_l).step_by(4) {
//             let mut accum0x0 = Vec4::ZERO;
//             let mut accum0x1 = Vec4::ZERO;
//             let mut accum0x2 = Vec4::ZERO;
//             let mut accum0x3 = Vec4::ZERO;
//             let mut accum1x0 = Vec4::ZERO;
//             let mut accum1x1 = Vec4::ZERO;
//             let mut accum1x2 = Vec4::ZERO;
//             let mut accum1x3 = Vec4::ZERO;
//             let mut accum2x0 = Vec4::ZERO;
//             let mut accum2x1 = Vec4::ZERO;
//             let mut accum2x2 = Vec4::ZERO;
//             let mut accum2x3 = Vec4::ZERO;
//             let mut accum3x0 = Vec4::ZERO;
//             let mut accum3x1 = Vec4::ZERO;
//             let mut accum3x2 = Vec4::ZERO;
//             let mut accum3x3 = Vec4::ZERO;
//             let mut accum4x0 = Vec4::ZERO;
//             let mut accum4x1 = Vec4::ZERO;
//             let mut accum4x2 = Vec4::ZERO;
//             let mut accum4x3 = Vec4::ZERO;
//             let mut accum5x0 = Vec4::ZERO;
//             let mut accum5x1 = Vec4::ZERO;
//             let mut accum5x2 = Vec4::ZERO;
//             let mut accum5x3 = Vec4::ZERO;
//             let mut accum6x0 = Vec4::ZERO;
//             let mut accum6x1 = Vec4::ZERO;
//             let mut accum6x2 = Vec4::ZERO;
//             let mut accum6x3 = Vec4::ZERO;
//             let mut accum7x0 = Vec4::ZERO;
//             let mut accum7x1 = Vec4::ZERO;
//             let mut accum7x2 = Vec4::ZERO;
//             let mut accum7x3 = Vec4::ZERO;
//             let mat_row0 = src_mat.as_vec4(l);
//             let mat_row1 = src_mat.as_vec4(l + 1);
//             let mat_row2 = src_mat.as_vec4(l + 2);
//             let mat_row3 = src_mat.as_vec4(l + 3);
//             // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and
//             // only if at least one of the margins (either matrix or vec) is filled with zeroes.
//             for (((((((((((
//                 &src0,
//                 &src1),
//                 &src2),
//                 &src3),
//                 &src4),
//                 &src5),
//                 &src6),
//                 &src7),
//                 &m0),
//                 &m1),
//                 &m2),
//                 &m3)
//             in src_slice0.iter()
//                 .zip(src_slice1)
//                 .zip(src_slice2)
//                 .zip(src_slice3)
//                 .zip(src_slice4)
//                 .zip(src_slice5)
//                 .zip(src_slice6)
//                 .zip(src_slice7)
//                 .zip(mat_row0)
//                 .zip(mat_row1)
//                 .zip(mat_row2)
//                 .zip(mat_row3) {
//                 accum0x0 += src0 * m0;
//                 accum0x1 += src0 * m1;
//                 accum0x2 += src0 * m2;
//                 accum0x3 += src0 * m3;
//                 accum1x0 += src1 * m0;
//                 accum1x1 += src1 * m1;
//                 accum1x2 += src1 * m2;
//                 accum1x3 += src1 * m3;
//                 accum2x0 += src2 * m0;
//                 accum2x1 += src2 * m1;
//                 accum2x2 += src2 * m2;
//                 accum2x3 += src2 * m3;
//                 accum3x0 += src3 * m0;
//                 accum3x1 += src3 * m1;
//                 accum3x2 += src3 * m2;
//                 accum3x3 += src3 * m3;
//                 accum4x0 += src4 * m0;
//                 accum4x1 += src4 * m1;
//                 accum4x2 += src4 * m2;
//                 accum4x3 += src4 * m3;
//                 accum5x0 += src5 * m0;
//                 accum5x1 += src5 * m1;
//                 accum5x2 += src5 * m2;
//                 accum5x3 += src5 * m3;
//                 accum6x0 += src6 * m0;
//                 accum6x1 += src6 * m1;
//                 accum6x2 += src6 * m2;
//                 accum6x3 += src6 * m3;
//                 accum7x0 += src7 * m0;
//                 accum7x1 += src7 * m1;
//                 accum7x2 += src7 * m2;
//                 accum7x3 += src7 * m3;
//             }
//             let sum0 = Mat4::from_cols(accum0x0, accum0x1, accum0x2, accum0x3).transpose();
//             dst_vecs[src_idx].put_vec4(l / 4, sum0.x_axis + sum0.y_axis + sum0.z_axis
//                 + sum0.w_axis);
//             let sum1 = Mat4::from_cols(accum1x0, accum1x1, accum1x2, accum1x3).transpose();
//             dst_vecs[src_idx + 1].put_vec4(l / 4, sum1.x_axis + sum1.y_axis + sum1.z_axis
//                 + sum1.w_axis);
//             let sum2 = Mat4::from_cols(accum2x0, accum2x1, accum2x2, accum2x3).transpose();
//             dst_vecs[src_idx + 2].put_vec4(l / 4, sum2.x_axis + sum2.y_axis + sum2.z_axis
//                 + sum2.w_axis);
//             let sum3 = Mat4::from_cols(accum3x0, accum3x1, accum3x2, accum3x3).transpose();
//             dst_vecs[src_idx + 3].put_vec4(l / 4, sum3.x_axis + sum3.y_axis + sum3.z_axis
//                 + sum3.w_axis);
//             let sum4 = Mat4::from_cols(accum4x0, accum4x1, accum4x2, accum4x3).transpose();
//             dst_vecs[src_idx + 4].put_vec4(l / 4, sum4.x_axis + sum4.y_axis + sum4.z_axis
//                 + sum4.w_axis);
//             let sum5 = Mat4::from_cols(accum5x0, accum5x1, accum5x2, accum5x3).transpose();
//             dst_vecs[src_idx + 5].put_vec4(l / 4, sum5.x_axis + sum5.y_axis + sum5.z_axis
//                 + sum5.w_axis);
//             let sum6 = Mat4::from_cols(accum6x0, accum6x1, accum6x2, accum6x3).transpose();
//             dst_vecs[src_idx + 6].put_vec4(l / 4, sum6.x_axis + sum6.y_axis + sum6.z_axis
//                 + sum6.w_axis);
//             let sum7 = Mat4::from_cols(accum7x0, accum7x1, accum7x2, accum7x3).transpose();
//             dst_vecs[src_idx + 7].put_vec4(l / 4, sum7.x_axis + sum7.y_axis + sum7.z_axis
//                 + sum7.w_axis);
//         }
//         // Compute the remaining 0-3 elements.
//         match dst_len - last_l {
//             3 => {
//                 let mut accum0x0 = Vec4::ZERO;
//                 let mut accum0x1 = Vec4::ZERO;
//                 let mut accum0x2 = Vec4::ZERO;
//                 let mut accum1x0 = Vec4::ZERO;
//                 let mut accum1x1 = Vec4::ZERO;
//                 let mut accum1x2 = Vec4::ZERO;
//                 let mut accum2x0 = Vec4::ZERO;
//                 let mut accum2x1 = Vec4::ZERO;
//                 let mut accum2x2 = Vec4::ZERO;
//                 let mut accum3x0 = Vec4::ZERO;
//                 let mut accum3x1 = Vec4::ZERO;
//                 let mut accum3x2 = Vec4::ZERO;
//                 let mut accum4x0 = Vec4::ZERO;
//                 let mut accum4x1 = Vec4::ZERO;
//                 let mut accum4x2 = Vec4::ZERO;
//                 let mut accum5x0 = Vec4::ZERO;
//                 let mut accum5x1 = Vec4::ZERO;
//                 let mut accum5x2 = Vec4::ZERO;
//                 let mut accum6x0 = Vec4::ZERO;
//                 let mut accum6x1 = Vec4::ZERO;
//                 let mut accum6x2 = Vec4::ZERO;
//                 let mut accum7x0 = Vec4::ZERO;
//                 let mut accum7x1 = Vec4::ZERO;
//                 let mut accum7x2 = Vec4::ZERO;
//                 let mat_row0 = src_mat.as_vec4(last_l);
//                 let mat_row1 = src_mat.as_vec4(last_l + 1);
//                 let mat_row2 = src_mat.as_vec4(last_l + 2);
//                 // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if
//                 // and only if at least one of the margins (either matrix or vec) is filled with
//                 // zeroes.
//                 for ((((((((((
//                     &src0,
//                     &src1),
//                     &src2),
//                     &src3),
//                     &src4),
//                     &src5),
//                     &src6),
//                     &src7),
//                     &m0),
//                     &m1),
//                     &m2)
//                 in src_slice0.iter()
//                     .zip(src_slice1)
//                     .zip(src_slice2)
//                     .zip(src_slice3)
//                     .zip(src_slice4)
//                     .zip(src_slice5)
//                     .zip(src_slice6)
//                     .zip(src_slice7)
//                     .zip(mat_row0)
//                     .zip(mat_row1)
//                     .zip(mat_row2) {
//                     accum0x0 += src0 * m0;
//                     accum0x1 += src0 * m1;
//                     accum0x2 += src0 * m2;
//                     accum1x0 += src1 * m0;
//                     accum1x1 += src1 * m1;
//                     accum1x2 += src1 * m2;
//                     accum2x0 += src2 * m0;
//                     accum2x1 += src2 * m1;
//                     accum2x2 += src2 * m2;
//                     accum3x0 += src3 * m0;
//                     accum3x1 += src3 * m1;
//                     accum3x2 += src3 * m2;
//                     accum4x0 += src4 * m0;
//                     accum4x1 += src4 * m1;
//                     accum4x2 += src4 * m2;
//                     accum5x0 += src5 * m0;
//                     accum5x1 += src5 * m1;
//                     accum5x2 += src5 * m2;
//                     accum6x0 += src6 * m0;
//                     accum6x1 += src6 * m1;
//                     accum6x2 += src6 * m2;
//                     accum7x0 += src7 * m0;
//                     accum7x1 += src7 * m1;
//                     accum7x2 += src7 * m2;
//                 }
//                 let sum0 = Mat4::from_cols(accum0x0, accum0x1, accum0x2, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx].put_vec4(last_l / 4, sum0.x_axis + sum0.y_axis + sum0.z_axis
//                     + sum0.w_axis);
//                 let sum1 = Mat4::from_cols(accum1x0, accum1x1, accum1x2, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 1].put_vec4(last_l / 4, sum1.x_axis + sum1.y_axis + sum1.z_axis
//                     + sum1.w_axis);
//                 let sum2 = Mat4::from_cols(accum2x0, accum2x1, accum2x2, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 2].put_vec4(last_l / 4, sum2.x_axis + sum2.y_axis + sum2.z_axis
//                     + sum2.w_axis);
//                 let sum3 = Mat4::from_cols(accum3x0, accum3x1, accum3x2, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 3].put_vec4(last_l / 4, sum3.x_axis + sum3.y_axis + sum3.z_axis
//                     + sum3.w_axis);
//                 let sum4 = Mat4::from_cols(accum4x0, accum4x1, accum4x2, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 4].put_vec4(last_l / 4, sum4.x_axis + sum4.y_axis + sum4.z_axis
//                     + sum4.w_axis);
//                 let sum5 = Mat4::from_cols(accum5x0, accum5x1, accum5x2, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 5].put_vec4(last_l / 4, sum5.x_axis + sum5.y_axis + sum5.z_axis
//                     + sum5.w_axis);
//                 let sum6 = Mat4::from_cols(accum6x0, accum6x1, accum6x2, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 6].put_vec4(last_l / 4, sum6.x_axis + sum6.y_axis + sum6.z_axis
//                     + sum6.w_axis);
//                 let sum7 = Mat4::from_cols(accum7x0, accum7x1, accum7x2, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 7].put_vec4(last_l / 4, sum7.x_axis + sum7.y_axis + sum7.z_axis
//                     + sum7.w_axis);
//             }
//             2 => {
//                 let mut accum0x0 = Vec4::ZERO;
//                 let mut accum0x1 = Vec4::ZERO;
//                 let mut accum1x0 = Vec4::ZERO;
//                 let mut accum1x1 = Vec4::ZERO;
//                 let mut accum2x0 = Vec4::ZERO;
//                 let mut accum2x1 = Vec4::ZERO;
//                 let mut accum3x0 = Vec4::ZERO;
//                 let mut accum3x1 = Vec4::ZERO;
//                 let mut accum4x0 = Vec4::ZERO;
//                 let mut accum4x1 = Vec4::ZERO;
//                 let mut accum5x0 = Vec4::ZERO;
//                 let mut accum5x1 = Vec4::ZERO;
//                 let mut accum6x0 = Vec4::ZERO;
//                 let mut accum6x1 = Vec4::ZERO;
//                 let mut accum7x0 = Vec4::ZERO;
//                 let mut accum7x1 = Vec4::ZERO;
//                 let mat_row0 = src_mat.as_vec4(last_l);
//                 let mat_row1 = src_mat.as_vec4(last_l + 1);
//                 // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if
//                 // and only if at least one of the margins (either matrix or vec) is filled with
//                 // zeroes.
//                 for (((((((((
//                     &src0,
//                     &src1),
//                     &src2),
//                     &src3),
//                     &src4),
//                     &src5),
//                     &src6),
//                     &src7),
//                     &m0),
//                     &m1)
//                 in src_slice0.iter()
//                     .zip(src_slice1)
//                     .zip(src_slice2)
//                     .zip(src_slice3)
//                     .zip(src_slice4)
//                     .zip(src_slice5)
//                     .zip(src_slice6)
//                     .zip(src_slice7)
//                     .zip(mat_row0)
//                     .zip(mat_row1) {
//                     accum0x0 += src0 * m0;
//                     accum0x1 += src0 * m1;
//                     accum1x0 += src1 * m0;
//                     accum1x1 += src1 * m1;
//                     accum2x0 += src2 * m0;
//                     accum2x1 += src2 * m1;
//                     accum3x0 += src3 * m0;
//                     accum3x1 += src3 * m1;
//                     accum4x0 += src4 * m0;
//                     accum4x1 += src4 * m1;
//                     accum5x0 += src5 * m0;
//                     accum5x1 += src5 * m1;
//                     accum6x0 += src6 * m0;
//                     accum6x1 += src6 * m1;
//                     accum7x0 += src7 * m0;
//                     accum7x1 += src7 * m1;
//                 }
//                 let sum0 = Mat4::from_cols(accum0x0, accum0x1, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx].put_vec4(last_l / 4, sum0.x_axis + sum0.y_axis + sum0.z_axis
//                     + sum0.w_axis);
//                 let sum1 = Mat4::from_cols(accum1x0, accum1x1, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 1].put_vec4(last_l / 4, sum1.x_axis + sum1.y_axis + sum1.z_axis
//                     + sum1.w_axis);
//                 let sum2 = Mat4::from_cols(accum2x0, accum2x1, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 2].put_vec4(last_l / 4, sum2.x_axis + sum2.y_axis + sum2.z_axis
//                     + sum2.w_axis);
//                 let sum3 = Mat4::from_cols(accum3x0, accum3x1, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 3].put_vec4(last_l / 4, sum3.x_axis + sum3.y_axis + sum3.z_axis
//                     + sum3.w_axis);
//                 let sum4 = Mat4::from_cols(accum4x0, accum4x1, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 4].put_vec4(last_l / 4, sum4.x_axis + sum4.y_axis + sum4.z_axis
//                     + sum4.w_axis);
//                 let sum5 = Mat4::from_cols(accum5x0, accum5x1, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 5].put_vec4(last_l / 4, sum5.x_axis + sum5.y_axis + sum5.z_axis
//                     + sum5.w_axis);
//                 let sum6 = Mat4::from_cols(accum6x0, accum6x1, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 6].put_vec4(last_l / 4, sum6.x_axis + sum6.y_axis + sum6.z_axis
//                     + sum6.w_axis);
//                 let sum7 = Mat4::from_cols(accum7x0, accum7x1, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 7].put_vec4(last_l / 4, sum7.x_axis + sum7.y_axis + sum7.z_axis
//                     + sum7.w_axis);
//             }
//             1 => {
//                 let mut accum0x0 = Vec4::ZERO;
//                 let mut accum1x0 = Vec4::ZERO;
//                 let mut accum2x0 = Vec4::ZERO;
//                 let mut accum3x0 = Vec4::ZERO;
//                 let mut accum4x0 = Vec4::ZERO;
//                 let mut accum5x0 = Vec4::ZERO;
//                 let mut accum6x0 = Vec4::ZERO;
//                 let mut accum7x0 = Vec4::ZERO;
//                 let mat_row0 = src_mat.as_vec4(last_l);
//                 // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if
//                 // and only if at least one of the margins (either matrix or vec) is filled with
//                 // zeroes.
//                 for ((((((((
//                     &src0,
//                     &src1),
//                     &src2),
//                     &src3),
//                     &src4),
//                     &src5),
//                     &src6),
//                     &src7),
//                     &m0) in src_slice0.iter()
//                     .zip(src_slice1)
//                     .zip(src_slice2)
//                     .zip(src_slice3)
//                     .zip(src_slice4)
//                     .zip(src_slice5)
//                     .zip(src_slice6)
//                     .zip(src_slice7)
//                     .zip(mat_row0) {
//                     accum0x0 += src0 * m0;
//                     accum1x0 += src1 * m0;
//                     accum2x0 += src2 * m0;
//                     accum3x0 += src3 * m0;
//                     accum4x0 += src4 * m0;
//                     accum5x0 += src5 * m0;
//                     accum6x0 += src6 * m0;
//                     accum7x0 += src7 * m0;
//                 }
//                 let sum0 = Mat4::from_cols(accum0x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx].put_vec4(last_l / 4, sum0.x_axis + sum0.y_axis + sum0.z_axis
//                     + sum0.w_axis);
//                 let sum1 = Mat4::from_cols(accum1x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 1].put_vec4(last_l / 4, sum1.x_axis + sum1.y_axis + sum1.z_axis
//                     + sum1.w_axis);
//                 let sum2 = Mat4::from_cols(accum2x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 2].put_vec4(last_l / 4, sum2.x_axis + sum2.y_axis + sum2.z_axis
//                     + sum2.w_axis);
//                 let sum3 = Mat4::from_cols(accum3x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 3].put_vec4(last_l / 4, sum3.x_axis + sum3.y_axis + sum3.z_axis
//                     + sum3.w_axis);
//                 let sum4 = Mat4::from_cols(accum4x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 4].put_vec4(last_l / 4, sum4.x_axis + sum4.y_axis + sum4.z_axis
//                     + sum4.w_axis);
//                 let sum5 = Mat4::from_cols(accum5x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 5].put_vec4(last_l / 4, sum5.x_axis + sum5.y_axis + sum5.z_axis
//                     + sum5.w_axis);
//                 let sum6 = Mat4::from_cols(accum6x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 6].put_vec4(last_l / 4, sum6.x_axis + sum6.y_axis + sum6.z_axis
//                     + sum6.w_axis);
//                 let sum7 = Mat4::from_cols(accum7x0, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO).transpose();
//                 dst_vecs[src_idx + 7].put_vec4(last_l / 4, sum7.x_axis + sum7.y_axis + sum7.z_axis
//                     + sum7.w_axis);
//             }
//             0 => {}
//             _ => unreachable!("Bad remainder: {}, {}", last_l, dst_len)
//         }
//     }
//
//     for src_idx in last_src_idx..src_vecs.len() {
//         vec4_matrix_vec_mul_v2(src_mat, &src_vecs[src_idx], &mut dst_vecs[src_idx]);
//     }
// }

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

fn bench_vec4_matrix_vec_multiply_v3(input: &MatrixVecMultiplyInput) {
    vec4_matrix_vec_mul_v3(&input.src_mat, &input.src_vecs, input.dst_vecs.borrow_mut().deref_mut());
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
    let chunk_size = if K >= 32 {
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

fn bench_rayon_vec4_matrix_vec_multiply_v3(input: &MatrixVecMultiplyInput) {
    let K = input.K;
    let src_mat = &input.src_mat;
    let chunk_size = if K >= 32 {
        4
    } else {
        8
    };
    input.dst_vecs.borrow_mut().par_chunks_mut(chunk_size)
        .zip_eq(input.src_vecs.par_chunks(chunk_size))
        .for_each(|(dst_vec_chunk, src_vec_chunk)| {
            vec4_matrix_vec_mul_v3(src_mat, src_vec_chunk, dst_vec_chunk);
        });
}

fn matrix_vec_multiply(c: &mut Criterion) {
    let ncpu = init_rayon();
    // let integrated_gpu = GpuInput::new(false);
    // let discrete_gpu = GpuInput::new(true);

    for K in [16usize, 100usize, 128usize, 1000usize, 2000usize] {
        for L in [10usize, 128usize, 1000usize, 2000usize] {
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

                group.bench_function("v3 vec4 single thread", |b| {
                    b.iter(|| bench_vec4_matrix_vec_multiply_v3(&input));
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

                    group.bench_function(format!("v3 vec4 {} threads", ncpu), |b| {
                        b.iter(|| bench_rayon_vec4_matrix_vec_multiply_v3(&input));
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
