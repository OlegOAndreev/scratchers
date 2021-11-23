#![allow(non_snake_case)]

use std::cell::RefCell;
use std::ops::{DerefMut, Range};
use std::sync::atomic;
use std::sync::atomic::AtomicBool;

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
            for (i, (g, d)) in golden_dst_vec.iter().zip(dst_vec.iter()).enumerate() {
                let diff = (g - d).abs();
                let max = g.abs().max(d.abs());
                if diff > f32::EPSILON && diff / max >= tolerance {
                    assert!(false, "Different values [{}]: {} vs {}", i, g, d);
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

// A variant of vec4_matrix_vec_mul_v3 without manual unrolling with configurable amount of rows to
// unroll.
fn vec4_matrix_vec_mul_v4<const N: usize>(
    src_mat: &AlignedMatrix,
    src_vecs: &[AlignedVec],
    dst_vecs: &mut [AlignedVec],
) {
    let last_src_idx = (src_vecs.len() / N) * N;
    for src_idx in (0..last_src_idx).step_by(N) {
        let dst_len = dst_vecs[src_idx].len();
        let mut src_slices: [&[Vec4]; N] = [&[]; N];
        for i in 0..N {
            src_slices[i] = src_vecs[src_idx + i].as_vec4();
        }
        let last_l = (dst_len / 4) * 4;
        for l in (0..last_l).step_by(4) {
            let mut accums: [[Vec4; 4]; N] = [[Vec4::ZERO; 4]; N];
            let mat_row0 = src_mat.as_vec4(l);
            let mat_row1 = src_mat.as_vec4(l + 1);
            let mat_row2 = src_mat.as_vec4(l + 2);
            let mat_row3 = src_mat.as_vec4(l + 3);
            // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and
            // only if at least one of the margins (either matrix or vec) is filled with zeroes.
            for ((((n, &m0), &m1), &m2), &m3) in mat_row0.iter()
                .enumerate()
                .zip(mat_row1)
                .zip(mat_row2)
                .zip(mat_row3) {
                for i in 0..N {
                    accums[i][0] += m0 * src_slices[i][n];
                    accums[i][1] += m1 * src_slices[i][n];
                    accums[i][2] += m2 * src_slices[i][n];
                    accums[i][3] += m3 * src_slices[i][n];
                }
            }
            for i in 0..N {
                let sum = Mat4::from_cols(accums[i][0], accums[i][1], accums[i][2], accums[i][3])
                    .transpose();
                dst_vecs[src_idx + i].put_vec4(l / 4, sum.x_axis + sum.y_axis + sum.z_axis
                    + sum.w_axis);
            }
        }
        // Compute the remaining 0-3 elements.
        match dst_len - last_l {
            3 => {
                let mut accums: [[Vec4; 3]; N] = [[Vec4::ZERO; 3]; N];
                let mat_row0 = src_mat.as_vec4(last_l);
                let mat_row1 = src_mat.as_vec4(last_l + 1);
                let mat_row2 = src_mat.as_vec4(last_l + 2);
                // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and
                // only if at least one of the margins (either matrix or vec) is filled with zeroes.
                for (((n, &m0), &m1), &m2) in mat_row0.iter()
                    .enumerate()
                    .zip(mat_row1)
                    .zip(mat_row2) {
                    for i in 0..N {
                        accums[i][0] += m0 * src_slices[i][n];
                        accums[i][1] += m1 * src_slices[i][n];
                        accums[i][2] += m2 * src_slices[i][n];
                    }
                }
                for i in 0..N {
                    let sum = Mat4::from_cols(accums[i][0], accums[i][1], accums[i][2], Vec4::ZERO)
                        .transpose();
                    dst_vecs[src_idx + i].put_vec4(last_l / 4, sum.x_axis + sum.y_axis + sum.z_axis
                        + sum.w_axis);
                }
            }
            2 => {
                let mut accums: [[Vec4; 2]; N] = [[Vec4::ZERO; 2]; N];
                let mat_row0 = src_mat.as_vec4(last_l);
                let mat_row1 = src_mat.as_vec4(last_l + 1);
                // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and
                // only if at least one of the margins (either matrix or vec) is filled with zeroes.
                for ((n, &m0), &m1) in mat_row0.iter()
                    .enumerate()
                    .zip(mat_row1) {
                    for i in 0..N {
                        accums[i][0] += m0 * src_slices[i][n];
                        accums[i][1] += m1 * src_slices[i][n];
                    }
                }
                for i in 0..N {
                    let sum = Mat4::from_cols(accums[i][0], accums[i][1], Vec4::ZERO, Vec4::ZERO)
                        .transpose();
                    dst_vecs[src_idx + i].put_vec4(last_l / 4, sum.x_axis + sum.y_axis + sum.z_axis
                        + sum.w_axis);
                }
            }
            1 => {
                let mut accums: [Vec4; N] = [Vec4::ZERO; N];
                let mat_row0 = src_mat.as_vec4(last_l);
                // NOTE: We read into the AlignedVec/AlignedMatrix margin here. This is correct if and
                // only if at least one of the margins (either matrix or vec) is filled with zeroes.
                for (n, &m0) in mat_row0.iter()
                    .enumerate() {
                    for i in 0..N {
                        accums[i] += m0 * src_slices[i][n];
                    }
                }
                for i in 0..N {
                    let sum = Mat4::from_cols(accums[i], Vec4::ZERO, Vec4::ZERO, Vec4::ZERO)
                        .transpose();
                    dst_vecs[src_idx + i].put_vec4(last_l / 4, sum.x_axis + sum.y_axis + sum.z_axis
                        + sum.w_axis);
                }
            }
            0 => {}
            _ => unreachable!("Bad remainder: {}, {}", last_l, dst_len)
        }
    }

    for src_idx in last_src_idx..src_vecs.len() {
        vec4_matrix_vec_mul_v2(src_mat, &src_vecs[src_idx], &mut dst_vecs[src_idx]);
    }
}

extern "C" {
    #[cfg(feature = "ispc")]
    pub fn ispc_matrix_vec_mul_v1(
        src_mat: *const f32,
        src_mat_stride: i32,
        src_vec: *const f32,
        dst_vec: *mut f32,
        K: i32,
        L: i32,
    );

    #[cfg(feature = "ispc")]
    pub fn ispc_matrix_vec_mul_v1_launch(
        src_mat: *const f32,
        src_mat_stride: i32,
        src_vec: *const f32,
        dst_vec: *mut f32,
        K: i32,
        L: i32,
    );

    #[cfg(feature = "ispc")]
    pub fn ispc_matrix_vec_mul_v2(
        src_mat: *const f32,
        src_mat_stride: i32,
        src_vec: *const f32,
        dst_vec: *mut f32,
        K: i32,
        L: i32,
    );

    #[cfg(feature = "ispc")]
    pub fn ispc_matrix_vec_mul_v3(
        src_mat: *const f32,
        src_mat_stride: i32,
        src_vec0: *const f32,
        src_vec1: *const f32,
        src_vec2: *const f32,
        src_vec3: *const f32,
        dst_vec0: *mut f32,
        dst_vec1: *mut f32,
        dst_vec2: *mut f32,
        dst_vec3: *mut f32,
        K: i32,
        L: i32,
    );
}

#[cfg(feature = "ispc")]
pub fn ispc_matrix_vec_mul_v1_wrapper(
    src_mat: &AlignedMatrix,
    src_vec: &AlignedVec,
    dst_vec: &mut AlignedVec,
) {
    let src_mat_slice = src_mat.as_f32_whole();
    let src_mat_stride = src_mat_slice.len() / src_mat.height();
    unsafe {
        ispc_matrix_vec_mul_v1(
            src_mat_slice.as_ptr(),
            src_mat_stride as i32,
            src_vec.as_f32().as_ptr(),
            dst_vec.as_f32_mut().as_mut_ptr(),
            src_vec.len() as i32,
            dst_vec.len() as i32,
        );
    }
}

#[cfg(feature = "ispc")]
pub fn ispc_matrix_vec_mul_v1_launch_wrapper(
    src_mat: &AlignedMatrix,
    src_vec: &AlignedVec,
    dst_vec: &mut AlignedVec,
) {
    let src_mat_slice = src_mat.as_f32_whole();
    let src_mat_stride = src_mat_slice.len() / src_mat.height();
    unsafe {
        ispc_matrix_vec_mul_v1_launch(
            src_mat_slice.as_ptr(),
            src_mat_stride as i32,
            src_vec.as_f32().as_ptr(),
            dst_vec.as_f32_mut().as_mut_ptr(),
            src_vec.len() as i32,
            dst_vec.len() as i32,
        );
    }
}

#[cfg(feature = "ispc")]
pub fn ispc_matrix_vec_mul_v2_wrapper(
    src_mat: &AlignedMatrix,
    src_vec: &AlignedVec,
    dst_vec: &mut AlignedVec,
) {
    let src_mat_slice = src_mat.as_f32_whole();
    let src_mat_stride = src_mat_slice.len() / src_mat.height();
    unsafe {
        ispc_matrix_vec_mul_v2(
            src_mat_slice.as_ptr(),
            src_mat_stride as i32,
            src_vec.as_f32().as_ptr(),
            dst_vec.as_f32_mut().as_mut_ptr(),
            src_vec.len() as i32,
            dst_vec.len() as i32,
        );
    }
}

#[cfg(feature = "ispc")]
pub fn ispc_matrix_vec_mul_v3_wrapper(
    src_mat: &AlignedMatrix,
    src_vec0: &AlignedVec,
    src_vec1: &AlignedVec,
    src_vec2: &AlignedVec,
    src_vec3: &AlignedVec,
    dst_vec0: &mut AlignedVec,
    dst_vec1: &mut AlignedVec,
    dst_vec2: &mut AlignedVec,
    dst_vec3: &mut AlignedVec,
) {
    let src_mat_slice = src_mat.as_f32_whole();
    let src_mat_stride = src_mat_slice.len() / src_mat.height();
    unsafe {
        ispc_matrix_vec_mul_v3(
            src_mat_slice.as_ptr(),
            src_mat_stride as i32,
            src_vec0.as_f32().as_ptr(),
            src_vec1.as_f32().as_ptr(),
            src_vec2.as_f32().as_ptr(),
            src_vec3.as_f32().as_ptr(),
            dst_vec0.as_f32_mut().as_mut_ptr(),
            dst_vec1.as_f32_mut().as_mut_ptr(),
            dst_vec2.as_f32_mut().as_mut_ptr(),
            dst_vec3.as_f32_mut().as_mut_ptr(),
            src_vec0.len() as i32,
            dst_vec0.len() as i32,
        );
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

fn bench_vec4_matrix_vec_multiply_v3(input: &MatrixVecMultiplyInput) {
    vec4_matrix_vec_mul_v3(&input.src_mat, &input.src_vecs, input.dst_vecs.borrow_mut().deref_mut());
}

fn bench_vec4_matrix_vec_multiply_v4<const N: usize>(input: &MatrixVecMultiplyInput) {
    vec4_matrix_vec_mul_v4::<N>(&input.src_mat, &input.src_vecs,
                                input.dst_vecs.borrow_mut().deref_mut());
}

#[cfg(feature = "ispc")]
fn bench_ispc_matrix_vec_multiply_v1(input: &MatrixVecMultiplyInput) {
    for (dst_vec, src_vec) in input.dst_vecs.borrow_mut()
        .iter_mut().zip(&input.src_vecs) {
        ispc_matrix_vec_mul_v1_wrapper(&input.src_mat, src_vec, dst_vec);
    }
}

#[cfg(feature = "ispc")]
fn bench_ispc_matrix_vec_multiply_v1_launch(input: &MatrixVecMultiplyInput) {
    for (dst_vec, src_vec) in input.dst_vecs.borrow_mut()
        .iter_mut().zip(&input.src_vecs) {
        ispc_matrix_vec_mul_v1_launch_wrapper(&input.src_mat, src_vec, dst_vec);
    }
}

#[cfg(feature = "ispc")]
fn bench_ispc_matrix_vec_multiply_v2(input: &MatrixVecMultiplyInput) {
    for (dst_vec, src_vec) in input.dst_vecs.borrow_mut()
        .iter_mut().zip(&input.src_vecs) {
        ispc_matrix_vec_mul_v2_wrapper(&input.src_mat, src_vec, dst_vec);
    }
}

#[cfg(feature = "ispc")]
fn bench_ispc_matrix_vec_multiply_v3(input: &MatrixVecMultiplyInput) {
    let M = input.src_vecs.len();
    let last_m = (M / 4) * 4;
    let mut dst_vecs = input.dst_vecs.borrow_mut();
    for m in (0..last_m).step_by(4) {
        let (dst_vecs0, dst_vecs1, dst_vecs2, dst_vecs3) = get_mut_by_index_4(&mut dst_vecs, m);
        ispc_matrix_vec_mul_v3_wrapper(
            &input.src_mat,
            &input.src_vecs[m],
            &input.src_vecs[m + 1],
            &input.src_vecs[m + 2],
            &input.src_vecs[m + 3],
            dst_vecs0,
            dst_vecs1,
            dst_vecs2,
            dst_vecs3,
        );
    }

    for m in last_m..M {
        ispc_matrix_vec_mul_v2_wrapper(&input.src_mat, &input.src_vecs[m], &mut dst_vecs[m]);
    }
}

// Unfortunately, Rust does not have convenient wrappers for getting multiple borrows for different
// slice elements.
#[allow(dead_code)]
fn get_mut_by_index_4<T>(s: &mut [T], base_index: usize) -> (&mut T, &mut T, &mut T, &mut T) {
    // This function can be rewritten via split_at_mut() and 4 split_first_mut()s, but it is
    // a) more code and b) potentially costlier due to bounds checks.
    assert!(base_index < s.len() && base_index + 4 <= s.len());
    unsafe {
        let ptr = s.as_mut_ptr().add(base_index);
        (&mut *ptr, &mut *ptr.add(1), &mut *ptr.add(2), &mut *ptr.add(3))
    }
}

fn bench_rayon_matrix_vec_multiply(input: &MatrixVecMultiplyInput) {
    let K = input.K;
    let L = input.L;
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
    let chunk_size = if K >= 32 {
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

fn bench_rayon_vec4_matrix_vec_multiply_v4<const N: usize>(input: &MatrixVecMultiplyInput) {
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
            vec4_matrix_vec_mul_v4::<N>(src_mat, src_vec_chunk, dst_vec_chunk);
        });
}

#[cfg(feature = "ispc")]
fn bench_rayon_ispc_matrix_vec_multiply_v1(input: &MatrixVecMultiplyInput) {
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
                ispc_matrix_vec_mul_v1_wrapper(src_mat, src_vec, dst_vec);
            }
        });
}

#[cfg(feature = "ispc")]
fn bench_rayon_ispc_matrix_vec_multiply_v2(input: &MatrixVecMultiplyInput) {
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
                ispc_matrix_vec_mul_v2_wrapper(src_mat, src_vec, dst_vec);
            }
        });
}

#[cfg(feature = "ispc")]
fn bench_rayon_ispc_matrix_vec_multiply_v3(input: &MatrixVecMultiplyInput) {
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
            let M = src_vec_chunk.len();
            let last_m = (M / 4) * 4;
            for m in (0..last_m).step_by(4) {
                let (dst_vecs0, dst_vecs1, dst_vecs2, dst_vecs3)
                    = get_mut_by_index_4(dst_vec_chunk, m);
                ispc_matrix_vec_mul_v3_wrapper(
                    src_mat,
                    &src_vec_chunk[m],
                    &src_vec_chunk[m + 1],
                    &src_vec_chunk[m + 2],
                    &src_vec_chunk[m + 3],
                    dst_vecs0,
                    dst_vecs1,
                    dst_vecs2,
                    dst_vecs3,
                );
            }

            for m in last_m..M {
                ispc_matrix_vec_mul_v2_wrapper(src_mat, &src_vec_chunk[m], &mut dst_vec_chunk[m]);
            }
        });
}

fn matrix_vec_multiply(c: &mut Criterion) {
    let ncpu = init_rayon();
    // let integrated_gpu = GpuInput::new(false);
    // let discrete_gpu = GpuInput::new(true);

    for K in [16usize, 100usize, 128usize, 1000usize, 4000usize] {
        for L in [10usize, 128usize, 1000usize, 4000usize] {
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

                group.bench_function("v4 vec4 x4 single thread", |b| {
                    b.iter(|| bench_vec4_matrix_vec_multiply_v4::<4>(&input));
                    input.compare_golden_dst();
                });

                group.bench_function("v4 vec4 x16 single thread", |b| {
                    b.iter(|| bench_vec4_matrix_vec_multiply_v4::<16>(&input));
                    input.compare_golden_dst();
                });

                #[cfg(feature = "ispc")]
                    group.bench_function("v1 ispc single thread", |b| {
                    b.iter(|| bench_ispc_matrix_vec_multiply_v1(&input));
                    input.compare_golden_dst();
                });

                #[cfg(feature = "ispc")]
                    group.bench_function("v2 ispc single thread", |b| {
                    b.iter(|| bench_ispc_matrix_vec_multiply_v2(&input));
                    input.compare_golden_dst();
                });

                #[cfg(feature = "ispc")]
                    group.bench_function("v3 ispc single thread", |b| {
                    b.iter(|| bench_ispc_matrix_vec_multiply_v3(&input));
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

                    group.bench_function(format!("v4 vec4 x4 {} threads", ncpu), |b| {
                        b.iter(|| bench_rayon_vec4_matrix_vec_multiply_v4::<4>(&input));
                        input.compare_golden_dst();
                    });

                    group.bench_function(format!("v4 vec4 x16 {} threads", ncpu), |b| {
                        b.iter(|| bench_rayon_vec4_matrix_vec_multiply_v4::<16>(&input));
                        input.compare_golden_dst();
                    });

                    #[cfg(feature = "ispc")]
                        group.bench_function("v1 ispc launch", |b| {
                        b.iter(|| bench_ispc_matrix_vec_multiply_v1_launch(&input));
                        input.compare_golden_dst();
                    });

                    #[cfg(feature = "ispc")]
                        group.bench_function(format!("v1 ispc {} threads", ncpu), |b| {
                        b.iter(|| bench_rayon_ispc_matrix_vec_multiply_v1(&input));
                        input.compare_golden_dst();
                    });

                    #[cfg(feature = "ispc")]
                        group.bench_function(format!("v2 ispc {} threads", ncpu), |b| {
                        b.iter(|| bench_rayon_ispc_matrix_vec_multiply_v2(&input));
                        input.compare_golden_dst();
                    });

                    #[cfg(feature = "ispc")]
                        group.bench_function(format!("v3 ispc {} threads", ncpu), |b| {
                        b.iter(|| bench_rayon_ispc_matrix_vec_multiply_v3(&input));
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
    if !RAYON_GLOBAL_INIT.swap(true, atomic::Ordering::SeqCst) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(ncpu)
            .build_global()
            .unwrap();
    }
    ncpu
}

criterion_group!(benches, matrix_vec_multiply);
criterion_main!(benches);
