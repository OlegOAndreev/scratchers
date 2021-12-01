#![allow(non_snake_case)]

use std::{iter, task, thread};
use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Display, Formatter};
use std::ops::{DerefMut, Range};
use std::pin::Pin;
use std::sync::atomic;
use std::sync::atomic::AtomicBool;
use std::task::Poll;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use futures::{executor, FutureExt};
use futures::Future;
use glam::{Mat4, Vec4};
use log::{info, warn};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use wgpu::util::DeviceExt;

use scratchers::aligned_vec::{AlignedMatrix, AlignedVec};

static RAYON_GLOBAL_INIT: AtomicBool = AtomicBool::new(false);

#[allow(dead_code)]
struct Gpu {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    pub const POLL_INTERVAL: Duration = Duration::from_micros(100);

    const BACKENDS: wgpu::Backends = wgpu::Backends::PRIMARY;

    pub fn new(discrete: bool) -> Option<Self> {
        let instance = wgpu::Instance::new(Self::BACKENDS);
        let adapter = Self::find_adapter(&instance, discrete)?;
        let adapter_info = adapter.get_info();
        info!("Using adapter {:?}", adapter_info);
        info!("Adapter limits: {:?}", adapter.limits());
        info!("Adapter features: {:?}", adapter.features());

        let (device, queue) = executor::block_on(Self::request_device(&adapter));
        Some(Self {
            instance,
            device,
            queue,
        })
    }

    fn find_adapter(instance: &wgpu::Instance, discrete: bool) -> Option<wgpu::Adapter> {
        instance.enumerate_adapters(Self::BACKENDS)
            .find(|a| if discrete {
                a.get_info().device_type == wgpu::DeviceType::DiscreteGpu
            } else {
                a.get_info().device_type == wgpu::DeviceType::IntegratedGpu
            })
    }

    async fn request_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
        let trace_dir = std::env::var("WGPU_TRACE");
        let features = adapter.features() & Self::desired_features();
        match adapter.request_device(
            &wgpu::DeviceDescriptor {
                features,
                ..Default::default()
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        ).await {
            Ok((device, queue)) => (device, queue),
            Err(e) => panic!("Failed to request device: {}", e),
        }
    }

    fn desired_features() -> wgpu::Features {
        wgpu::Features::PIPELINE_STATISTICS_QUERY
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::CLEAR_COMMANDS
            // Ignore the warning, we want to measure the actual perf loss.
            | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
    }

    pub fn wait_for_submitted_work_done(&self, stats: &mut GpuStats) {
        let start_wait_complete_time = Instant::now();
        let mut fut = self.queue.on_submitted_work_done();
        let mut cx = task::Context::from_waker(futures::task::noop_waker_ref());
        'poll: loop {
            match Pin::new(&mut fut).poll(&mut cx) {
                Poll::Ready(_) => {
                    break 'poll;
                }
                Poll::Pending => {
                    self.device.poll(wgpu::Maintain::Poll);
                    thread::sleep(Gpu::POLL_INTERVAL);
                }
            }
        }
        stats.wait_complete_time += Instant::now() - start_wait_complete_time;
    }
}

// Matrix-vector multiplication.

struct MatrixVecMultiplyInput {
    K: usize,
    L: usize,
    // Matrix sized KxL.
    src_mat: AlignedMatrix,
    // M vectors, each sized K
    src_vecs: Vec<AlignedVec>,
    // M vectors, each sized L. RefCell is used because criterion requires inputs to be passed
    // by immutable reference.
    dst_vecs: RefCell<Vec<AlignedVec>>,
    golden_dst_vecs: Vec<AlignedVec>,
}

impl MatrixVecMultiplyInput {
    fn new(K: usize, L: usize, M: usize) -> Self {
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

        Self {
            K,
            L,
            src_mat,
            src_vecs,
            dst_vecs: RefCell::new(dst_vecs),
            golden_dst_vecs: vec![],
        }
    }

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

struct MatrixVecGpuPipelines {
    // One bind group layout for all shader variants.
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_v1: wgpu::ComputePipeline,
}

impl MatrixVecGpuPipelines {
    pub fn new(gpu: &Gpu) -> Self {
        let bind_group_layout = gpu.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: false,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                ],
            });
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let matrix_mul_wgsl = include_str!("matrix_mul.wgsl");

        let shader_v1 = gpu.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::from(matrix_mul_wgsl)),
        });
        let pipeline_v1 = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_v1,
            entry_point: "main_v1",
        });
        Self {
            bind_group_layout,
            pipeline_v1,
        }
    }
}

#[allow(dead_code)]
struct MatrixVecMultiplyGpuInput {
    // If true, do not use staging buffer for copying data.
    unified_memory: bool,

    // This is un-aligned number of rows.
    num_rows: u32,
    // Aligned length of dst_vec (num_rows <= dst_vec_len <= num_rows + 3).
    dst_vec_len: u32,
    num_vecs: u32,

    uniforms_buffer: wgpu::Buffer,
    // Corresponds to in_matrix.
    // TODO: Check if we should merge src_mat and src_vecs and use (dynamic?) buffer offsets.
    src_mat_buffer: wgpu::Buffer,
    // Corresponds to in_vec. Contains all M vectors.
    src_vecs_buffer: wgpu::Buffer,
    // Corresponds to out_vec. Contains all M vectors.
    dst_vecs_buffer: wgpu::Buffer,
    // Used only if unified_memory is false. Contains data for both src_mat + src_vecs and dst_vecs.
    staging_buffer: Option<wgpu::Buffer>,

    bind_group: wgpu::BindGroup,
}

#[derive(Default)]
struct GpuStats {
    pub map_time: Duration,
    pub copy_time: Duration,
    pub unmap_time: Duration,
    pub submit_time: Duration,
    pub wait_complete_time: Duration,

    pub count: u32,
}

impl Display for GpuStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.count == 0 {
            return write!(f, "no invocations");
        }
        write!(f, "map time: {:?}, copy time: {:?}, unmap time: {:?}, submit time: {:?}, \
            wait complete time: {:?}",
               self.map_time / self.count,
               self.copy_time / self.count,
               self.unmap_time / self.count,
               self.submit_time / self.count,
               self.wait_complete_time / self.count,
        )
    }
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Uniforms {
    // Both K and L must be aligned. Otherwise we need to pass stride both for src_mat, src_vecs
    // and dst_vecs which is just too much hassle.
    K: u32,
    L: u32,
}

impl MatrixVecMultiplyGpuInput {
    fn from(
        gpu: &Gpu,
        pipeline: &MatrixVecGpuPipelines,
        input: &MatrixVecMultiplyInput,
        unified_memory: bool,
    ) -> Self {
        // All buffers contain AlignedMatrix/AlignedVec, use aligned K and L as well.
        let K = input.src_mat.u8_stride() / 4;
        let L = input.dst_vecs.borrow()[0].u8_len() / 4;
        let uniforms = Uniforms {
            K: K as u32,
            L: L as u32,
        };
        let uniforms_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut src_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if unified_memory {
            src_usage |= wgpu::BufferUsages::MAP_WRITE;
        }
        let mut dst_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        if unified_memory {
            dst_usage |= wgpu::BufferUsages::MAP_READ;
        }
        let src_mat_size = K * L * 4;
        let src_mat_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("src_mat"),
            size: src_mat_size as u64,
            usage: src_usage,
            mapped_at_creation: false,
        });
        let M = input.src_vecs.len();
        let src_vecs_size = K * M * 4;
        let src_vecs_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("src_vecs"),
            size: src_vecs_size as u64,
            usage: src_usage,
            mapped_at_creation: false,
        });
        let dst_vecs_size = L * M * 4;
        let dst_vecs_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dst_vecs"),
            size: dst_vecs_size as u64,
            usage: dst_usage,
            mapped_at_creation: false,
        });
        let staging_buffer_size = (src_mat_size + src_vecs_size).max(dst_vecs_size);
        let staging_buffer = if unified_memory {
            None
        } else {
            Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_buffer"),
                size: staging_buffer_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
                mapped_at_creation: false,
            }))
        };

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_mat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: src_vecs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dst_vecs_buffer.as_entire_binding(),
                },
            ],
        });

        let ret = Self {
            unified_memory,

            num_rows: input.L as u32,
            dst_vec_len: L as u32,
            num_vecs: input.src_vecs.len() as u32,

            uniforms_buffer,
            src_mat_buffer,
            src_vecs_buffer,
            dst_vecs_buffer,
            staging_buffer,

            bind_group,
        };
        ret.copy_src_to_gpu(gpu, input, &mut GpuStats::default());
        ret
    }

    pub fn reset_dst(&self, gpu: &Gpu) {
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor::default());
        encoder.clear_buffer(&self.dst_vecs_buffer, 0, None);
        let cmd = encoder.finish();
        gpu.queue.submit(iter::once(cmd));
        gpu.device.poll(wgpu::Maintain::Wait);
    }

    // We could accept CommandEncoder here and add copy command to it, but 1) most real programs
    // would probably submit the data before dispatching the jobs/draw calls and 2) it may be
    // an optimization: overlap copying data with dispatching the jobs/draw calls.
    pub fn copy_src_to_gpu(&self, gpu: &Gpu, input: &MatrixVecMultiplyInput, stats: &mut GpuStats) {
        if self.unified_memory {
            let start_map_time = Instant::now();
            self.map_buffers(gpu, &[&self.src_mat_buffer, &self.src_vecs_buffer],
                             wgpu::MapMode::Write);
            stats.map_time += Instant::now() - start_map_time;

            let start_copy_time = Instant::now();
            let mut src_mat_data = self.src_mat_buffer.slice(..).get_mapped_range_mut();
            let src_mat_u8 = input.src_mat.as_u8_whole();
            let src_mat_size = src_mat_u8.len();
            src_mat_data[0..src_mat_size].copy_from_slice(src_mat_u8);

            let mut src_vecs_data = self.src_vecs_buffer.slice(..).get_mapped_range_mut();
            let mut offset = 0;
            for src_vec in input.src_vecs.iter() {
                let src_vec_u8 = src_vec.as_u8();
                let src_vec_size = src_vec_u8.len();
                src_vecs_data[offset..offset + src_vec_size].copy_from_slice(src_vec_u8);
                offset += src_vec_size;
            }
            stats.copy_time += Instant::now() - start_copy_time;

            let start_unmap_time = Instant::now();
            drop(src_mat_data);
            self.src_mat_buffer.unmap();
            drop(src_vecs_data);
            self.src_vecs_buffer.unmap();
            stats.unmap_time += Instant::now() - start_unmap_time;
        } else {
            // Map the buffer.
            let start_map_time = Instant::now();
            let staging_buffer = self.staging_buffer.as_ref().unwrap();
            self.map_buffers(gpu, &[staging_buffer], wgpu::MapMode::Write);
            let mut staging_data = staging_buffer.slice(..).get_mapped_range_mut();
            stats.map_time += Instant::now() - start_map_time;

            // Actually copy the data.
            let start_copy_time = Instant::now();
            let src_mat_u8 = input.src_mat.as_u8_whole();
            let src_mat_size = src_mat_u8.len();
            staging_data[0..src_mat_size].copy_from_slice(src_mat_u8);

            let mut offset = src_mat_size;
            for src_vec in input.src_vecs.iter() {
                let src_vec_u8 = src_vec.as_u8();
                let src_vec_size = src_vec_u8.len();
                staging_data[offset..offset + src_vec_size].copy_from_slice(src_vec_u8);
                offset += src_vec_size;
            }
            let src_vecs_size = offset - src_mat_size;
            stats.copy_time += Instant::now() - start_copy_time;

            let start_unmap_time = Instant::now();
            drop(staging_data);
            staging_buffer.unmap();
            stats.unmap_time = Instant::now() - start_unmap_time;

            // Submit copy staging_buffer -> src_mat_buffer/src_vecs_buffer.
            let submit_start_time = Instant::now();
            let mut encoder = gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(staging_buffer, 0, &self.src_mat_buffer, 0,
                                          src_mat_size as u64);
            encoder.copy_buffer_to_buffer(staging_buffer, src_mat_size as u64,
                                          &self.src_vecs_buffer, 0, src_vecs_size as u64);
            let cmd = encoder.finish();
            gpu.queue.submit(iter::once(cmd));
            stats.submit_time += Instant::now() - submit_start_time;
        }
    }

    // We split the copy buffers and map+read from staging in two separate methods
    // (submit_copy_dst_from_gpu and copy_dst_from_gpu) for two reasons: reduce the amount of
    // GPU -> CPU synchronization and get more accurate statistics for submit vs wait vs copy
    // in GpuStats.
    pub fn submit_copy_dst_from_gpu(&self, gpu: &Gpu, stats: &mut GpuStats) {
        if !self.unified_memory {
            // Submit copy dst_vecs_buffer -> staging_buffer.
            let dst_vecs_size = self.dst_vec_len * self.num_vecs * 4;
            let submit_start_time = Instant::now();
            let mut encoder = gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(&self.dst_vecs_buffer, 0,
                                          self.staging_buffer.as_ref().unwrap(), 0,
                                          dst_vecs_size as u64);
            let cmd = encoder.finish();
            gpu.queue.submit(iter::once(cmd));
            stats.submit_time += Instant::now() - submit_start_time;
        }
    }

    pub fn copy_dst_from_gpu(
        &self,
        gpu: &Gpu,
        input: &mut MatrixVecMultiplyInput,
        stats: &mut GpuStats,
    ) {
        if self.unified_memory {
            let start_map_time = Instant::now();
            self.map_buffers(gpu, &[&self.dst_vecs_buffer], wgpu::MapMode::Read);
            stats.map_time += Instant::now() - start_map_time;

            let start_copy_time = Instant::now();
            let mut dst_vecs_data = self.dst_vecs_buffer.slice(..).get_mapped_range_mut();
            let mut offset = 0;
            for dst_vec in input.dst_vecs.borrow_mut().iter_mut() {
                let dst_vec_u8 = dst_vec.as_u8_mut();
                let dst_vec_size = dst_vec_u8.len();
                dst_vec_u8.copy_from_slice(&mut dst_vecs_data[offset..offset + dst_vec_size]);
                offset += dst_vec_size;
            }
            stats.copy_time += Instant::now() - start_copy_time;

            let start_unmap_time = Instant::now();
            drop(dst_vecs_data);
            self.dst_vecs_buffer.unmap();
            stats.unmap_time += Instant::now() - start_unmap_time;
        } else {
            // Map the buffer.
            let start_map_time = Instant::now();
            let staging_buffer = self.staging_buffer.as_ref().unwrap();
            self.map_buffers(gpu, &[staging_buffer], wgpu::MapMode::Read);
            let mut staging_data = staging_buffer.slice(..).get_mapped_range_mut();
            stats.map_time += Instant::now() - start_map_time;

            // Actually copy the data.
            let start_copy_time = Instant::now();
            let mut offset = 0;
            for dst_vec in input.dst_vecs.borrow_mut().iter_mut() {
                let dst_vec_u8 = dst_vec.as_u8_mut();
                let dst_vec_size = dst_vec_u8.len();
                dst_vec_u8.copy_from_slice(&mut staging_data[offset..offset + dst_vec_size]);
                offset += dst_vec_size;
            }
            stats.copy_time += Instant::now() - start_copy_time;

            let start_unmap_time = Instant::now();
            drop(staging_data);
            staging_buffer.unmap();
            stats.unmap_time = Instant::now() - start_unmap_time;
        }
    }

    pub fn dispatch_main_v1(
        &self,
        gpu: &Gpu,
        pipelines: &MatrixVecGpuPipelines,
        stats: &mut GpuStats,
    ) {
        let submit_start_time = Instant::now();
        let mut encoder = gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipelines.pipeline_v1);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch(self.num_rows, self.num_vecs, 1);
        }
        let cmd = encoder.finish();
        gpu.queue.submit(iter::once(cmd));
        stats.submit_time += Instant::now() - submit_start_time;
    }

    fn map_buffers(&self, gpu: &Gpu, buffers: &[&wgpu::Buffer], map_mode: wgpu::MapMode) {
        let start_time = Instant::now();
        let mut futs: Vec<_> = buffers.iter()
            .map(|b| b.slice(..).map_async(map_mode).boxed())
            .collect();
        // We do not sleep, just poll the device.
        let mut cx = task::Context::from_waker(futures::task::noop_waker_ref());
        while !futs.is_empty() {
            futs.retain_mut(|fut| {
                match Pin::new(fut).poll(&mut cx) {
                    Poll::Ready(result) => {
                        result.expect("Failed to map the buffer");
                        true
                    }
                    Poll::Pending => false,
                }
            });
            gpu.device.poll(wgpu::Maintain::Poll);
            thread::yield_now();
        }
        let polled_for = Instant::now() - start_time;
        if polled_for > Duration::from_millis(50) {
            warn!("Waited for buffer mapping for {:?}", polled_for);
        }
    }
}

// Removes all elements which match the filter from vector. Does not retain the original order.
// Works as Vec::remove(), but passes &mut reference to the filter. Will be replaced with
// drain_filter() as soon as it stabilizes.
trait RetainMutExt<T> {
    fn retain_mut<F: FnMut(&mut T) -> bool>(&mut self, filter: F);
}

impl<T> RetainMutExt<T> for Vec<T> {
    fn retain_mut<F: FnMut(&mut T) -> bool>(&mut self, mut filter: F) {
        let mut i = 0;
        while i < self.len() {
            if filter(&mut self[i]) {
                self.swap_remove(i);
            } else {
                i += 1;
            }
        }
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
    let src_mat_stride = src_mat_slice.len() / src_mat.num_rows();
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
    let src_mat_stride = src_mat_slice.len() / src_mat.num_rows();
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
    let src_mat_stride = src_mat_slice.len() / src_mat.num_rows();
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
    let src_mat_stride = src_mat_slice.len() / src_mat.num_rows();
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

#[derive(Copy, Clone, Debug, PartialEq)]
#[allow(dead_code)]
enum WithCopy {
    None,
    CopyDst,
    CopySrcAndDst,
}

fn bench_gpu_v1(
    gpu: &Gpu,
    gpu_pipelines: &MatrixVecGpuPipelines,
    gpu_input: &MatrixVecMultiplyGpuInput,
    input: &mut MatrixVecMultiplyInput,
    with_copy: WithCopy,
    stats: &mut GpuStats,
) {
    if with_copy == WithCopy::CopySrcAndDst {
        gpu_input.copy_src_to_gpu(gpu, input, stats);
    }
    gpu_input.dispatch_main_v1(gpu, gpu_pipelines, stats);
    if with_copy != WithCopy::None {
        gpu_input.submit_copy_dst_from_gpu(gpu, stats);
    }
    gpu.wait_for_submitted_work_done(stats);
    if with_copy != WithCopy::None {
        gpu_input.copy_dst_from_gpu(gpu, input, stats);
    }

    stats.count += 1;
}

fn matrix_vec_multiply(c: &mut Criterion) {
    let ncpu = init_rayon();
    let integrated_gpu = Gpu::new(false);
    let discrete_gpu = Gpu::new(true);

    let integrated_gpu_pipelines = integrated_gpu.as_ref()
        .map(|gpu| MatrixVecGpuPipelines::new(gpu));
    let discrete_gpu_pipelines = discrete_gpu.as_ref()
        .map(|gpu| MatrixVecGpuPipelines::new(gpu));

    for K in [16usize, 100usize, 128usize, 1000usize, 4000usize] {
        for L in [10usize, 128usize, 1000usize, 4000usize] {
            for M in [1usize, 64usize, 500usize] {
                let mut group = c.benchmark_group(
                    format!("matrix_vec_multiply/size {}x{}, {} vecs", K, L, M));

                // Compute ~throughput in Gflops.
                group.throughput(criterion::Throughput::Elements(
                    K as u64 * L as u64 * M as u64 * 2));

                let mut input = MatrixVecMultiplyInput::new(K, L, M);
                bench_single_thread_matrix_vec_multiply(&input);
                input.store_golden_dst();

                let integrated_unified_gpu_input = integrated_gpu.as_ref().map(|gpu| {
                    MatrixVecMultiplyGpuInput::from(gpu, integrated_gpu_pipelines.as_ref().unwrap(),
                                                    &input, true)
                });
                let integrated_staging_gpu_input = integrated_gpu.as_ref().map(|gpu| {
                    MatrixVecMultiplyGpuInput::from(gpu, integrated_gpu_pipelines.as_ref().unwrap(),
                                                    &input, false)
                });
                let discrete_unified_gpu_input = discrete_gpu.as_ref().map(|gpu| {
                    MatrixVecMultiplyGpuInput::from(gpu, discrete_gpu_pipelines.as_ref().unwrap(),
                                                    &input, true)
                });
                let discrete_staging_gpu_input = discrete_gpu.as_ref().map(|gpu| {
                    MatrixVecMultiplyGpuInput::from(gpu, discrete_gpu_pipelines.as_ref().unwrap(),
                                                    &input, false)
                });

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

                if integrated_gpu.is_some() {
                    run_gpu_bench(
                        &mut group,
                        &integrated_gpu,
                        &integrated_gpu_pipelines,
                        &integrated_unified_gpu_input,
                        &integrated_staging_gpu_input,
                        &mut input,
                        bench_gpu_v1,
                        "v1 gpu integrated",
                    );
                }
                if integrated_gpu.is_some() {
                    run_gpu_bench(
                        &mut group,
                        &discrete_gpu,
                        &discrete_gpu_pipelines,
                        &discrete_unified_gpu_input,
                        &discrete_staging_gpu_input,
                        &mut input,
                        bench_gpu_v1,
                        "v1 gpu discrete",
                    );
                }

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

fn run_gpu_bench<F>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    gpu: &Option<Gpu>,
    gpu_pipelines: &Option<MatrixVecGpuPipelines>,
    unified_gpu_input: &Option<MatrixVecMultiplyGpuInput>,
    staging_gpu_input: &Option<MatrixVecMultiplyGpuInput>,
    input: &mut MatrixVecMultiplyInput,
    bench_fn: F,
    bench_name: &str,
) where F: Fn(
    &Gpu,
    &MatrixVecGpuPipelines,
    &MatrixVecMultiplyGpuInput,
    &mut MatrixVecMultiplyInput,
    WithCopy,
    &mut GpuStats,
)
{
    let gpu = gpu.as_ref().unwrap();
    let pipelines = gpu_pipelines.as_ref().unwrap();

    let unified_input = unified_gpu_input.as_ref().unwrap();
    unified_input.reset_dst(gpu);
    let mut stats = GpuStats::default();
    group.bench_function(format!("{} unified with copy", bench_name), |b| {
        b.iter(|| bench_fn(gpu, pipelines, unified_input, input, WithCopy::CopySrcAndDst,
                           &mut stats));
        input.compare_golden_dst();
    });
    if stats.count > 0 {
        println!("{} unified with copy: {}", bench_name, stats);
    }

    unified_input.reset_dst(gpu);
    let mut stats = GpuStats::default();
    group.bench_function(format!("{} unified no copy", bench_name), |b| {
        b.iter(|| bench_fn(gpu, pipelines, unified_input, input, WithCopy::None, &mut stats));
        unified_input.submit_copy_dst_from_gpu(gpu, &mut GpuStats::default());
        unified_input.copy_dst_from_gpu(gpu, input, &mut GpuStats::default());
        input.compare_golden_dst();
    });
    if stats.count > 0 {
        println!("{} unified no copy: {}", bench_name, stats);
    }

    let staging_input = staging_gpu_input.as_ref().unwrap();
    staging_input.reset_dst(gpu);
    let mut stats = GpuStats::default();
    group.bench_function(format!("{} staging with copy", bench_name), |b| {
        b.iter(|| bench_fn(gpu, pipelines, staging_input, input, WithCopy::CopySrcAndDst,
                           &mut stats));
        input.compare_golden_dst();
    });
    if stats.count > 0 {
        println!("{} staging with copy: {}", bench_name, stats);
    }

    staging_input.reset_dst(gpu);
    let mut stats = GpuStats::default();
    group.bench_function(format!("{} staging no copy", bench_name), |b| {
        b.iter(|| bench_fn(gpu, pipelines, staging_input, input, WithCopy::None, &mut stats));
        staging_input.submit_copy_dst_from_gpu(gpu, &mut GpuStats::default());
        staging_input.copy_dst_from_gpu(gpu, input, &mut GpuStats::default());
        input.compare_golden_dst();
    });
    if stats.count > 0 {
        println!("{} staging no copy: {}", bench_name, stats);
    }
}

fn init_rayon() -> usize {
    env_logger::init();

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
