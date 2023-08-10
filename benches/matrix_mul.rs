// Benchmark naive gemm implementations.

#![allow(dead_code)]
#![allow(unused_imports)]

use std::sync::atomic;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::distributions::Distribution;
use scratchers::tile_mul::{mul_tile_generic, mul_tile_simd};

static RAYON_GLOBAL_INIT: AtomicBool = AtomicBool::new(false);

const SIZE_AND_COUNT: &[(usize, usize)] = &[
    (128, 1000),
    (1024, 1),
    (1088, 1),
    (4000, 1),
    (8000, 1),
    // (16000, 1),
];

struct MatrixMulInput {
    size: usize,
    mat_1: Vec<Vec<f32>>,
    mat_2: Vec<Vec<f32>>,
    mat_dst: Vec<Vec<f32>>,
}

impl MatrixMulInput {
    fn new(size: usize, count: usize) -> Self {
        let mut rng = rand::thread_rng();
        let rnd = rand::distributions::Uniform::new(0.5f32, 1.5f32);

        let mut mat_l = vec![];
        let mut mat_r = vec![];
        let mut mat_dst = vec![];
        for _ in 0..count {
            let mut ml = vec![0.0f32; size * size];
            let mut mr = vec![0.0f32; size * size];
            for k in 0..size {
                for l in 0..size {
                    ml[k * size + l] = rnd.sample(&mut rng);
                    mr[k * size + l] = rnd.sample(&mut rng);
                }
            }
            mat_l.push(ml);
            mat_r.push(mr);
            mat_dst.push(vec![0.0f32; size * size]);
        }
        Self {
            size,
            mat_1: mat_l,
            mat_2: mat_r,
            mat_dst,
        }
    }

    fn clear(&mut self) {
        for md in &mut self.mat_dst {
            md.fill(0.0);
        }
    }

    fn assert_dst_equals_to(&self, golden_dst: &Vec<Vec<f32>>) {
        let tolerance = self.size as f32 * f32::EPSILON;
        let n = self.size;
        for c in 0..self.mat_dst.len() {
            for i in 0..n {
                for j in 0..n {
                    let m = golden_dst[c][i * n + j].max(self.mat_dst[c][i * n + j]);
                    let d = (golden_dst[c][i * n + j] - self.mat_dst[c][i * n + j]).abs();
                    if d > tolerance * m {
                        assert!(false, "Matrix {} differs, abs: {}", c, d);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "ispc")]
#[link(name = "scratchers_ispc")]
extern "C" {
    pub fn simplest_ispc_mul_impl(
        size: i32,
        src_mat_1: *const f32,
        src_mat_2: *const f32,
        dst_mat: *mut f32,
    );
}

// Simplest algorithm: three nested loops.
fn simplest_mul(input: &mut MatrixMulInput) {
    let n = input.size;
    for c in 0..input.mat_dst.len() {
        let m1 = &input.mat_1[c];
        let m2 = &input.mat_2[c];
        let md = &mut input.mat_dst[c];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0f32;
                for k in 0..n {
                    unsafe {
                        acc += *m1.get_unchecked(i * n + k) * *m2.get_unchecked(k * n + j);
                    }
                }
                md[i * n + j] = acc;
            }
        }
    }
}

// Simplest tiling algorithm: single size tile.
fn simple_tile_mul<const TILE_SIZE: usize>(input: &mut MatrixMulInput) {
    assert_eq!(
        input.size % TILE_SIZE,
        0,
        "input size {} must be divisible by {}",
        input.size,
        TILE_SIZE
    );

    let n = input.size;
    for c in 0..input.mat_dst.len() {
        let m1 = &input.mat_1[c];
        let m2 = &input.mat_2[c];
        let md = &mut input.mat_dst[c];
        md.fill(0.0);
        for ib in (0..n).step_by(TILE_SIZE) {
            for jb in (0..n).step_by(TILE_SIZE) {
                for kb in (0..n).step_by(TILE_SIZE) {
                    mul_tile_generic::<TILE_SIZE>(m1, m2, md, ib, jb, kb, n);
                }
            }
        }
    }
}

// Two tiered tiling algorithm: bigger tiles are from TILE_MUL x TILE_MUL smaller tiles. Bigger
// tiles should fit in L2 cache while smaller tiles should fit into L1 cache.
fn two_tiered_tile_mul<const TILE_SIZE: usize, const TILE_MUL: usize>(input: &mut MatrixMulInput) {
    // Rust does not allow const here =(
    let big_tile_size = TILE_SIZE * TILE_MUL;
    assert_eq!(
        input.size % big_tile_size,
        0,
        "input size {} must be divisible by {}",
        input.size,
        big_tile_size
    );

    let n = input.size;
    for c in 0..input.mat_dst.len() {
        let m1 = &input.mat_1[c];
        let m2 = &input.mat_2[c];
        let md = &mut input.mat_dst[c];
        md.fill(0.0);
        for ib in (0..n).step_by(big_tile_size) {
            for jb in (0..n).step_by(big_tile_size) {
                for kb in (0..n).step_by(big_tile_size) {
                    // Multiply big tiles and add to destination.
                    for it in (ib..ib + big_tile_size).step_by(TILE_SIZE) {
                        for jt in (jb..jb + big_tile_size).step_by(TILE_SIZE) {
                            for kt in (kb..kb + big_tile_size).step_by(TILE_SIZE) {
                                // Multiply inner tiles.
                                mul_tile_generic::<TILE_SIZE>(m1, m2, md, it, jt, kt, n);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Simplest tiling algorithm with SIMD: single size tile.
fn simple_tile_mul_simd<const TILE_SIZE: usize>(input: &mut MatrixMulInput) {
    assert_eq!(
        input.size % TILE_SIZE,
        0,
        "input size {} must be divisible by {}",
        input.size,
        TILE_SIZE
    );

    let n = input.size;
    for c in 0..input.mat_dst.len() {
        let m1 = &input.mat_1[c];
        let m2 = &input.mat_2[c];
        let md = &mut input.mat_dst[c];
        md.fill(0.0);
        for ib in (0..n).step_by(TILE_SIZE) {
            for jb in (0..n).step_by(TILE_SIZE) {
                for kb in (0..n).step_by(TILE_SIZE) {
                    mul_tile_simd::<TILE_SIZE>(m1, m2, md, ib, jb, kb, n);
                }
            }
        }
    }
}

// Simplest algorithm in ISPC: three nested loops.
#[cfg(feature = "ispc")]
fn simplest_ispc_mul(input: &mut MatrixMulInput) {
    let n = input.size;
    for c in 0..input.mat_dst.len() {
        let m1 = &input.mat_1[c];
        let m2 = &input.mat_2[c];
        let md = &mut input.mat_dst[c];
        unsafe {
            simplest_ispc_mul_impl(n as i32, m1.as_ptr(), m2.as_ptr(), md.as_mut_ptr());
        }
    }
}

fn make_golden_out(input: &mut MatrixMulInput) -> Vec<Vec<f32>> {
    simple_tile_mul::<32>(input);
    let ret = input.mat_dst.clone();
    input.clear();
    ret
}

fn matrix_mul_multiply(c: &mut Criterion) {
    let _ncpu = init_rayon();

    for &(size, count) in SIZE_AND_COUNT {
        let mut group =
            c.benchmark_group(format!("matrix_mul_multiply/size {}x{} times", size, count));

        // Compute ~throughput in Gflops (assuming we do fma).
        group.throughput(criterion::Throughput::Elements(
            size as u64 * size as u64 * size as u64 * count as u64,
        ));
        group.sample_size(10);
        group.warm_up_time(Duration::from_nanos(1));

        // Generate matrices.
        let mut input = MatrixMulInput::new(size, count);
        let golden_dst = make_golden_out(&mut input);

        if size <= 1536 {
            group.bench_function("CPU standard single thread", |b| {
                b.iter(|| simplest_mul(&mut input));
                input.assert_dst_equals_to(&golden_dst);
            });

            #[cfg(feature = "ispc")]
            group.bench_function("CPU ISPC standard single thread", |b| {
                b.iter(|| simplest_ispc_mul(&mut input));
                input.assert_dst_equals_to(&golden_dst);
            });

            group.bench_function("CPU simple tile 8x8 single thread", |b| {
                b.iter(|| simple_tile_mul::<8>(&mut input));
                input.assert_dst_equals_to(&golden_dst);
            });

            group.bench_function("CPU simple tile 16x16 single thread", |b| {
                b.iter(|| simple_tile_mul::<16>(&mut input));
                input.assert_dst_equals_to(&golden_dst);
            });

            group.bench_function("CPU simple tile 32x32 single thread", |b| {
                b.iter(|| simple_tile_mul::<32>(&mut input));
                input.assert_dst_equals_to(&golden_dst);
            });

            if size % 128 == 0 {
                group.bench_function("CPU two tiered tile 16x16 in 128x128 single thread", |b| {
                    b.iter(|| two_tiered_tile_mul::<16, 8>(&mut input));
                    input.assert_dst_equals_to(&golden_dst);
                });

                group.bench_function("CPU two tiered tile 32x32 in 128x128 single thread", |b| {
                    b.iter(|| two_tiered_tile_mul::<32, 4>(&mut input));
                    input.assert_dst_equals_to(&golden_dst);
                });
            }

            if size % 256 == 0 {
                group.bench_function("CPU two tiered tile 8x8 in 256x256 single thread", |b| {
                    b.iter(|| two_tiered_tile_mul::<8, 32>(&mut input));
                    input.assert_dst_equals_to(&golden_dst);
                });

                group.bench_function("CPU two tiered tile 16x16 in 256x256 single thread", |b| {
                    b.iter(|| two_tiered_tile_mul::<16, 16>(&mut input));
                    input.assert_dst_equals_to(&golden_dst);
                });

                group.bench_function("CPU two tiered tile 32x32 in 256x256 single thread", |b| {
                    b.iter(|| two_tiered_tile_mul::<32, 8>(&mut input));
                    input.assert_dst_equals_to(&golden_dst);
                });
            }

            if size % 512 == 0 {
                group.bench_function("CPU two tiered tile 32x32 in 512x512 single thread", |b| {
                    b.iter(|| two_tiered_tile_mul::<32, 16>(&mut input));
                    input.assert_dst_equals_to(&golden_dst);
                });
            }
        }

        group.bench_function("CPU simple simd tile 8x8 single thread", |b| {
            b.iter(|| simple_tile_mul_simd::<8>(&mut input));
            input.assert_dst_equals_to(&golden_dst);
        });

        group.bench_function("CPU simple simd tile 16x16 single thread", |b| {
            b.iter(|| simple_tile_mul_simd::<16>(&mut input));
            input.assert_dst_equals_to(&golden_dst);
        });

        group.bench_function("CPU simple simd tile 32x32 single thread", |b| {
            b.iter(|| simple_tile_mul_simd::<32>(&mut input));
            input.assert_dst_equals_to(&golden_dst);
        });

        // group.bench_function("CPU vectorized tile 8x8 single thread", |b| {
        //     b.iter(|| interleaved_tile_mul::<8>(&mut input));
        //     input.assert_dst_equals_to(&golden_dst);
        // });
        //
        // group.bench_function("CPU vectorized tile 16x16 single thread", |b| {
        //     b.iter(|| interleaved_tile_mul::<16>(&mut input));
        //     input.assert_dst_equals_to(&golden_dst);
        // });
        //
        // group.bench_function("CPU vectorized tile 32x32 single thread", |b| {
        //     b.iter(|| interleaved_tile_mul::<32>(&mut input));
        //     input.assert_dst_equals_to(&golden_dst);
        // });
    }
}

fn init_rayon() -> usize {
    let ncpu = if let Ok(v) = std::env::var("NUM_CPUS") {
        v.parse::<usize>().unwrap()
    } else {
        num_cpus::get_physical()
    };
    #[cfg(feature = "ispc")]
    if !RAYON_GLOBAL_INIT.swap(true, atomic::Ordering::SeqCst) {
        rayon::ThreadPoolBuilder::new().num_threads(ncpu).build_global().unwrap();
    }
    ncpu
}

criterion_group!(matrix_mul_benches, matrix_mul_multiply);
criterion_main!(matrix_mul_benches);
