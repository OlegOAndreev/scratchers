#![allow(dead_code)]
#![allow(unused_variables)]

use std::sync::{Arc, atomic};
use std::sync::atomic::{AtomicBool, AtomicI64};

use criterion::{black_box, Criterion, criterion_group, criterion_main};
#[cfg(feature = "futures")]
use futures::{future, future::FutureExt, task::SpawnExt};
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use scratchers::atomic_latch::AtomicLatch;

static RAYON_GLOBAL_INIT: AtomicBool = AtomicBool::new(false);

fn simple_job(job_size: i64, start: usize) -> f64 {
    let mut ret = 0.0;
    for i in 0..job_size {
        ret += ((i + start as i64) as f64).sqrt();
    }
    ret
}

// The simplest case: calls the job in a loop.
fn single_threaded(job_size: i64, num_jobs: usize) -> i64 {
    let mut ret = 0i64;
    for i in 0..num_jobs {
        ret += simple_job(job_size, i) as i64;
    }
    ret
}

// Uses par_iter from rayon.
#[cfg(feature = "rayon")]
fn rayon_par_iter(job_size: i64, num_jobs: usize, result: i64) {
    let ret: i64 = (0..num_jobs).into_par_iter().map(|i| simple_job(job_size, i) as i64).sum();
    assert_eq!(ret, result);
}

// Spawns each job individually via rayon.
#[cfg(feature = "rayon")]
fn rayon_simple(job_size: i64, num_jobs: usize, result: i64) {
    let ret = AtomicI64::new(0);
    rayon::scope(|s| {
        for i in 0..num_jobs {
            let ret = &ret;
            s.spawn(move |_| {
                ret.fetch_add(simple_job(job_size, i) as i64, atomic::Ordering::Relaxed);
            });
        }
    });
    assert_eq!(ret.load(atomic::Ordering::Relaxed), result);
}

// Spawns each job individually via rayon with spawn_fifo.
#[cfg(feature = "rayon")]
fn rayon_fifo(job_size: i64, num_jobs: usize, result: i64) {
    let ret = AtomicI64::new(0);
    rayon::scope_fifo(|s| {
        for i in 0..num_jobs {
            let ret = &ret;
            s.spawn_fifo(move |_| {
                ret.fetch_add(simple_job(job_size, i) as i64, atomic::Ordering::Relaxed);
            });
        }
    });
    assert_eq!(ret.load(atomic::Ordering::Relaxed), result);
}

// This function is required async lambdas are not stabilized.
async fn simple_future(job_size: i64, start: usize) -> i64 {
    simple_job(job_size, start) as i64
}

async fn accum_future(job_size: i64, start: usize, accum: Arc<AtomicI64>, latch: Arc<AtomicLatch>) {
    let v = simple_job(job_size, start) as i64;
    accum.fetch_add(v, atomic::Ordering::Relaxed);
    latch.done();
}

#[cfg(feature = "tokio")]
async fn sum_join_handles(handles: Vec<tokio::task::JoinHandle<i64>>) -> i64 {
    let mut ret = 0;
    for handle in handles {
        ret += handle.await.unwrap()
    }
    ret
}

#[cfg(feature = "futures")]
async fn sum_remote_handles(handles: Vec<future::RemoteHandle<i64>>) -> i64 {
    let mut ret = 0;
    for handle in handles {
        ret += handle.await;
    }
    ret
}

// Spawns each job individually via tokio and await the JoinHandles.
#[cfg(feature = "tokio")]
fn tokio_simple(rt: &tokio::runtime::Runtime, job_size: i64, num_jobs: usize, result: i64) {
    let mut handles = Vec::with_capacity(num_jobs);
    for i in 0..num_jobs {
        handles.push(rt.spawn(simple_future(job_size, i)));
    }
    assert_eq!(rt.block_on(sum_join_handles(handles)), result);
}

#[cfg(feature = "futures")]
fn futures_simple(tp: &futures::executor::ThreadPool, job_size: i64, num_jobs: usize, result: i64) {
    let mut handles = Vec::with_capacity(num_jobs);
    for i in 0..num_jobs {
        handles.push(tp.spawn_with_handle(simple_future(job_size, i)).unwrap());
    }
    assert_eq!(futures::executor::block_on(sum_remote_handles(handles)), result);
}

#[cfg(feature = "futures")]
fn futures_accum(tp: &futures::executor::ThreadPool, job_size: i64, num_jobs: usize, result: i64) {
    let accum = Arc::new(AtomicI64::new(0));
    let latch = Arc::new(AtomicLatch::new(num_jobs as u64));
    for i in 0..num_jobs {
        tp.spawn(accum_future(job_size, i, accum.clone(), latch.clone())).unwrap();
    }
    latch.wait();
    assert_eq!(accum.load(atomic::Ordering::Relaxed), result);
}

#[cfg(feature = "rayon")]
fn rayon_chained_job<'a>(
    s: &rayon::Scope<'a>,
    job_size: i64,
    left_jobs: usize,
    cur: i64,
    ret: &'a AtomicI64,
) {
    let v = simple_job(job_size, left_jobs - 1) as i64 + cur;
    if left_jobs == 0 {
        ret.store(v as i64, atomic::Ordering::Relaxed);
    } else {
        s.spawn(move |s| rayon_chained_job(s, job_size, left_jobs - 1, v, ret));
    }
}

// Spawns the next job after the previous one completes via rayon.
#[cfg(feature = "rayon")]
fn rayon_chained(job_size: i64, num_jobs: usize, parallelism: usize, result: i64) {
    let mut ret = vec![];
    for _ in 0..parallelism {
        ret.push(AtomicI64::new(0));
    }
    rayon::scope(|s| {
        for i in 0..parallelism {
            let ret = &ret[i];
            s.spawn(move |s| rayon_chained_job(s, job_size, num_jobs, 0, ret));
        }
    });
    for i in 0..parallelism {
        assert_eq!(ret[i].load(atomic::Ordering::Relaxed), result);
    }
}

// Spawns the next job after the previous one completes via tokio.
#[cfg(feature = "futures")]
fn chained_future(job_size: i64, left_jobs: usize) -> future::BoxFuture<'static, i64> {
    async move {
        let v = simple_job(job_size, left_jobs - 1) as i64;
        if left_jobs == 1 {
            v
        } else {
            chained_future(job_size, left_jobs - 1).await + v
        }
    }
    .boxed()
}

#[cfg(feature = "tokio")]
async fn get_join_handles(handles: Vec<tokio::task::JoinHandle<i64>>) -> Vec<i64> {
    let mut ret = Vec::with_capacity(handles.len());
    for handle in handles {
        ret.push(handle.await.unwrap());
    }
    ret
}

#[cfg(feature = "futures")]
async fn get_remote_handles(handles: Vec<future::RemoteHandle<i64>>) -> Vec<i64> {
    let mut ret = Vec::with_capacity(handles.len());
    for handle in handles {
        ret.push(handle.await);
    }
    ret
}

// Spawns the next job after the previous one completes via tokio.
#[cfg(feature = "tokio")]
fn tokio_chained(
    rt: &tokio::runtime::Runtime,
    job_size: i64,
    num_jobs: usize,
    parallelism: usize,
    result: i64,
) {
    let mut handles = Vec::with_capacity(num_jobs);
    for _ in 0..parallelism {
        handles.push(rt.spawn(chained_future(job_size, num_jobs)));
    }
    let ret = rt.block_on(get_join_handles(handles));
    for v in &ret {
        assert_eq!(*v, result);
    }
}

// Spawns the next job after the previous one completes via futures.
#[cfg(feature = "futures")]
fn futures_chained(
    tp: &futures::executor::ThreadPool,
    job_size: i64,
    num_jobs: usize,
    parallelism: usize,
    result: i64,
) {
    let mut handles = Vec::with_capacity(num_jobs);
    for _ in 0..parallelism {
        handles.push(tp.spawn_with_handle(chained_future(job_size, num_jobs)).unwrap());
    }
    let ret = futures::executor::block_on(get_remote_handles(handles));
    for v in &ret {
        assert_eq!(*v, result);
    }
}

struct Runtimes {
    #[cfg(feature = "tokio")]
    rt: tokio::runtime::Runtime,
    #[cfg(feature = "futures")]
    tp: futures::executor::ThreadPool,
}

fn run_simple_bench(
    c: &mut Criterion,
    r: &Runtimes,
    job_size: i64,
    ncpu: usize,
) {
    const NUM_JOBS: usize = 10000;

    let mut group =
        c.benchmark_group(format!("threadpool/job_size={} num_cpus={}", job_size, ncpu));
    let result = single_threaded(black_box(job_size), NUM_JOBS);
    group.sample_size(10).bench_function("single thread", |b| {
        b.iter(|| single_threaded(black_box(job_size), NUM_JOBS))
    });
    #[cfg(feature = "rayon")]
    group.bench_function("rayon par iter", |b| {
        b.iter(|| rayon_par_iter(black_box(job_size), NUM_JOBS, result))
    });
    #[cfg(feature = "rayon")]
    group.bench_function("rayon simple", |b| {
        b.iter(|| rayon_simple(black_box(job_size), NUM_JOBS, result))
    });
    #[cfg(feature = "rayon")]
    group.bench_function("rayon fifo", |b| {
        b.iter(|| rayon_fifo(black_box(job_size), NUM_JOBS, result))
    });
    #[cfg(feature = "tokio")]
    group.bench_function("tokio simple", |b| {
        b.iter(|| tokio_simple(&r.rt, black_box(job_size), NUM_JOBS, result))
    });
    #[cfg(feature = "futures")]
    group.bench_function("futures simple", |b| {
        b.iter(|| futures_simple(&r.tp, black_box(job_size), NUM_JOBS, result))
    });
    #[cfg(feature = "futures")]
    group.bench_function("futures accum", |b| {
        b.iter(|| futures_accum(&r.tp, black_box(job_size), NUM_JOBS, result))
    });
    group.finish();
}

fn run_chained_bench(
    c: &mut Criterion,
    r: &Runtimes,
    job_size: i64,
    parallelism: usize,
    ncpu: usize,
) {
    const NUM_JOBS: usize = 10000;

    let mut group = c.benchmark_group(format!(
        "threadpool/chained parallel={} job_size={} num_cpus={}",
        parallelism, job_size, ncpu
    ));
    let result = single_threaded(black_box(job_size), NUM_JOBS);
    if parallelism == 1 {
        group.sample_size(10).bench_function("single thread", |b| {
            b.iter(|| single_threaded(black_box(job_size), NUM_JOBS))
        });
    }
    #[cfg(feature = "rayon")]
    group.bench_function("rayon", |b| {
        b.iter(|| rayon_chained(black_box(job_size), NUM_JOBS, parallelism, result))
    });
    #[cfg(feature = "tokio")]
    group.bench_function("tokio", |b| {
        b.iter(|| tokio_chained(&r.rt, black_box(job_size), NUM_JOBS, parallelism, result))
    });
    #[cfg(feature = "futures")]
    group.bench_function("futures", |b| {
        b.iter(|| futures_chained(&r.tp, black_box(job_size), NUM_JOBS, parallelism, result))
    });
    group.finish();
}

// Benchmarks several threadpools by running simple_job() with varying size.
// Spawns separate tasks:
//  * single-thread
//  * rayon par_iter()
//  * rayon spawn()
//  * rayon spawn_fifo()
//  * tokio spawn + .await
fn threadpool_simple_benchmark(c: &mut Criterion) {
    run_threadpool_benchmark(|r, ncpu| {
        run_simple_bench(c, r, 10, ncpu);
        run_simple_bench(c, r, 100, ncpu);
        run_simple_bench(c, r, 1000, ncpu);
        run_simple_bench(c, r, 10000, ncpu);
    });
}

// Spawns several (1 or num_cpus) chained tasks (each task spawns after the previous one):
//  * single-thread
//  * rayon spawn()
//  * tokio spawn + .await
fn threadpool_chained_benchmark(c: &mut Criterion) {
    run_threadpool_benchmark(|r, ncpu| {
        run_chained_bench(c, r, 10, 1, ncpu);
        run_chained_bench(c, r, 100, 1, ncpu);
        run_chained_bench(c, r, 1000, 1, ncpu);

        run_chained_bench(c, r, 10, ncpu, ncpu);
        run_chained_bench(c, r, 100, ncpu, ncpu);
        run_chained_bench(c, r, 1000, ncpu, ncpu);
    });
}

fn run_threadpool_benchmark<F>(f: F)
where
    F: FnOnce(&Runtimes, usize),
{
    let ncpu = if let Ok(v) = std::env::var("NUM_CPUS") {
        v.parse::<usize>().unwrap()
    } else {
        // The simple_job is CPU-bound, not memory-bound, so limit the number of CPUs to physical
        // CPUs.
        num_cpus::get_physical()
    };
    #[cfg(feature = "rayon")]
    if !RAYON_GLOBAL_INIT.swap(true, atomic::Ordering::SeqCst) {
        rayon::ThreadPoolBuilder::new().num_threads(ncpu).build_global().unwrap();
    }
    #[cfg(feature = "tokio")]
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(ncpu)
        .build()
        .unwrap();
    #[cfg(feature = "futures")]
    let tp = futures::executor::ThreadPoolBuilder::new().pool_size(ncpu).create().unwrap();
    let r = Runtimes{
        #[cfg(feature = "tokio")]
        rt,
        #[cfg(feature = "futures")]
        tp,
    };
    f(&r, ncpu);
}

criterion_group!(benches, threadpool_simple_benchmark, threadpool_chained_benchmark);
criterion_main!(benches);
