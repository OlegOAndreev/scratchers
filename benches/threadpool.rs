use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use criterion::{black_box, Criterion, criterion_group, criterion_main};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tokio::task::JoinHandle;
use futures::future::{BoxFuture, FutureExt};

use scratchers::atomic_latch::AtomicLatch;

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
fn rayon_par_iter(job_size: i64, num_jobs: usize, result: i64) {
    let ret: i64 = (0..num_jobs).into_par_iter().map(|i| simple_job(job_size, i) as i64).sum();
    assert_eq!(ret, result);
}

// Spawns each job individually via rayon.
fn rayon_simple(job_size: i64, num_jobs: usize, result: i64) {
    let ret = AtomicI64::new(0);
    rayon::scope(|s| {
        for i in 0..num_jobs {
            let ret = &ret;
            s.spawn(move |_| {
                ret.fetch_add(simple_job(job_size, i) as i64, Ordering::Relaxed);
            });
        }
    });
    assert_eq!(ret.load(Ordering::Relaxed), result);
}

// Spawns each job individually via rayon with spawn_fifo.
fn rayon_fifo(job_size: i64, num_jobs: usize, result: i64) {
    let ret = AtomicI64::new(0);
    rayon::scope_fifo(|s| {
        for i in 0..num_jobs {
            let ret = &ret;
            s.spawn_fifo(move |_| {
                ret.fetch_add(simple_job(job_size, i) as i64, Ordering::Relaxed);
            });
        }
    });
    assert_eq!(ret.load(Ordering::Relaxed), result);
}

// This function is required async lambdas are not stabilized.
async fn tokio_simple_future(job_size: i64, start: usize) -> i64 {
    simple_job(job_size, start) as i64
}

// See comment for tokio_simple_future.
async fn sum_join_handles(handles: Vec<JoinHandle<i64>>) -> i64 {
    let mut ret = 0;
    for handle in handles {
        ret += handle.await.unwrap()
    }
    ret
}

// Spawns each job individually via tokio and await the JoinHandles.
fn tokio_simple(rt: &tokio::runtime::Runtime, job_size: i64, num_jobs: usize, result: i64) {
    let mut handles = Vec::with_capacity(num_jobs);
    for i in 0..num_jobs {
        handles.push(rt.spawn(tokio_simple_future(job_size, i)));
    }
    assert_eq!(rt.block_on(sum_join_handles(handles)), result);
}

struct TokioLatchParams {
    job_size: i64,
    latch: AtomicLatch,
    ret: AtomicI64,
}

// See comment for tokio_simple_future.
async fn tokio_latch_future(params: Arc<TokioLatchParams>, start: usize) {
    params.ret.fetch_add(simple_job(params.job_size, start) as i64, Ordering::Relaxed);
    params.latch.done();
}

// Spawns each job individually via tokio and uses AtomicLatch and AtomicI64 to pass the result from
// the job back.
fn tokio_latch(rt: &tokio::runtime::Runtime, job_size: i64, num_jobs: usize, result: i64) {
    let params = Arc::new(TokioLatchParams {
        job_size,
        latch: AtomicLatch::new(num_jobs as u64),
        ret: AtomicI64::new(0),
    });
    for i in 0..num_jobs {
        rt.spawn(tokio_latch_future(params.clone(), i));
    }
    params.latch.wait();
    assert_eq!(params.ret.load(Ordering::Relaxed), result);
}

fn rayon_chained_job<'a>(s: &rayon::Scope<'a>, job_size: i64, left_jobs: usize, cur: i64, ret: &'a AtomicI64) {
    let v = simple_job(job_size, left_jobs - 1) as i64 + cur;
    if left_jobs == 0 {
        ret.store(v as i64, Ordering::Relaxed);
    } else {
        s.spawn(move |s| rayon_chained_job(s, job_size, left_jobs - 1, v, ret));
    }
}

// Spawns the next job after the previous one completes via rayon.
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
        assert_eq!(ret[i].load(Ordering::Relaxed), result);
    }
}

// Spawns the next job after the previous one completes via tokio.
fn tokio_chained_future(job_size: i64, left_jobs: usize) -> BoxFuture<'static, i64> {
    async move {
        let v = simple_job(job_size, left_jobs - 1) as i64;
        if left_jobs == 1 {
            v
        } else {
            tokio_chained_future(job_size, left_jobs - 1).await + v
        }
    }.boxed()
}

// See comment for tokio_simple_future.
async fn get_handles(handles: Vec<JoinHandle<i64>>) -> Vec<i64> {
    let mut ret = Vec::with_capacity(handles.len());
    for handle in handles {
        ret.push(handle.await.unwrap());
    }
    ret
}

// Spawns the next job after the previous one completes via tokio.
fn tokio_chained(rt: &tokio::runtime::Runtime, job_size: i64, num_jobs: usize, parallelism: usize, result: i64) {
    let mut handles = Vec::with_capacity(num_jobs);
    for _ in 0..parallelism {
        handles.push(rt.spawn(tokio_chained_future(job_size, num_jobs)));
    }
    let ret = rt.block_on(get_handles(handles));
    for v in &ret {
        assert_eq!(*v, result);
    }
}

fn run_simple_bench(c: &mut Criterion, rt: &tokio::runtime::Runtime, job_size: i64) {
    const NUM_JOBS: usize = 10000;

    let mut group = c.benchmark_group(format!("job_size={}", job_size));
    let result = single_threaded(black_box(job_size), NUM_JOBS);
    group.sample_size(10)
        .bench_function("single thread", |b| b.iter(|| single_threaded(black_box(job_size), NUM_JOBS)));
    group.bench_function("rayon par iter", |b| b.iter(|| rayon_par_iter(black_box(job_size), NUM_JOBS, result)));
    group.bench_function("rayon simple", |b| b.iter(|| rayon_simple(black_box(job_size), NUM_JOBS, result)));
    group.bench_function("rayon fifo", |b| b.iter(|| rayon_fifo(black_box(job_size), NUM_JOBS, result)));
    group.bench_function("tokio simple", |b| b.iter(|| tokio_simple(rt, black_box(job_size), NUM_JOBS, result)));
    group.bench_function("tokio latch", |b| b.iter(|| tokio_latch(rt, black_box(job_size), NUM_JOBS, result)));
    group.finish();
}

fn run_chained_bench(c: &mut Criterion, rt: &tokio::runtime::Runtime, job_size: i64, parallelism: usize) {
    const NUM_JOBS: usize = 10000;

    let mut group = c.benchmark_group(format!("chained parallel={} job_size={}", parallelism, job_size));
    let result = single_threaded(black_box(job_size), NUM_JOBS);
    if parallelism == 1 {
        group.sample_size(10)
            .bench_function("single thread", |b| b.iter(|| single_threaded(black_box(job_size), NUM_JOBS)));
    }
    group.bench_function("rayon", |b| b.iter(|| rayon_chained(black_box(job_size), NUM_JOBS, parallelism, result)));
    group.bench_function("tokio", |b| b.iter(|| tokio_chained(rt, black_box(job_size), NUM_JOBS, parallelism, result)));
    group.finish();
}

fn threadpool_simple_benchmark(c: &mut Criterion) {
    let ncpu = num_cpus::get_physical();
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(ncpu)
    //     .build_global()
    //     .unwrap();

    let rt = tokio::runtime::Runtime::new().unwrap();
    run_simple_bench(c, &rt, 10);
    run_simple_bench(c, &rt, 100);
    run_simple_bench(c, &rt, 1000);
    run_simple_bench(c, &rt, 10000);

    run_chained_bench(c, &rt, 10, 1);
    run_chained_bench(c, &rt, 100, 1);
    run_chained_bench(c, &rt, 1000, 1);

    run_chained_bench(c, &rt, 10, ncpu);
    run_chained_bench(c, &rt, 100, ncpu);
    run_chained_bench(c, &rt, 1000, ncpu);
}

criterion_group!(benches, threadpool_simple_benchmark);
criterion_main!(benches);
