use std::{any, ops};
use std::collections::HashMap;
use std::fmt::Debug;

use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use criterion::measurement::WallTime;
use rand::Rng;

pub fn sparse_vec_f32_benchmark(c: &mut Criterion) {
    sparse_vec_benchmark::<f32>(c);
}

pub fn sparse_vec_f64_benchmark(c: &mut Criterion) {
    sparse_vec_benchmark::<f64>(c);
}

fn sparse_vec_benchmark<T>(c: &mut Criterion)
where
    T: Debug,
    T: Default,
    T: One,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
    T: PartialEq,
{
    let prefix = format!("sparse_vec_{}", any::type_name::<T>());

    let data = generate_uniform_vec::<T>(10, 0.25, 5000);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform 0.25/10 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_uniform_vec::<T>(100, 0.25, 1000);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform 0.25/100 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_uniform_vec::<T>(1000, 0.25, 200);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform 0.25/1000 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_uniform_vec::<T>(1000, 0.1, 200);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform 0.1/1000 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_uniform_vec::<T>(10000, 0.1, 100);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform 0.1/10000 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_uniform_vec::<T>(10000, 0.01, 100);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform 0.01/10000 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_uniform_vec::<T>(100000, 0.01, 50);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform 0.01/100000 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_uniform_vec::<T>(1000000, 0.001, 25);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform 0.01/1000000 elem");
        run_sparse_vec_benchmark(group, data);
    }

    let data = generate_uniform_vec::<T>(10, 1.0, 5000);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform dense/10 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_uniform_vec::<T>(100, 1.0, 1000);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform dense/100 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_uniform_vec::<T>(1000, 1.0, 200);
    {
        let group = c.benchmark_group(prefix.clone() + "/uniform dense/1000 elem");
        run_sparse_vec_benchmark(group, data);
    }

    let data = generate_exp_vec::<T>(10, 5000);
    {
        let group = c.benchmark_group(prefix.clone() + "/exp/10 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_exp_vec::<T>(100, 100);
    {
        let group = c.benchmark_group(prefix.clone() + "/exp/100 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_exp_vec::<T>(1000, 200);
    {
        let group = c.benchmark_group(prefix.clone() + "/exp/1000 elem");
        run_sparse_vec_benchmark(group, data);
    }
    let data = generate_exp_vec::<T>(10000, 100);
    {
        let group = c.benchmark_group(prefix.clone() + "/exp/10000 elem");
        run_sparse_vec_benchmark(group, data);
    }
}

fn run_sparse_vec_benchmark<T>(mut group: BenchmarkGroup<WallTime>, data: Vec<Vec<T>>)
where
    T: Debug,
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
    T: PartialEq,
{
    let base_input = prepare_base_input(&data);
    let sparse_input_u32 = prepare_sparse_input::<T, u32>(&data);
    let sparse_input_u64 = prepare_sparse_input::<T, u64>(&data);
    let sparse_input_soa_u32 = prepare_sparse_input_soa::<T, u32>(&data);
    let sparse_input_soa_u64 = prepare_sparse_input_soa::<T, u64>(&data);
    let hashmap_input = prepare_hashmap_input::<T>(&data);

    let target = bench_base(&base_input);
    assert_eq!(target, bench_base_iter(&base_input));
    assert_eq!(target, bench_base_unsafe(&base_input));
    assert_eq!(target, bench_sparse(&sparse_input_u32));
    assert_eq!(target, bench_sparse_unsafe(&sparse_input_u32));
    assert_eq!(target, bench_sparse(&sparse_input_u64));
    assert_eq!(target, bench_sparse_unsafe(&sparse_input_u64));
    assert_eq!(target, bench_sparse_soa(&sparse_input_soa_u32));
    assert_eq!(target, bench_sparse_soa_unsafe(&sparse_input_soa_u32));
    assert_eq!(target, bench_sparse_soa(&sparse_input_soa_u64));
    assert_eq!(target, bench_sparse_soa_unsafe(&sparse_input_soa_u64));
    assert_eq!(target, bench_hashmap(&hashmap_input));

    group.sample_size(10);

    group.bench_function("base", |b| b.iter(|| bench_base(&base_input)));
    group.bench_function("base zip", |b| b.iter(|| bench_base_iter(&base_input)));
    group.bench_function("base unsafe", |b| b.iter(|| bench_base_unsafe(&base_input)));

    group.bench_function("sparse u32", |b| b.iter(|| bench_sparse(&sparse_input_u32)));
    group
        .bench_function("sparse u32 unsafe", |b| b.iter(|| bench_sparse_unsafe(&sparse_input_u32)));
    group.bench_function("sparse sentinel u32 unsafe", |b| {
        b.iter(|| bench_sparse_sentinel_unsafe(&sparse_input_u32))
    });
    group.bench_function("sparse soa u32", |b| b.iter(|| bench_sparse_soa(&sparse_input_soa_u32)));
    group.bench_function("sparse soa u32 unsafe", |b| {
        b.iter(|| bench_sparse_soa_unsafe(&sparse_input_soa_u32))
    });
    group.bench_function("sparse soa sentinel u32 unsafe", |b| {
        b.iter(|| bench_sparse_soa_sentinel_unsafe(&sparse_input_soa_u32))
    });

    group.bench_function("sparse u64", |b| b.iter(|| bench_sparse(&sparse_input_u64)));
    group
        .bench_function("sparse u64 unsafe", |b| b.iter(|| bench_sparse_unsafe(&sparse_input_u64)));
    group.bench_function("sparse soa u64", |b| b.iter(|| bench_sparse_soa(&sparse_input_soa_u64)));
    group.bench_function("sparse soa u64 unsafe", |b| {
        b.iter(|| bench_sparse_soa_unsafe(&sparse_input_soa_u64))
    });

    group.bench_function("hashmap", |b| b.iter(|| bench_hashmap(&hashmap_input)));
}

struct BaseInput<T> {
    vecs: Vec<Box<[T]>>,
}

fn prepare_base_input<T>(data: &Vec<Vec<T>>) -> BaseInput<T>
where
    T: Clone,
{
    let mut vecs = Vec::with_capacity(data.len());
    for vec in data {
        vecs.push(vec.clone().into_boxed_slice());
    }
    BaseInput { vecs }
}

fn bench_base<T>(input: &BaseInput<T>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
{
    let mut ret = T::default();
    let n = input.vecs.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            let vec1 = &input.vecs[i];
            let vec2 = &input.vecs[j];
            let l = vec1.len().min(vec2.len());
            for k in 0..l {
                a += vec1[k] * vec2[k];
            }
            ret += a;
        }
    }
    ret
}

fn bench_base_iter<T>(input: &BaseInput<T>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
{
    let mut ret = T::default();
    let n = input.vecs.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            for (&v1, &v2) in input.vecs[i].iter().zip(input.vecs[j].iter()) {
                a += v1 * v2;
            }
            ret += a;
        }
    }
    ret
}

fn bench_base_unsafe<T>(input: &BaseInput<T>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
{
    let mut ret = T::default();
    let n = input.vecs.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            let vec1 = &input.vecs[i];
            let vec2 = &input.vecs[j];
            let l = vec1.len().min(vec2.len());
            unsafe {
                for k in 0..l {
                    a += *vec1.get_unchecked(k) * *vec2.get_unchecked(k);
                }
            }
            ret += a;
        }
    }
    ret
}

struct SparseInput<T, I> {
    vecs: Vec<Box<[(I, T)]>>,
}

fn prepare_sparse_input<T, I>(data: &Vec<Vec<T>>) -> SparseInput<T, I>
where
    T: Default,
    T: Copy,
    T: PartialEq,
    I: TryFrom<usize>,
    <I as TryFrom<usize>>::Error: Debug,
    I: MaxValue,
{
    let mut vecs = Vec::with_capacity(data.len());
    for vec in data {
        let mut sparse = vec![];
        for (i, v) in vec.iter().enumerate() {
            if *v != T::default() {
                sparse.push((I::try_from(i).unwrap(), *v));
            }
        }
        sparse.push((I::max_value(), T::default()));
        vecs.push(sparse.into_boxed_slice());
    }
    SparseInput { vecs }
}

fn bench_sparse<T, I>(input: &SparseInput<T, I>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
    I: Copy,
    I: Ord,
    I: MaxValue,
{
    let mut ret = T::default();
    let n = input.vecs.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            let vec1 = &input.vecs[i];
            let vec2 = &input.vecs[j];
            let l1 = vec1.len();
            let l2 = vec2.len();
            if l1 == 0 || l2 == 0 {
                continue;
            }
            let mut idx1 = 0;
            let mut idx2 = 0;
            let mut sidx1 = vec1[idx1].0;
            let mut sidx2 = vec2[idx2].0;
            loop {
                if sidx1 < sidx2 {
                    idx1 += 1;
                    if idx1 == l1 {
                        break;
                    }
                    sidx1 = vec1[idx1].0;
                } else if sidx1 > sidx2 {
                    idx2 += 1;
                    if idx2 == l2 {
                        break;
                    }
                    sidx2 = vec2[idx2].0;
                } else {
                    a += vec1[idx1].1 * vec2[idx2].1;
                    idx1 += 1;
                    idx2 += 1;
                    if idx1 == l1 || idx2 == l2 {
                        break;
                    }
                    sidx1 = vec1[idx1].0;
                    sidx2 = vec2[idx2].0;
                }
            }
            ret += a;
        }
    }
    ret
}

fn bench_sparse_unsafe<T, I>(input: &SparseInput<T, I>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
    I: Copy,
    I: Ord,
    I: MaxValue,
{
    let mut ret = T::default();
    let n = input.vecs.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            let vec1 = &input.vecs[i];
            let vec2 = &input.vecs[j];
            let l1 = vec1.len();
            let l2 = vec2.len();
            if l1 == 0 || l2 == 0 {
                continue;
            }
            let mut idx1 = 0;
            let mut idx2 = 0;
            unsafe {
                let mut sidx1 = vec1.get_unchecked(idx1).0;
                let mut sidx2 = vec2.get_unchecked(idx2).0;
                loop {
                    if sidx1 < sidx2 {
                        idx1 += 1;
                        if idx1 == l1 {
                            break;
                        }
                        sidx1 = vec1.get_unchecked(idx1).0;
                    } else if sidx1 > sidx2 {
                        idx2 += 1;
                        if idx2 == l2 {
                            break;
                        }
                        sidx2 = vec2.get_unchecked(idx2).0;
                    } else {
                        a += vec1.get_unchecked(idx1).1 * vec2.get_unchecked(idx2).1;
                        idx1 += 1;
                        idx2 += 1;
                        if idx1 == l1 || idx2 == l2 {
                            break;
                        }
                        sidx1 = vec1.get_unchecked(idx1).0;
                        sidx2 = vec2.get_unchecked(idx2).0;
                    }
                }
            }
            ret += a;
        }
    }
    ret
}

fn bench_sparse_sentinel_unsafe<T, I>(input: &SparseInput<T, I>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
    I: Copy,
    I: Ord,
    I: MaxValue,
{
    let mut ret = T::default();
    let n = input.vecs.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            let vec1 = &input.vecs[i];
            let vec2 = &input.vecs[j];
            unsafe {
                let base1 = vec1.as_ptr();
                let base2 = vec2.as_ptr();
                let mut ptr1 = base1;
                let mut ptr2 = base2;
                let mut sidx1 = (*ptr1).0;
                let mut sidx2 = (*ptr2).0;
                loop {
                    if sidx1 < sidx2 {
                        ptr1 = ptr1.wrapping_add(1);
                        sidx1 = (*ptr1).0;
                    } else if sidx1 > sidx2 {
                        ptr2 = ptr2.wrapping_add(1);
                        sidx2 = (*ptr2).0;
                    } else {
                        if sidx1 == I::max_value() {
                            break;
                        }
                        a += (*ptr1).1 * (*ptr2).1;
                        ptr1 = ptr1.wrapping_add(1);
                        ptr2 = ptr2.wrapping_add(1);
                        sidx1 = (*ptr1).0;
                        sidx2 = (*ptr2).0;
                    }
                }
            }
            ret += a;
        }
    }
    ret
}

struct SparseInputSoa<T, I> {
    vecs: Vec<(Box<[I]>, Box<[T]>)>,
}

fn prepare_sparse_input_soa<T, I>(data: &Vec<Vec<T>>) -> SparseInputSoa<T, I>
where
    T: Default,
    T: Copy,
    T: PartialEq,
    I: TryFrom<usize>,
    <I as TryFrom<usize>>::Error: Debug,
    I: MaxValue,
{
    let mut vecs = Vec::with_capacity(data.len());
    for vec in data {
        let mut indices = vec![];
        let mut values = vec![];
        for (i, v) in vec.iter().enumerate() {
            if *v != T::default() {
                indices.push(I::try_from(i).unwrap());
                values.push(*v);
            }
        }
        indices.push(I::max_value());
        values.push(T::default());
        vecs.push((indices.into_boxed_slice(), values.into_boxed_slice()));
    }
    SparseInputSoa { vecs }
}

fn bench_sparse_soa<T, I>(input: &SparseInputSoa<T, I>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
    I: Copy,
    I: Ord,
    I: MaxValue,
{
    let mut ret = T::default();
    let n = input.vecs.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            let (indices1, values1) = &input.vecs[i];
            let (indices2, values2) = &input.vecs[j];
            let l1 = indices1.len();
            let l2 = indices2.len();
            if l1 == 0 || l2 == 0 {
                continue;
            }
            let mut idx1 = 0;
            let mut idx2 = 0;
            let mut sidx1 = indices1[idx1];
            let mut sidx2 = indices2[idx2];
            loop {
                if sidx1 < sidx2 {
                    idx1 += 1;
                    if idx1 == l1 {
                        break;
                    }
                    sidx1 = indices1[idx1];
                } else if sidx1 > sidx2 {
                    idx2 += 1;
                    if idx2 == l2 {
                        break;
                    }
                    sidx2 = indices2[idx2];
                } else {
                    a += values1[idx1] * values2[idx2];
                    idx1 += 1;
                    idx2 += 1;
                    if idx1 == l1 || idx2 == l2 {
                        break;
                    }
                    sidx1 = indices1[idx1];
                    sidx2 = indices2[idx2];
                }
            }
            ret += a;
        }
    }
    ret
}

fn bench_sparse_soa_unsafe<T, I>(input: &SparseInputSoa<T, I>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
    I: Copy,
    I: Ord,
    I: MaxValue,
{
    let mut ret = T::default();
    let n = input.vecs.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            let (indices1, values1) = &input.vecs[i];
            let (indices2, values2) = &input.vecs[j];
            let l1 = indices1.len();
            let l2 = indices2.len();
            if l1 == 0 || l2 == 0 {
                continue;
            }
            let mut idx1 = 0;
            let mut idx2 = 0;
            unsafe {
                let mut sidx1 = *indices1.get_unchecked(idx1);
                let mut sidx2 = *indices2.get_unchecked(idx2);
                loop {
                    if sidx1 < sidx2 {
                        idx1 += 1;
                        if idx1 == l1 {
                            break;
                        }
                        sidx1 = *indices1.get_unchecked(idx1);
                    } else if sidx1 > sidx2 {
                        idx2 += 1;
                        if idx2 == l2 {
                            break;
                        }
                        sidx2 = *indices2.get_unchecked(idx2);
                    } else {
                        a += *values1.get_unchecked(idx1) * *values2.get_unchecked(idx2);
                        idx1 += 1;
                        idx2 += 1;
                        if idx1 == l1 || idx2 == l2 {
                            break;
                        }
                        sidx1 = *indices1.get_unchecked(idx1);
                        sidx2 = *indices2.get_unchecked(idx2);
                    }
                }
            }
            ret += a;
        }
    }
    ret
}

fn bench_sparse_soa_sentinel_unsafe<T, I>(input: &SparseInputSoa<T, I>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
    I: Copy,
    I: Ord,
    I: MaxValue,
{
    let mut ret = T::default();
    let n = input.vecs.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            let (indices1, values1) = &input.vecs[i];
            let (indices2, values2) = &input.vecs[j];
            unsafe {
                let base1 = indices1.as_ptr();
                let base2 = indices2.as_ptr();
                let mut ptr1 = base1;
                let mut ptr2 = base2;
                let mut sidx1 = *ptr1;
                let mut sidx2 = *ptr2;
                loop {
                    if sidx1 < sidx2 {
                        ptr1 = ptr1.wrapping_add(1);
                        sidx1 = *ptr1;
                    } else if sidx1 > sidx2 {
                        ptr2 = ptr2.wrapping_add(1);
                        sidx2 = *ptr2;
                    } else {
                        if sidx1 == I::max_value() {
                            break;
                        }
                        let idx1 = ptr1.offset_from(base1) as usize;
                        let idx2 = ptr2.offset_from(base2) as usize;
                        a += *values1.get_unchecked(idx1) * *values2.get_unchecked(idx2);
                        ptr1 = ptr1.wrapping_add(1);
                        ptr2 = ptr2.wrapping_add(1);
                        sidx1 = *ptr1;
                        sidx2 = *ptr2;
                    }
                }
            }
            ret += a;
        }
    }
    ret
}

struct MapInput<T> {
    maps: Vec<HashMap<usize, T>>,
}

fn prepare_hashmap_input<T>(data: &Vec<Vec<T>>) -> MapInput<T>
where
    T: Default,
    T: Copy,
    T: PartialEq,
{
    let mut maps = Vec::with_capacity(data.len());
    for vec in data {
        let mut map = HashMap::new();
        for (i, v) in vec.iter().enumerate() {
            if *v != T::default() {
                map.insert(i, *v);
            }
        }
        maps.push(map);
    }
    MapInput { maps }
}

fn bench_hashmap<T>(input: &MapInput<T>) -> T
where
    T: Default,
    T: Copy,
    T: ops::AddAssign,
    T: ops::Mul<Output = T>,
{
    let mut ret = T::default();
    let n = input.maps.len();
    for i in 0..n {
        for j in 0..n {
            let mut a = T::default();
            let map1 = &input.maps[i];
            let map2 = &input.maps[j];
            for (k, v1) in map1.iter() {
                if let Some(v2) = map2.get(k) {
                    a += *v1 * *v2;
                }
            }
            ret += a;
        }
    }
    ret
}

fn generate_uniform_vec<T>(size: usize, fill_rate: f64, num_vecs: usize) -> Vec<Vec<T>>
where
    T: Default,
    T: One,
{
    let mut ret = Vec::with_capacity(num_vecs);
    for _ in 0..num_vecs {
        let mut vec = Vec::with_capacity(size);
        for _ in 0..size {
            if rand::random::<f64>() < fill_rate {
                vec.push(T::one());
            } else {
                vec.push(T::default());
            }
        }
        ret.push(vec);
    }
    ret
}

fn generate_exp_vec<T>(size: usize, num_vecs: usize) -> Vec<Vec<T>>
where
    T: Default,
    T: One,
{
    const BASE: f64 = 0.75;
    let mut ret = Vec::with_capacity(num_vecs);
    for _ in 0..num_vecs {
        let mut vec = Vec::with_capacity(size);
        let center = rand::thread_rng().gen_range(0..size);
        for i in 0..size {
            if rand::random::<f64>() < BASE.powi(i.abs_diff(center) as i32) {
                vec.push(T::one());
            } else {
                vec.push(T::default());
            }
        }
        ret.push(vec);
    }
    ret
}

trait One {
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

trait MaxValue {
    fn max_value() -> Self;
}

impl MaxValue for u32 {
    fn max_value() -> Self {
        u32::MAX
    }
}

impl MaxValue for u64 {
    fn max_value() -> Self {
        u64::MAX
    }
}

criterion_group!(sparse_vec_benches, sparse_vec_f32_benchmark, sparse_vec_f64_benchmark);
criterion_main!(sparse_vec_benches);

// fn main() {
//     let data = generate_uniform_vec::<f32>(10000, 0.01, 50);
//     let input = prepare_sparse_input_soa::<f32, u32>(&data);
//     let start = std::time::Instant::now();
//     for _ in 0..10000 {
//         criterion::black_box(bench_sparse_soa_sentinel_unsafe(&input));
//         // criterion::black_box(bench_sparse_unsafe(&input));
//     }
//     println!("Got {:?}", std::time::Instant::now() - start);
// }
