use std::fs;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use criterion::{Criterion, criterion_group, criterion_main};
use rodio::Source;

const NUM_CHANNELS: i32 = 2;
// All files in data/ are in 44.1kHz.
const SRC_RATE: i32 = 44100;
// Many modern audio outputs have a 48kHz rate.
const ALT_RATE: i32 = 48000;
// Mix for two seconds.
const MIX_INTERVAL: i32 = 2;

// cpal::Device is not a trait, so we cannot use rodio::OutputStream and have to manually
// construct dynamic mixer with f32 type (see stream.rs in rodio)
struct RodioMixer {
    mixer: rodio::dynamic_mixer::DynamicMixer<f32>,
    controller: Arc<rodio::dynamic_mixer::DynamicMixerController<f32>>,
}

type RodioSource = dyn rodio::Source<Item=f32> + Send;

fn make_rodio_source(data: &Arc<[u8]>) -> Box<RodioSource> {
    // Copy from rodio::Sink::append(). We ignore all the sync costs as they should generally
    // be negligible (only run at the start and the end of playing the sound).
    Box::new(rodio::Decoder::new(Cursor::new(data.clone())).unwrap()
        .pausable(false)
        .amplify(1.0)
        .stoppable()
        .convert_samples::<f32>())
}

struct BenchRodioInput {
    mixer: RodioMixer,
    src_data: Vec<Arc<[u8]>>,
    num_srcs: usize,
    out: Vec<(f32, f32)>,
}

fn bench_play_rodio(mut input: BenchRodioInput) {
    for i in 0..input.num_srcs {
        let src = &input.src_data[i % input.src_data.len()];
        input.mixer.controller.add(make_rodio_source(src));
    }

    for i in 0..input.out.len() {
        match input.mixer.mixer.next() {
            None => { input.out[i].0 = 0.0; }
            Some(v) => { input.out[i].0 = v; }
        }
        match input.mixer.mixer.next() {
            None => { input.out[i].1 = 0.0; }
            Some(v) => { input.out[i].1 = v; }
        }
    }

    // write_to_file(input.mixer.mixer.sample_rate(), &input.out);
}

struct BenchRg3dInput {
    engine: Arc<Mutex<rg3d_sound::engine::SoundEngine>>,
    context: rg3d_sound::context::SoundContext,
    buffers: Vec<rg3d_sound::buffer::SoundBufferResource>,
    num_srcs: usize,
    out: Vec<(f32, f32)>,
}

fn bench_play_rg3d(mut input: BenchRg3dInput) {
    for i in 0..input.num_srcs {
        let source = rg3d_sound::source::generic::GenericSourceBuilder::new()
            .with_buffer(input.buffers[i % input.buffers.len()].clone())
            .with_status(rg3d_sound::source::Status::Playing)
            .build_source()
            .unwrap();
        input.context.state().add_source(source);
    }

    input.engine.lock().unwrap().render(&mut input.out);

    // write_to_file(rg3d_sound::context::SAMPLE_RATE, &input.out);
}

#[allow(dead_code)]
fn write_to_file(rate: u32, buf: &[(f32, f32)]) {
    let wav_spec = hound::WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut wav_writer = hound::WavWriter::create("debug.wav", wav_spec).unwrap();

    for &(l, r) in buf.iter() {
        wav_writer.write_sample(l).unwrap();
        wav_writer.write_sample(r).unwrap();
    }

    wav_writer.finalize().unwrap();
}

// Benchmarks audio library mixers.
fn audio_mixer_wav_benchmark(c: &mut Criterion) {
    let beep_data: Vec<u8> = fs::read("benches/data/beep.wav").unwrap();
    let beep2_data: Vec<u8> = fs::read("benches/data/beep2.wav").unwrap();

    {
        let mut group = c.benchmark_group("audio/empty");
        group.bench_function("rodio", |b| b.iter_batched(
            || prepare_rodio_input(SRC_RATE, vec![], 0),
            |input| bench_play_rodio(input),
            criterion::BatchSize::SmallInput));
        group.bench_function("rg3d", |b| b.iter_batched(
            || prepare_rg3d_input(SRC_RATE, vec![], 0),
            |input| bench_play_rg3d(input),
            criterion::BatchSize::SmallInput));
    }

    {
        let mut group = c.benchmark_group("audio/1 wav/resample=no");
        group.bench_function("rodio", |b| b.iter_batched(
            || prepare_rodio_input(SRC_RATE, vec![&beep_data, &beep2_data], 1),
            |input| bench_play_rodio(input),
            criterion::BatchSize::SmallInput));
        group.bench_function("rg3d", |b| b.iter_batched(
            || prepare_rg3d_input(SRC_RATE, vec![&beep_data, &beep2_data], 1),
            |input| bench_play_rg3d(input),
            criterion::BatchSize::SmallInput));
    }
    {
        let mut group = c.benchmark_group("audio/1 wav/resample=yes");
        group.bench_function("rodio", |b| b.iter_batched(
            || prepare_rodio_input(ALT_RATE, vec![&beep_data, &beep2_data], 1),
            |input| bench_play_rodio(input),
            criterion::BatchSize::SmallInput));
        // group.bench_function("rg3d", |b| b.iter_batched(
        //     || prepare_rg3d_input(ALT_RATE, vec![&beep_data, &beep2_data], 1),
        //     |input| bench_play_rg3d(input),
        //     criterion::BatchSize::SmallInput));
    }

    {
        let mut group = c.benchmark_group("audio/10 wav/resample=no");
        group.bench_function("rodio", |b| b.iter_batched(
            || prepare_rodio_input(SRC_RATE, vec![&beep_data, &beep2_data], 10),
            |input| bench_play_rodio(input),
            criterion::BatchSize::SmallInput));
        group.bench_function("rg3d", |b| b.iter_batched(
            || prepare_rg3d_input(SRC_RATE, vec![&beep_data, &beep2_data], 10),
            |input| bench_play_rg3d(input),
            criterion::BatchSize::SmallInput));
    }
    {
        let mut group = c.benchmark_group("audio/10 wav/resample=yes");
        group.bench_function("rodio", |b| b.iter_batched(
            || prepare_rodio_input(ALT_RATE, vec![&beep_data, &beep2_data], 10),
            |input| bench_play_rodio(input),
            criterion::BatchSize::SmallInput));
        // group.bench_function("rg3d", |b| b.iter_batched(
        //     || prepare_rg3d_input(ALT_RATE, vec![&beep_data, &beep2_data], 10),
        //     |input| bench_play_rg3d(input),
        //     criterion::BatchSize::SmallInput));
    }

    {
        let mut group = c.benchmark_group("audio/100 wav/resample=no");
        group.sample_size(10).bench_function("rodio", |b| b.iter_batched(
            || prepare_rodio_input(SRC_RATE, vec![&beep_data, &beep2_data], 100),
            |input| bench_play_rodio(input),
            criterion::BatchSize::SmallInput));
        group.sample_size(10).bench_function("rg3d", |b| b.iter_batched(
            || prepare_rg3d_input(SRC_RATE, vec![&beep_data, &beep2_data], 100),
            |input| bench_play_rg3d(input),
            criterion::BatchSize::SmallInput));
    }
    {
        let mut group = c.benchmark_group("audio/100 wav/resample=yes");
        group.sample_size(10).bench_function("rodio", |b| b.iter_batched(
            || prepare_rodio_input(ALT_RATE, vec![&beep_data, &beep2_data], 100),
            |input| bench_play_rodio(input),
            criterion::BatchSize::SmallInput));
        // group.sample_size(10).bench_function("rg3d", |b| b.iter_batched(
        //     || prepare_rg3d_input(ALT_RATE, vec![&beep_data, &beep2_data], 100),
        //     |input| bench_play_rg3d(input),
        //     criterion::BatchSize::SmallInput));
    }
}

fn prepare_rodio_input(rate: i32, src_data: Vec<&[u8]>, num_srcs: usize) -> BenchRodioInput {
    let (controller, mixer) = rodio::dynamic_mixer::mixer(NUM_CHANNELS as u16, rate as u32);
    let src_data_arc: Vec<Arc<[u8]>> = src_data.iter()
        .map(|&d| Arc::from(d))
        .collect();
    let out = vec![(0.0f32, 0.0f32); (MIX_INTERVAL * rate) as usize];
    BenchRodioInput {
        mixer: RodioMixer {
            controller,
            mixer,
        },
        src_data: src_data_arc,
        num_srcs,
        out,
    }
}

fn prepare_rg3d_input(rate: i32, src_data: Vec<&[u8]>, num_srcs: usize) -> BenchRg3dInput {
    let engine = rg3d_sound::engine::SoundEngine::without_device();
    let context = rg3d_sound::context::SoundContext::new();
    engine.lock().unwrap().add_context(context.clone());
    let buffers: Vec<_> = src_data.iter()
        .map(|d| {
            rg3d_sound::buffer::SoundBufferResource::new_generic(
                rg3d_sound::buffer::DataSource::from_memory(d.to_vec()))
                .unwrap()
        })
        .collect();

    let out = vec![(0.0f32, 0.0f32); (MIX_INTERVAL * rate) as usize];
    BenchRg3dInput {
        engine,
        context,
        buffers,
        num_srcs,
        out,
    }
}

criterion_group!(benches, audio_mixer_wav_benchmark);
criterion_main!(benches);
