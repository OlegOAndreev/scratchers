use std::fs;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use criterion::{Criterion, criterion_group, criterion_main};
use hound::SampleFormat;
use rodio::Source;

const NUM_CHANNELS: i32 = 2;
// All files in data/ are in 44.1kHz.
const SRC_RATE: i32 = 44100;
// Many modern audio outputs have a 48kHz rate.
const ALT_RATE: i32 = 48000;
// Mix for two seconds.
const MIX_INTERVAL: i32 = 2;

// Enable to write the output results to debug.wav (you generally want to run specific benches
// with this parameter set to true).
const DEBUG_WRITE_TO_FILE: bool = false;
// const DEBUG_WRITE_TO_FILE: bool = true;

// rodio.

// cpal::Device is not a trait, so we cannot use rodio::OutputStream and have to manually
// construct dynamic mixer with f32 type (see stream.rs in rodio)
struct RodioMixer {
    mixer: rodio::dynamic_mixer::DynamicMixer<f32>,
    controller: Arc<rodio::dynamic_mixer::DynamicMixerController<f32>>,
}

type RodioSource = dyn rodio::Source<Item=f32> + Send;

struct BenchRodioInput {
    mixer: RodioMixer,
    src_data: Vec<Arc<[u8]>>,
    num_srcs: usize,
    out: Vec<(f32, f32)>,
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

fn make_rodio_source(data: &Arc<[u8]>) -> Box<RodioSource> {
    // Copy from append() from rodio::SpatialSink and rodio::Sink. We ignore all the sync costs as
    // they should generally be negligible (only run at the start and the end of playing the sound).
    let decoder_src = rodio::Decoder::new(Cursor::new(data.clone())).unwrap()
        .pausable(false)
        .amplify(1.0)
        .stoppable()
        // Added because both rg3d-sound and soloud support changing the speed.
        .speed(1.0)
        .convert_samples::<f32>();
    Box::new(rodio::source::Spatial::new(
        decoder_src,
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ))
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

    criterion::black_box(&input.out);
    debug_write_to_file("rodio", input.mixer.mixer.sample_rate(), &input.out);
}

// rg3d-sound.

struct BenchRg3dInput {
    engine: Arc<Mutex<rg3d_sound::engine::SoundEngine>>,
    context: rg3d_sound::context::SoundContext,
    buffers: Vec<rg3d_sound::buffer::SoundBufferResource>,
    num_srcs: usize,
    out: Vec<(f32, f32)>,
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

fn bench_play_rg3d(mut input: BenchRg3dInput) {
    for i in 0..input.num_srcs {
        let src = rg3d_sound::source::generic::GenericSourceBuilder::new()
            .with_buffer(input.buffers[i % input.buffers.len()].clone())
            .with_status(rg3d_sound::source::Status::Playing)
            .build()
            .unwrap();
        let spatial_src = rg3d_sound::source::spatial::SpatialSourceBuilder::new(src)
            // .with_position([0.5, 0.0, 0.0].into())
            .build_source();
        input.context.state().add_source(spatial_src);
    }

    input.engine.lock().unwrap().render(&mut input.out);

    criterion::black_box(&input.out);
    debug_write_to_file("rg3d", rg3d_sound::context::SAMPLE_RATE, &input.out);
}

// Oddio.

struct BenchOddioInput {
    rate: i32,
    scene: oddio::SplitSignal<oddio::SpatialScene>,
    scene_handle: oddio::Handle<oddio::SpatialScene>,
    frames: Vec<Arc<oddio::Frames<f32>>>,
    num_srcs: usize,
    out: Vec<[f32; 2]>,
}

fn prepare_oddio_input(rate: i32, src_data: Vec<&[u8]>, num_srcs: usize) -> BenchOddioInput {
    let (scene_handle, scene) = oddio::split(oddio::SpatialScene::new(rate as u32, 0.1));
    let mut frames = vec![];
    for data in src_data {
        let (frames_data, sample_rate) = read_audio(data).unwrap();
        frames.push(oddio::Frames::from_slice(sample_rate, &frames_data));
    }

    let out = vec![[0.0f32, 0.0f32]; (MIX_INTERVAL * rate) as usize];
    BenchOddioInput {
        rate,
        scene,
        scene_handle,
        frames,
        num_srcs,
        out,
    }
}

fn bench_play_oddio(mut input: BenchOddioInput) {
    for i in 0..input.num_srcs {
        let frames_signal = oddio::FramesSignal::from(input.frames[i % input.frames.len()].clone());
        // Copy controls from rodio. Stop is already included, see SpatialSceneControl.
        let signal = oddio::Speed::new(oddio::Gain::new(frames_signal, 1.0));
        // let signal = frames_signal;
        input.scene_handle.control().play(signal, oddio::SpatialOptions {
            position: [0.0, 0.0, 0.0].into(),
            // position: [0.5, 0.0, 0.0].into(),
            velocity: [0.0, 0.0, 0.0].into(),
            ..Default::default()
        });
    }

    // oddio requires processing the output in small chunks. We used the 100ms buffer size when
    // creating SpatialScene, use 50ms here.
    let chunk_size = (input.rate as f32 * 0.05) as usize;
    for i in (0..input.out.len()).step_by(chunk_size) {
        let end = (i + chunk_size).min(input.out.len());
        oddio::run(&input.scene, input.rate as u32, &mut input.out[i..end]);
    }

    criterion::black_box(&input.out);
    debug_write_to_file_arr("oddio", input.rate as u32, &input.out);
}

// Generic sound loading

// Parses audio data and returns an array of frames and a sample rate.
fn read_audio(data: &[u8]) -> Result<(Vec<f32>, u32)> {
    if let Ok(reader) = hound::WavReader::new(Cursor::new(data)) {
        let spec = reader.spec();
        match spec.sample_format {
            SampleFormat::Float => {
                if spec.bits_per_sample != 32 {
                    bail!("Strange WAV file format: {}bit float", spec.bits_per_sample);
                }
                let samples: Vec<_> = reader.into_samples::<f32>()
                    .map(|s| s.unwrap())
                    .collect();
                return Ok((samples, spec.sample_rate));
            }
            SampleFormat::Int => {
                let mut buf = vec![0.0f32; reader.len() as usize];
                if spec.bits_per_sample == 16 {
                    let samples = reader.into_samples::<i16>().map(|s| s.unwrap());
                    for (dst, src) in buf.iter_mut().zip(samples) {
                        *dst = src as f32 / 32768.0f32;
                    }
                } else if spec.bits_per_sample == 32 {
                    let samples = reader.into_samples::<i32>().map(|s| s.unwrap());
                    for (dst, src) in buf.iter_mut().zip(samples) {
                        *dst = src as f32 / 2147483648.0f32;
                    }
                } else {
                    bail!("Strange WAV file format: {}bit int", spec.bits_per_sample);
                }
                return Ok((buf, spec.sample_rate));
            }
        }
    }
    bail!("Could not parse audio file")
}


// Debug utilities.

fn debug_write_to_file(prefix: &str, rate: u32, buf: &[(f32, f32)]) {
    if !DEBUG_WRITE_TO_FILE {
        return;
    }

    let wav_spec = hound::WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let file_name = format!("{}-debug.wav", prefix);
    let mut wav_writer = hound::WavWriter::create(file_name, wav_spec).unwrap();

    for &(l, r) in buf.iter() {
        wav_writer.write_sample(l).unwrap();
        wav_writer.write_sample(r).unwrap();
    }

    wav_writer.finalize().unwrap();
}

fn debug_write_to_file_arr(prefix: &str, rate: u32, buf: &[[f32; 2]]) {
    if !DEBUG_WRITE_TO_FILE {
        return;
    }

    let wav_spec = hound::WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let file_name = format!("{}-debug.wav", prefix);
    let mut wav_writer = hound::WavWriter::create(file_name, wav_spec).unwrap();

    for &s in buf.iter() {
        wav_writer.write_sample(s[0]).unwrap();
        wav_writer.write_sample(s[1]).unwrap();
    }

    wav_writer.finalize().unwrap();
}

// Benchmarks audio library mixers.
pub fn audio_mixer_wav_benchmark(c: &mut Criterion) {
    let beep_data: Vec<u8> = fs::read("benches/data/beep.wav").unwrap();
    let beep2_data: Vec<u8> = fs::read("benches/data/beep2.wav").unwrap();
    let beep48k_data: Vec<u8> = fs::read("benches/data/beep48k.wav").unwrap();

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

        group.bench_function("oddio", |b| b.iter_batched(
            || prepare_oddio_input(SRC_RATE, vec![], 0),
            |input| bench_play_oddio(input),
            criterion::BatchSize::SmallInput));
    }

    // Benchmark WAV file mixing (and built-in resampling).
    for num_srcs in [1, 2, 10, 100] {
        {
            let mut group = c.benchmark_group(format!("audio/{} wav 44.1k/out 44.1k", num_srcs));
            group.bench_function("rodio", |b| b.iter_batched(
                || prepare_rodio_input(SRC_RATE, vec![&beep_data, &beep2_data], num_srcs),
                |input| bench_play_rodio(input),
                criterion::BatchSize::SmallInput));

            group.bench_function("rg3d", |b| b.iter_batched(
                || prepare_rg3d_input(SRC_RATE, vec![&beep_data, &beep2_data], num_srcs),
                |input| bench_play_rg3d(input),
                criterion::BatchSize::SmallInput));

            group.bench_function("oddio", |b| b.iter_batched(
                || prepare_oddio_input(SRC_RATE, vec![&beep_data, &beep2_data], num_srcs),
                |input| bench_play_oddio(input),
                criterion::BatchSize::SmallInput));
        }
        {
            let mut group = c.benchmark_group(format!("audio/{} wav 48k/out 44.1k", num_srcs));
            group.bench_function("rodio", |b| b.iter_batched(
                || prepare_rodio_input(SRC_RATE, vec![&beep48k_data], num_srcs),
                |input| bench_play_rodio(input),
                criterion::BatchSize::SmallInput));

            // NOTE: The quality here is absolutely horrible.
            group.bench_function("rg3d", |b| b.iter_batched(
                || prepare_rg3d_input(SRC_RATE, vec![&beep48k_data], num_srcs),
                |input| bench_play_rg3d(input),
                criterion::BatchSize::SmallInput));

            group.bench_function("oddio", |b| b.iter_batched(
                || prepare_oddio_input(SRC_RATE, vec![&beep48k_data], num_srcs),
                |input| bench_play_oddio(input),
                criterion::BatchSize::SmallInput));
        }
        {
            let mut group = c.benchmark_group(format!("audio/{} wav 44.1k/out 48k", num_srcs));
            group.bench_function("rodio", |b| b.iter_batched(
                || prepare_rodio_input(ALT_RATE, vec![&beep_data, &beep2_data], num_srcs),
                |input| bench_play_rodio(input),
                criterion::BatchSize::SmallInput));

            // TODO: Add resampler to bench_play_rg3d.
            // group.bench_function("rg3d", |b| b.iter_batched(
            //     || prepare_rg3d_input(ALT_RATE, vec![&beep_data, &beep2_data], num_srcs),
            //     |input| bench_play_rg3d(input),
            //     criterion::BatchSize::SmallInput));

            group.bench_function("oddio", |b| b.iter_batched(
                || prepare_oddio_input(ALT_RATE, vec![&beep_data, &beep2_data], num_srcs),
                |input| bench_play_oddio(input),
                criterion::BatchSize::SmallInput));
        }
    }
}

criterion_group!(benches, audio_mixer_wav_benchmark);
criterion_main!(benches);
