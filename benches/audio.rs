use std::fs;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, Result};
use criterion::{Criterion, criterion_group, criterion_main};
use rodio::Source;
use rubato::Resampler;

const NUM_CHANNELS: i32 = 2;
// All files in data/ are in 44.1kHz.
const SRC_RATE: i32 = 44100;
// Many modern audio outputs have a 48kHz rate.
const ALT_RATE: i32 = 48000;
// Mix for two seconds.
const MIX_INTERVAL: i32 = 2;

// Buffer size for various resamplers etc.
const BUF_SIZE: usize = 2048;

// Enable to write the output results to debug.wav (you generally want to run specific benches
// with this parameter set to true).
static DEBUG_WRITE_TO_FILE: AtomicBool = AtomicBool::new(false);

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

struct BenchRg3dInput<R: rubato::Resampler<f32>> {
    rate: i32,
    engine: Arc<Mutex<rg3d_sound::engine::SoundEngine>>,
    context: rg3d_sound::context::SoundContext,
    buffers: Vec<rg3d_sound::buffer::SoundBufferResource>,
    num_srcs: usize,
    out_resampler: R,
    out: Vec<(f32, f32)>,
}

fn prepare_rg3d_input_fft(
    rate: i32,
    src_data: Vec<&[u8]>,
    num_srcs: usize,
    with_hrtf: bool,
) -> BenchRg3dInput<rubato::FftFixedOut<f32>> {
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
    if with_hrtf {
        let hrir_sphere = rg3d_sound::hrtf::HrirSphere::from_file(
            "benches/data/IRC_1002_C.bin",
            rg3d_sound::context::SAMPLE_RATE,
        ).unwrap();
        context.state().set_renderer(rg3d_sound::renderer::Renderer::HrtfRenderer(
            rg3d_sound::renderer::hrtf::HrtfRenderer::new(hrir_sphere)
        ));
    }

    let out = vec![(0.0f32, 0.0f32); (MIX_INTERVAL * rate) as usize];
    let out_resampler = rubato::FftFixedOut::<f32>::new(
        rg3d_sound::context::SAMPLE_RATE as usize,
        rate as usize,
        BUF_SIZE,
        1,
        2);
    BenchRg3dInput {
        rate,
        engine,
        context,
        buffers,
        num_srcs,
        out_resampler,
        out,
    }
}

// rg3d-sound requires the render() to be called with a specific buffer size
// (rg3d_sound::engine::SoundEngine::render_buffer_len()) while we need to produce chunks
// of different size. Rg3dHrtfReader allows rendering into buffers of any size.
struct Rg3dHrtfReader {
    chunk: Vec<(f32, f32)>,
    pos: usize,
}

impl Rg3dHrtfReader {
    fn new() -> Self {
        let chunk_size = rg3d_sound::engine::SoundEngine::render_buffer_len();
        Self {
            chunk: vec![(0.0, 0.0); chunk_size],
            pos: chunk_size,
        }
    }

    fn render(&mut self, engine: &mut rg3d_sound::engine::SoundEngine, buf: &mut [(f32, f32)]) {
        let mut buf_pos = 0;
        loop {
            let buf_remaining = buf.len() - buf_pos;
            let chunk_remaining = self.chunk.len() - self.pos;
            if buf_remaining <= chunk_remaining {
                buf[buf_pos..].copy_from_slice(&self.chunk[self.pos..self.pos + buf_remaining]);
                self.pos += buf_remaining;
                return;
            }
            buf[buf_pos..buf_pos + chunk_remaining].copy_from_slice(&self.chunk[self.pos..]);
            buf_pos += chunk_remaining;
            engine.render(&mut self.chunk);
            self.pos = 0;
        }
    }
}

fn bench_play_rg3d<R: rubato::Resampler<f32>>(mut input: BenchRg3dInput<R>, with_hrtf: bool) {
    for i in 0..input.num_srcs {
        let src = rg3d_sound::source::generic::GenericSourceBuilder::new()
            .with_buffer(input.buffers[i % input.buffers.len()].clone())
            .with_status(rg3d_sound::source::Status::Playing)
            .build()
            .unwrap();
        let spatial_src = rg3d_sound::source::spatial::SpatialSourceBuilder::new(src)
            .with_position([0.1, 0.0, 0.0].into())
            .build_source();
        input.context.state().add_source(spatial_src);
    }

    let mut buf = vec![(0.0f32, 0.0f32); BUF_SIZE * 2];
    let mut left_buf = vec![0.0f32; BUF_SIZE * 2];
    let mut right_buf = vec![0.0f32; BUF_SIZE * 2];
    let mut hrtf_reader = Rg3dHrtfReader::new();

    for i in (0..input.out.len()).step_by(BUF_SIZE) {
        if input.rate != rg3d_sound::context::SAMPLE_RATE as i32 {
            let end = (i + BUF_SIZE).min(input.out.len());
            let in_samples = input.out_resampler.nbr_frames_needed();
            if with_hrtf {
                hrtf_reader.render(&mut input.engine.lock().unwrap(), &mut buf[0..in_samples]);
            } else {
                input.engine.lock().unwrap().render(&mut buf[0..in_samples]);
            }
            split_stereo(&buf[0..in_samples], &mut left_buf, &mut right_buf);
            let resampled = input.out_resampler
                .process(&[&left_buf[0..in_samples], &right_buf[0..in_samples]])
                .unwrap();
            interleave_stereo(&resampled[0], &resampled[1], &mut input.out[i..end]);
        } else {
            let end = (i + BUF_SIZE).min(input.out.len());
            if with_hrtf {
                hrtf_reader.render(&mut input.engine.lock().unwrap(), &mut buf[0..BUF_SIZE]);
            } else {
                input.engine.lock().unwrap().render(&mut buf[0..BUF_SIZE]);
            }
            input.out[i..end].copy_from_slice(&buf[0..end - i]);
        }
    }

    let prefix = if with_hrtf {
        "rg3d-fft-hrtf"
    } else {
        "rg3d-fft"
    };
    criterion::black_box(&input.out);
    debug_write_to_file(prefix, input.rate as u32, &input.out);
}

fn bench_play_rg3d_moving<R: rubato::Resampler<f32>>(mut input: BenchRg3dInput<R>) {
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

    let mut buf = vec![(0.0f32, 0.0f32); BUF_SIZE * 2];
    let mut left_buf = vec![0.0f32; BUF_SIZE * 2];
    let mut right_buf = vec![0.0f32; BUF_SIZE * 2];
    for i in (0..input.out.len()).step_by(BUF_SIZE) {
        // Move listener a bit.
        let listener_x = (i as f32 * 0.1).sin();
        let listener_y = (i as f32 * 0.1).cos();
        input.context.state().listener_mut().set_position([listener_x, listener_y, 0.0].into());

        if input.rate != rg3d_sound::context::SAMPLE_RATE as i32 {
            let end = (i + BUF_SIZE).min(input.out.len());
            let in_samples = input.out_resampler.nbr_frames_needed();
            input.engine.lock().unwrap().render(&mut buf[0..in_samples]);
            split_stereo(&buf[0..in_samples], &mut left_buf, &mut right_buf);
            let resampled = input.out_resampler
                .process(&[&left_buf[0..in_samples], &right_buf[0..in_samples]])
                .unwrap();
            interleave_stereo(&resampled[0], &resampled[1], &mut input.out[i..end]);
        } else {
            let end = (i + BUF_SIZE).min(input.out.len());
            input.engine.lock().unwrap().render(&mut buf[0..BUF_SIZE]);
            input.out[i..end].copy_from_slice(&buf[0..end - i]);
        }
    }

    criterion::black_box(&input.out);
    debug_write_to_file("rg3d-fft-moving", input.rate as u32, &input.out);
}

fn split_stereo(stereo: &[(f32, f32)], left: &mut [f32], right: &mut [f32]) {
    for i in 0..stereo.len() {
        let (l, r) = stereo[i];
        left[i] = l;
        right[i] = r;
    }
}

fn interleave_stereo(left: &[f32], right: &[f32], stereo: &mut [(f32, f32)]) {
    for i in 0..stereo.len() {
        stereo[i] = (left[i], right[i]);
    }
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

    let mut buf = [[0.0f32, 0.0f32]; BUF_SIZE];
    for i in (0..input.out.len()).step_by(BUF_SIZE) {
        let end = (i + BUF_SIZE).min(input.out.len());
        let buf_len = end - i;
        oddio::run(&input.scene, input.rate as u32, &mut buf[0..buf_len]);
        input.out[i..end].copy_from_slice(&buf[0..buf_len]);
    }

    criterion::black_box(&input.out);
    debug_write_to_file_arr("oddio", input.rate as u32, &input.out);
}

// Generic sound loading

// Parses audio data and returns an array of frames and a sample rate.
fn read_audio(data: &[u8]) -> Result<(Vec<f32>, u32)> {
    if let Ok(reader) = hound::WavReader::new(Cursor::new(data)) {
        let spec = reader.spec();
        return match spec.sample_format {
            hound::SampleFormat::Float => {
                if spec.bits_per_sample != 32 {
                    bail!("Strange WAV file format: {}bit float", spec.bits_per_sample);
                }
                let samples: Vec<_> = reader.into_samples::<f32>()
                    .map(|s| s.unwrap())
                    .collect();
                Ok((samples, spec.sample_rate))
            }
            hound::SampleFormat::Int => {
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
                Ok((buf, spec.sample_rate))
            }
        };
    }
    bail!("Could not parse audio file")
}


// Debug utilities.

fn debug_write_to_file(prefix: &str, rate: u32, buf: &[(f32, f32)]) {
    if !DEBUG_WRITE_TO_FILE.load(Ordering::SeqCst) {
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
    if !DEBUG_WRITE_TO_FILE.load(Ordering::SeqCst) {
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

// Benchmarks audio mixers.
pub fn audio_mixer_wav_benchmark(c: &mut Criterion) {
    if std::env::var("DEBUG_WRITE_TO_FILE").is_ok() {
        DEBUG_WRITE_TO_FILE.store(true, Ordering::SeqCst);
    }

    let beep_data: Vec<u8> = fs::read("benches/data/beep.wav").unwrap();
    let beep2_data: Vec<u8> = fs::read("benches/data/beep2.wav").unwrap();
    let beep48k_data: Vec<u8> = fs::read("benches/data/beep48k.wav").unwrap();

    {
        let mut group = c.benchmark_group("audio/empty");
        group.bench_function("rodio", |b| b.iter_batched(
            || prepare_rodio_input(SRC_RATE, vec![], 0),
            |input| bench_play_rodio(input),
            criterion::BatchSize::SmallInput));

        group.bench_function("rg3d-fft", |b| b.iter_batched(
            || prepare_rg3d_input_fft(SRC_RATE, vec![], 0, false),
            |input| bench_play_rg3d(input, false),
            criterion::BatchSize::SmallInput));

        group.bench_function("rg3d-fft-moving", |b| b.iter_batched(
            || prepare_rg3d_input_fft(SRC_RATE, vec![], 0, false),
            |input| bench_play_rg3d_moving(input),
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

            group.bench_function("rg3d-fft", |b| b.iter_batched(
                || prepare_rg3d_input_fft(SRC_RATE, vec![&beep_data, &beep2_data], num_srcs, false),
                |input| bench_play_rg3d(input, false),
                criterion::BatchSize::SmallInput));

            group.bench_function("rg3d-fft-hrtf", |b| b.iter_batched(
                || prepare_rg3d_input_fft(SRC_RATE, vec![&beep_data, &beep2_data], num_srcs, true),
                |input| bench_play_rg3d(input, true),
                criterion::BatchSize::SmallInput));

            group.bench_function("rg3d-fft-moving", |b| b.iter_batched(
                || prepare_rg3d_input_fft(SRC_RATE, vec![&beep_data, &beep2_data], num_srcs, false),
                |input| bench_play_rg3d_moving(input),
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
            group.bench_function("rg3d-fft", |b| b.iter_batched(
                || prepare_rg3d_input_fft(SRC_RATE, vec![&beep48k_data], num_srcs, false),
                |input| bench_play_rg3d(input, false),
                criterion::BatchSize::SmallInput));

            group.bench_function("rg3d-fft-hrtf", |b| b.iter_batched(
                || prepare_rg3d_input_fft(SRC_RATE, vec![&beep_data, &beep2_data], num_srcs, true),
                |input| bench_play_rg3d(input, true),
                criterion::BatchSize::SmallInput));

            group.bench_function("rg3d-fft-moving", |b| b.iter_batched(
                || prepare_rg3d_input_fft(SRC_RATE, vec![&beep48k_data], num_srcs, false),
                |input| bench_play_rg3d_moving(input),
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

            group.bench_function("rg3d-fft", |b| b.iter_batched(
                || prepare_rg3d_input_fft(ALT_RATE, vec![&beep_data, &beep2_data], num_srcs, false),
                |input| bench_play_rg3d(input, false),
                criterion::BatchSize::SmallInput));

            group.bench_function("rg3d-fft-hrtf", |b| b.iter_batched(
                || prepare_rg3d_input_fft(ALT_RATE, vec![&beep_data, &beep2_data], num_srcs, true),
                |input| bench_play_rg3d(input, true),
                criterion::BatchSize::SmallInput));

            group.bench_function("rg3d-fft-moving", |b| b.iter_batched(
                || prepare_rg3d_input_fft(ALT_RATE, vec![&beep_data, &beep2_data], num_srcs, false),
                |input| bench_play_rg3d_moving(input),
                criterion::BatchSize::SmallInput));

            group.bench_function("oddio", |b| b.iter_batched(
                || prepare_oddio_input(ALT_RATE, vec![&beep_data, &beep2_data], num_srcs),
                |input| bench_play_oddio(input),
                criterion::BatchSize::SmallInput));
        }
    }
}

// Rubato FFT

fn prepare_rubato_fft(in_rate: u32, out_rate: u32) -> rubato::FftFixedOut<f32> {
    rubato::FftFixedOut::<f32>::new(
        in_rate as usize,
        out_rate as usize,
        BUF_SIZE,
        1,
        2)
}

fn bench_resample_rubato_fft(
    mut resampler: rubato::FftFixedOut<f32>,
    in_frames: &[(f32, f32)],
    out_rate: u32,
    out_frames: &mut [(f32, f32)],
) {
    let mut in_pos = 0;
    let mut left = [0.0f32; BUF_SIZE * 2];
    let mut right = [0.0f32; BUF_SIZE * 2];
    for i in (0..out_frames.len()).step_by(BUF_SIZE) {
        let end = (i + BUF_SIZE).min(out_frames.len());
        let in_samples = resampler.nbr_frames_needed();
        split_stereo(&in_frames[in_pos..in_pos + in_samples],
                     &mut left[0..in_samples], &mut right[0..in_samples]);
        in_pos += in_samples;
        let resampled = resampler.process(&[&left[0..in_samples], &right[0..in_samples]]).unwrap();
        interleave_stereo(&resampled[0], &resampled[1], &mut out_frames[i..end]);
    }

    criterion::black_box(&out_frames);
    debug_write_to_file("rubato-fft", out_rate, out_frames);
}


// Rubato sinc

fn prepare_rubato_sinc(in_rate: u32, out_rate: u32) -> rubato::SincFixedOut<f32> {
    let resample_ratio = out_rate as f64 / in_rate as f64;
    rubato::SincFixedOut::<f32>::new(
        resample_ratio,
        // Copy-pasted from rubato/examples
        rubato::InterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.925914648491266,
            oversampling_factor: 320,
            interpolation: rubato::InterpolationType::Linear,
            window: rubato::WindowFunction::Blackman2,
        },
        BUF_SIZE,
        2)
}

fn bench_resample_rubato_sinc(
    mut resampler: rubato::SincFixedOut<f32>,
    in_frames: &[(f32, f32)],
    out_rate: u32,
    out_frames: &mut [(f32, f32)],
) {
    let mut in_pos = 0;
    let mut left = [0.0f32; BUF_SIZE * 2];
    let mut right = [0.0f32; BUF_SIZE * 2];
    for i in (0..out_frames.len()).step_by(BUF_SIZE) {
        let end = (i + BUF_SIZE).min(out_frames.len());
        let in_samples = resampler.nbr_frames_needed();
        split_stereo(&in_frames[in_pos..in_pos + in_samples],
                     &mut left[0..in_samples], &mut right[0..in_samples]);
        in_pos += in_samples;
        let resampled = resampler.process(&[&left[0..in_samples], &right[0..in_samples]]).unwrap();
        interleave_stereo(&resampled[0], &resampled[1], &mut out_frames[i..end]);
    }

    criterion::black_box(&out_frames);
    debug_write_to_file("rubato-sinc", out_rate, out_frames);
}

// Speexdsp-resampler

fn prepare_resample_speexdsp(
    in_rate: u32,
    out_rate: u32,
    quality: usize,
) -> speexdsp_resampler::State {
    speexdsp_resampler::State::new(
        2,
        in_rate as usize,
        out_rate as usize,
        quality,
    ).unwrap()
}

fn bench_resample_speexdsp(
    mut resampler: speexdsp_resampler::State,
    in_frames: &[(f32, f32)],
    out_rate: u32,
    out_frames: &mut [(f32, f32)],
) {
    let in_flat = flatten_stereo(in_frames);
    let mut in_pos = 0;
    let out_len = out_frames.len() * 2;
    let out_flat = flatten_stereo_mut(out_frames);
    for i in (0..out_len).step_by(BUF_SIZE * 2) {
        let end = (i + BUF_SIZE * 2).min(out_len);
        let (in_processed, out_processed) = resampler
            .process_float(0, &in_flat[in_pos..], &mut out_flat[i..end])
            .unwrap();
        assert_eq!(out_processed, end - i);
        in_pos += in_processed;
    }

    criterion::black_box(&out_frames);
    let file_prefix = format!("speexdsp-{}", resampler.get_quality());
    debug_write_to_file(&file_prefix, out_rate, out_frames);
}

// Libsamplerate

fn prepare_resample_samplerate(
    in_rate: u32,
    out_rate: u32,
    quality: samplerate::ConverterType,
) -> samplerate::Samplerate {
    samplerate::Samplerate::new(
        quality,
        in_rate,
        out_rate,
        2,
    ).unwrap()
}

fn bench_resample_samplerate(
    resampler: samplerate::Samplerate,
    in_frames: &[(f32, f32)],
    out_rate: u32,
    out_frames: &mut [(f32, f32)],
    quality: samplerate::ConverterType,
) {
    let in_flat = flatten_stereo(in_frames);
    let in_chunk_size = (BUF_SIZE as u64 * resampler.from_rate() as u64
        / resampler.to_rate() as u64) as usize * 2 + 1;
    let out_len = out_frames.len() * 2;
    let out_flat = flatten_stereo_mut(out_frames);
    let mut out_pos = 0;
    for i in (0..).step_by(in_chunk_size) {
        let out_chunk = resampler.process(&in_flat[i..i + in_chunk_size]).unwrap();
        let end = (out_pos + out_chunk.len()).min(out_len);
        out_flat[out_pos..end].copy_from_slice(&out_chunk[0..end - out_pos]);
        out_pos = end;
        if out_pos >= out_len {
            break;
        }
    }

    criterion::black_box(&out_frames);
    let file_prefix = format!("samplerate-{:?}", quality);
    debug_write_to_file(&file_prefix, out_rate, out_frames);
}

// SDL2 resampler

fn prepare_resample_sdl2(in_rate: u32, out_rate: u32) -> sdl2::audio::AudioCVT {
    sdl2::audio::AudioCVT::new(
        sdl2::audio::AudioFormat::F32LSB,
        2,
        in_rate as i32,
        sdl2::audio::AudioFormat::F32LSB,
        2,
        out_rate as i32,
    ).unwrap()
}

fn bench_resample_sdl2(
    resampler: sdl2::audio::AudioCVT,
    in_rate: u32,
    in_frames: &[(f32, f32)],
    out_rate: u32,
    out_frames: &mut [(f32, f32)],
) {
    let in_chunk_size = (BUF_SIZE as u64 * in_rate as u64 / out_rate as u64) as usize + 1;
    let out_len = out_frames.len();
    let mut out_pos = 0;
    for i in (0..).step_by(in_chunk_size) {
        let in_chunk = stereo_to_u8(&in_frames[i..i + in_chunk_size]);
        let out_chunk = resampler.convert(Vec::from(in_chunk));
        let out_chunk_slice = stereo_from_u8(&out_chunk);
        let end = (out_pos + out_chunk_slice.len()).min(out_len);
        out_frames[out_pos..end].copy_from_slice(&out_chunk_slice[0..end - out_pos]);
        out_pos = end;
        if out_pos >= out_len {
            break;
        }
    }

    criterion::black_box(&out_frames);
    debug_write_to_file("sdl2-cvt", out_rate, out_frames);
}

// Benchmarks audio resamplers.
pub fn audio_resampler_benchmark(c: &mut Criterion) {
    if std::env::var("DEBUG_WRITE_TO_FILE").is_ok() {
        DEBUG_WRITE_TO_FILE.store(true, Ordering::SeqCst);
    }

    {
        let beep_data: Vec<u8> = fs::read("benches/data/beep.wav").unwrap();
        // Let's test every resampler on stereo data as it is probably the .
        let (beep_frames_mono, in_rate) = read_audio(&beep_data).unwrap();
        let beep_frames: Vec<_> = beep_frames_mono.into_iter()
            .map(|v| (v, v))
            .collect();
        assert_eq!(in_rate, 44100);

        let out_rate = 48000u32;
        let mut out = vec![(0.0f32, 0.0f32); out_rate as usize * MIX_INTERVAL as usize];

        let mut group = c.benchmark_group("resample/in 44.1k/out 48k");
        group.bench_function("rubato-fft", |b| b.iter_batched(
            || prepare_rubato_fft(in_rate, out_rate),
            |input| bench_resample_rubato_fft(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("rubato-sinc", |b| b.iter_batched(
            || prepare_rubato_sinc(in_rate, out_rate),
            |input| bench_resample_rubato_sinc(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("speexdsp-0", |b| b.iter_batched(
            || prepare_resample_speexdsp(in_rate, out_rate, 0),
            |input| bench_resample_speexdsp(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("speexdsp-5", |b| b.iter_batched(
            || prepare_resample_speexdsp(in_rate, out_rate, 5),
            |input| bench_resample_speexdsp(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("speexdsp-10", |b| b.iter_batched(
            || prepare_resample_speexdsp(in_rate, out_rate, 10),
            |input| bench_resample_speexdsp(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("samplerate-linear", |b| b.iter_batched(
            || prepare_resample_samplerate(in_rate, out_rate, samplerate::ConverterType::Linear),
            |input| bench_resample_samplerate(input, &beep_frames, out_rate, &mut out,
                                              samplerate::ConverterType::Linear),
            criterion::BatchSize::SmallInput));
        group.bench_function("samplerate-sinc-fastest", |b| b.iter_batched(
            || prepare_resample_samplerate(in_rate, out_rate,
                                           samplerate::ConverterType::SincFastest),
            |input| bench_resample_samplerate(input, &beep_frames, out_rate, &mut out,
                                              samplerate::ConverterType::SincFastest),
            criterion::BatchSize::SmallInput));
        group.bench_function("samplerate-sinc-medium", |b| b.iter_batched(
            || prepare_resample_samplerate(in_rate, out_rate,
                                           samplerate::ConverterType::SincMediumQuality),
            |input| bench_resample_samplerate(input, &beep_frames, out_rate, &mut out,
                                              samplerate::ConverterType::SincMediumQuality),
            criterion::BatchSize::SmallInput));
        group.bench_function("samplerate-sinc-best", |b| b.iter_batched(
            || prepare_resample_samplerate(in_rate, out_rate,
                                           samplerate::ConverterType::SincBestQuality),
            |input| bench_resample_samplerate(input, &beep_frames, out_rate, &mut out,
                                              samplerate::ConverterType::SincBestQuality),
            criterion::BatchSize::SmallInput));
        group.bench_function("sdl2", |b| b.iter_batched(
            || prepare_resample_sdl2(in_rate, out_rate),
            |input| bench_resample_sdl2(input, in_rate, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
    }

    {
        let beep_data: Vec<u8> = fs::read("benches/data/beep48k.wav").unwrap();
        // Let's test every resampler on stereo data as it is probably the .
        let (beep_frames_mono, in_rate) = read_audio(&beep_data).unwrap();
        let beep_frames: Vec<_> = beep_frames_mono.into_iter()
            .map(|v| (v, v))
            .collect();
        assert_eq!(in_rate, 48000);

        let out_rate = 44100u32;
        let mut out = vec![(0.0f32, 0.0f32); out_rate as usize * MIX_INTERVAL as usize];

        let mut group = c.benchmark_group("resample/in 48k/out 44.1k");
        group.bench_function("rubato-fft", |b| b.iter_batched(
            || prepare_rubato_fft(in_rate, out_rate),
            |input| bench_resample_rubato_fft(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("rubato-sinc", |b| b.iter_batched(
            || prepare_rubato_sinc(in_rate, out_rate),
            |input| bench_resample_rubato_sinc(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("speexdsp-0", |b| b.iter_batched(
            || prepare_resample_speexdsp(in_rate, out_rate, 0),
            |input| bench_resample_speexdsp(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("speexdsp-5", |b| b.iter_batched(
            || prepare_resample_speexdsp(in_rate, out_rate, 5),
            |input| bench_resample_speexdsp(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("speexdsp-10", |b| b.iter_batched(
            || prepare_resample_speexdsp(in_rate, out_rate, 10),
            |input| bench_resample_speexdsp(input, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
        group.bench_function("samplerate-linear", |b| b.iter_batched(
            || prepare_resample_samplerate(in_rate, out_rate, samplerate::ConverterType::Linear),
            |input| bench_resample_samplerate(input, &beep_frames, out_rate, &mut out,
                                              samplerate::ConverterType::Linear),
            criterion::BatchSize::SmallInput));
        group.bench_function("samplerate-sinc-fastest", |b| b.iter_batched(
            || prepare_resample_samplerate(in_rate, out_rate,
                                           samplerate::ConverterType::SincFastest),
            |input| bench_resample_samplerate(input, &beep_frames, out_rate, &mut out,
                                              samplerate::ConverterType::SincFastest),
            criterion::BatchSize::SmallInput));
        group.bench_function("samplerate-sinc-medium", |b| b.iter_batched(
            || prepare_resample_samplerate(in_rate, out_rate,
                                           samplerate::ConverterType::SincMediumQuality),
            |input| bench_resample_samplerate(input, &beep_frames, out_rate, &mut out,
                                              samplerate::ConverterType::SincMediumQuality),
            criterion::BatchSize::SmallInput));
        group.bench_function("samplerate-sinc-best", |b| b.iter_batched(
            || prepare_resample_samplerate(in_rate, out_rate,
                                           samplerate::ConverterType::SincBestQuality),
            |input| bench_resample_samplerate(input, &beep_frames, out_rate, &mut out,
                                              samplerate::ConverterType::SincBestQuality),
            criterion::BatchSize::SmallInput));
        group.bench_function("sdl2", |b| b.iter_batched(
            || prepare_resample_sdl2(in_rate, out_rate),
            |input| bench_resample_sdl2(input, in_rate, &beep_frames, out_rate, &mut out),
            criterion::BatchSize::SmallInput));
    }
}

pub fn flatten_stereo(xs: &[(f32, f32)]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(xs.as_ptr() as _, xs.len() * 2) }
}

pub fn flatten_stereo_mut(xs: &mut [(f32, f32)]) -> &mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(xs.as_mut_ptr() as _, xs.len() * 2) }
}

pub fn stereo_to_u8(xs: &[(f32, f32)]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(xs.as_ptr() as _, xs.len() * 8) }
}

pub fn stereo_from_u8(xs: &[u8]) -> &[(f32, f32)] {
    unsafe { std::slice::from_raw_parts(xs.as_ptr() as _, xs.len() / 8) }
}

criterion_group!(mixer_benches, audio_mixer_wav_benchmark);
criterion_group!(resampler_benches, audio_resampler_benchmark);
criterion_main!(mixer_benches, resampler_benches);
