type Vec4 = vec4<f32>;

[[block]]
struct Uniforms {
    K: u32;
    L: u32;
};

[[block]]
struct F32Array {
    data: array<f32>;
};

[[group(0), binding(0)]]
var<uniform> uniforms: Uniforms;
[[group(0), binding(1)]]
var<storage, read> src_mat: F32Array;
[[group(0), binding(2)]]
var<storage, read> src_vecs: F32Array;
[[group(0), binding(3)]]
var<storage, read_write> dst_vecs: F32Array;

// The most basic version: run a thread per output f32 cell. src_vecs and dst_vecs contain one vector each.
[[stage(compute), workgroup_size(1, 1)]]
fn main_v1([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let row_idx = global_id.x;
    let vec_idx = global_id.y;
    let K = uniforms.K;
    let L = uniforms.L;
    var ret = 0.0;
    for (var i = 0u; i < K; i = i + 1u) {
        ret = ret + src_mat.data[row_idx * K + i] * src_vecs.data[vec_idx * K + i];
    }
    dst_vecs.data[vec_idx * L + row_idx] = ret;
}
