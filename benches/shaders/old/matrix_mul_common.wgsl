// The file actually contains the liquid template, see https://shopify.github.io/liquid/

struct Uniforms {
    K: u32,
    L: u32,
    M: u32,
};

struct F32Array {
    data: array<f32>,
};

struct Dst {
    // Right now we only register errors when we spawn too many workgroups.
    errors: atomic<i32>,
    @align(16)
    data: array<f32>,
};

[[group(0), binding(0)]]
var<uniform> uniforms: Uniforms;
[[group(0), binding(1)]]
var<storage, read> src_mat: F32Array;
[[group(0), binding(2)]]
var<storage, read> src_vecs: F32Array;
[[group(0), binding(3)]]
var<storage, read_write> dst_vecs: Dst;
