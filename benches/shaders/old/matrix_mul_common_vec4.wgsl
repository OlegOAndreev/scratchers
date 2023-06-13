// The file actually contains the liquid template, see https://shopify.github.io/liquid/

type Vec4 = vec4<f32>;
type Mat4 = mat4x4<f32>;

struct Uniforms {
    K: u32;
    L: u32;
    M: u32;
};

struct Vec4Array {
    data: array<Vec4>;
};

struct Dst {
    // Right now we only register errors when we spawn too many workgroups.
    errors: atomic<i32>;
    [[align(16)]]
    data: array<Vec4>;
};

[[group(0), binding(0)]]
var<uniform> uniforms: Uniforms;
[[group(0), binding(1)]]
var<storage, read> src_mat: Vec4Array;
[[group(0), binding(2)]]
var<storage, read> src_vecs: Vec4Array;
[[group(0), binding(3)]]
var<storage, read_write> dst_vecs: Dst;
