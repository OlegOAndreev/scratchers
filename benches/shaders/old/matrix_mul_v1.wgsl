// The file actually contains the liquid template, see https://shopify.github.io/liquid/
{% include 'matrix_mul_common.wgsl' %}

// v1 is the most basic version: run a thread per output f32 cell.
[[stage(compute), workgroup_size({{wg_x}}, {{wg_y}})]]
fn main_v1(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>
) {
    let row_idx = global_id.x;
    let vec_idx = global_id.y;
    let K = uniforms.K;
    let L = uniforms.L;
    let M = uniforms.M;
    if (row_idx < L && vec_idx < M) {
        var ret = 0.0;
        for (var i = 0u; i < K; i = i + 1u) {
            ret = ret + src_mat.data[row_idx * K + i] * src_vecs.data[vec_idx * K + i];
        }
        dst_vecs.data[vec_idx * L + row_idx] = ret;
        return;
    }
    if (row_idx > L + {{wg_x}}u || vec_idx > M + {{wg_y}}u) {
        let unused = atomicAdd(&(dst_vecs.errors), 1);
    }
}
