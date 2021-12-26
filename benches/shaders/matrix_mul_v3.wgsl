// The file actually contains the liquid template, see https://shopify.github.io/liquid/
{% include 'matrix_mul_common.wgsl' %}

// v3 runs a thread per output f32 cell, but uses workgroup memory to preload parts of src_mat and src_vecs,
// see https://cnugteren.github.io/tutorial/pages/page4.html for details.

{% assign tile_size = "8" %}

// TODO: Replace with native WGSL multiplication as soon as WGSL gets constant evaluation.
var<workgroup> src_mat_v3_{{wg_x}}_{{wg_y}}: array<f32, {{ tile_size | times: wg_x }}>;
var<workgroup> src_vecs_v3_{{wg_x}}_{{wg_y}}: array<f32, {{ tile_size | times: wg_y }}>;

[[stage(compute), workgroup_size({{wg_x}}, {{wg_y}})]]
fn main_v3_tile{{tile_size}}(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
    [[builtin(local_invocation_id)]] local_id: vec3<u32>,
) {
    let row_idx = global_id.x;
    let vec_idx = global_id.y;
    let local_row_idx = local_id.x;
    let local_vec_idx = local_id.y;
    let K = uniforms.K;
    let L = uniforms.L;
    let M = uniforms.M;
    var ret = 0.0;
    for (var k = 0u; k < K; k = k + {{tile_size}}u) {
        // Fill workgroup tiles. Assume that the tile_size is <= min(wg_x, wg_y) to simplify the loading code.
        if (local_vec_idx < {{tile_size}}u) {
            if (k + local_vec_idx < K) {
                src_mat_v3_{{wg_x}}_{{wg_y}}[local_row_idx * {{tile_size}}u + local_vec_idx] =
                    src_mat.data[row_idx * K + k + local_vec_idx];
            } else {
                src_mat_v3_{{wg_x}}_{{wg_y}}[local_row_idx * {{tile_size}}u + local_vec_idx] = 0.0;
            }
        }
        if (local_row_idx < {{tile_size}}u) {
            if (k + local_row_idx < K) {
                src_vecs_v3_{{wg_x}}_{{wg_y}}[local_vec_idx * {{tile_size}}u + local_row_idx] =
                    src_vecs.data[vec_idx * K + k + local_row_idx];
            } else {
                src_vecs_v3_{{wg_x}}_{{wg_y}}[local_vec_idx * {{tile_size}}u + local_row_idx] = 0.0;
            }
        }

        workgroupBarrier();

        // Multiply the workgroup src mat row and src vec.
        for (var i = 0u; i < {{tile_size}}u; i = i + 1u) {
            ret = ret + src_mat_v3_{{wg_x}}_{{wg_y}}[local_row_idx * {{tile_size}}u + i]
                * src_vecs_v3_{{wg_x}}_{{wg_y}}[local_vec_idx * {{tile_size}}u + i];
        }

        workgroupBarrier();
    }
    if (row_idx < L && vec_idx < M) {
        dst_vecs.data[vec_idx * L + row_idx] = ret;
        return;
    }
    if (row_idx > L + {{wg_x}}u || vec_idx > M + {{wg_y}}u) {
        let unused = atomicAdd(&(dst_vecs.errors), 1);
    }
}
