// The file actually contains the liquid template, see https://shopify.github.io/liquid/
{% include 'matrix_mul_common_vec4.wgsl' %}

// v2_vec8_aligned is the version of v2_vec8 which has no special cases (requires all K to be multiple of 8,
// L to be multiple of 4).
[[stage(compute), workgroup_size({{wg_x}}, {{wg_y}})]]
fn main_v2_vec8_aligned(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>
) {
    let row_idx = global_id.x * 4u;
    let vec_idx = global_id.y;
    let K = uniforms.K / 4u;
    let L = uniforms.L;
    let M = uniforms.M;
    var ret0_0 = Vec4(0.0);
    var ret0_1 = Vec4(0.0);
    var ret1_0 = Vec4(0.0);
    var ret1_1 = Vec4(0.0);
    var ret2_0 = Vec4(0.0);
    var ret2_1 = Vec4(0.0);
    var ret3_0 = Vec4(0.0);
    var ret3_1 = Vec4(0.0);
    var i = 0u;
    for (; i < K - 1u; i = i + 2u) {
        let src_vec_0 = src_vecs.data[vec_idx * K + i];
        let src_vec_1 = src_vecs.data[vec_idx * K + i + 1u];
        ret0_0 = ret0_0 + src_mat.data[row_idx * K + i] * src_vec_0;
        ret0_1 = ret0_1 + src_mat.data[row_idx * K + i + 1u] * src_vec_1;
        ret1_0 = ret1_0 + src_mat.data[(row_idx + 1u) * K + i] * src_vec_0;
        ret1_1 = ret1_1 + src_mat.data[(row_idx + 1u) * K + i + 1u] * src_vec_1;
        ret2_0 = ret2_0 + src_mat.data[(row_idx + 2u) * K + i] * src_vec_0;
        ret2_1 = ret2_1 + src_mat.data[(row_idx + 2u) * K + i + 1u] * src_vec_1;
        ret3_0 = ret3_0 + src_mat.data[(row_idx + 3u) * K + i] * src_vec_0;
        ret3_1 = ret3_1 + src_mat.data[(row_idx + 3u) * K + i + 1u] * src_vec_1;
    }
    var ret0 = ret0_0 + ret0_1;
    var ret1 = ret1_0 + ret1_1;
    var ret2 = ret2_0 + ret2_1;
    var ret3 = ret3_0 + ret3_1;
    var ret_mat = transpose(Mat4(ret0, ret1, ret2, ret3));
    var ret = ret_mat[0] + ret_mat[1] + ret_mat[2] + ret_mat[3];
    dst_vecs.data[(vec_idx * L + row_idx) / 4u] = ret;
}
