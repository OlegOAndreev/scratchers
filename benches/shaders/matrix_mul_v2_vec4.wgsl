// The file actually contains the liquid template, see https://shopify.github.io/liquid/
{% include 'matrix_mul_common_vec4.wgsl' %}

// v2 is one thread for 4 rows, using vec4 to read/write and multiply rows.

[[stage(compute), workgroup_size({{wg_x}}, {{wg_y}})]]
fn main_v2_vec4(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>
) {
    let row_idx = global_id.x * 4u;
    let vec_idx = global_id.y;
    let K = uniforms.K / 4u;
    let L = uniforms.L;
    let M = uniforms.M;
    if (row_idx < L - 3u && vec_idx < M) {
        var ret0 = Vec4(0.0);
        var ret1 = Vec4(0.0);
        var ret2 = Vec4(0.0);
        var ret3 = Vec4(0.0);
        for (var i = 0u; i < K; i = i + 1u) {
            let src_vec = src_vecs.data[vec_idx * K + i];
            ret0 = ret0 + src_mat.data[row_idx * K + i] * src_vec;
            ret1 = ret1 + src_mat.data[(row_idx + 1u) * K + i] * src_vec;
            ret2 = ret2 + src_mat.data[(row_idx + 2u) * K + i] * src_vec;
            ret3 = ret3 + src_mat.data[(row_idx + 3u) * K + i] * src_vec;
        }
        var ret_mat = transpose(Mat4(ret0, ret1, ret2, ret3));
        var ret = ret_mat[0] + ret_mat[1] + ret_mat[2] + ret_mat[3];
        dst_vecs.data[(vec_idx * L + row_idx) / 4u] = ret;
        return;
    }
    if (row_idx < L - 2u && vec_idx < M) {
        var ret0 = Vec4(0.0);
        var ret1 = Vec4(0.0);
        var ret2 = Vec4(0.0);
        var ret3 = Vec4(0.0);
        for (var i = 0u; i < K; i = i + 1u) {
            let src_vec = src_vecs.data[vec_idx * K + i];
            ret0 = ret0 + src_mat.data[row_idx * K + i] * src_vec;
            ret1 = ret1 + src_mat.data[(row_idx + 1u) * K + i] * src_vec;
            ret2 = ret2 + src_mat.data[(row_idx + 2u) * K + i] * src_vec;
        }
        var ret_mat = transpose(Mat4(ret0, ret1, ret2, ret3));
        var ret = ret_mat[0] + ret_mat[1] + ret_mat[2] + ret_mat[3];
        dst_vecs.data[(vec_idx * L + row_idx) / 4u] = ret;
        return;
    }
    if (row_idx < L - 1u && vec_idx < M) {
        var ret0 = Vec4(0.0);
        var ret1 = Vec4(0.0);
        var ret2 = Vec4(0.0);
        var ret3 = Vec4(0.0);
        for (var i = 0u; i < K; i = i + 1u) {
            let src_vec = src_vecs.data[vec_idx * K + i];
            ret0 = ret0 + src_mat.data[row_idx * K + i] * src_vec;
            ret1 = ret1 + src_mat.data[(row_idx + 1u) * K + i] * src_vec;
        }
        var ret_mat = transpose(Mat4(ret0, ret1, ret2, ret3));
        var ret = ret_mat[0] + ret_mat[1] + ret_mat[2] + ret_mat[3];
        dst_vecs.data[(vec_idx * L + row_idx) / 4u] = ret;
        return;
    }
    if (row_idx < L && vec_idx < M) {
        var ret0 = Vec4(0.0);
        var ret1 = Vec4(0.0);
        var ret2 = Vec4(0.0);
        var ret3 = Vec4(0.0);
        for (var i = 0u; i < K; i = i + 1u) {
            ret0 = ret0 + src_mat.data[row_idx * K + i] * src_vecs.data[vec_idx * K + i];
        }
        var ret_mat = transpose(Mat4(ret0, ret1, ret2, ret3));
        var ret = ret_mat[0] + ret_mat[1] + ret_mat[2] + ret_mat[3];
        dst_vecs.data[(vec_idx * L + row_idx) / 4u] = ret;
        return;
    }
    if (row_idx > L + {{wg_x}}u * 4u || vec_idx > M + {{wg_y}}u) {
        let unused = atomicAdd(&(dst_vecs.errors), 1);
    }
}
