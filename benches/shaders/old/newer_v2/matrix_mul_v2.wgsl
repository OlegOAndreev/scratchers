// The file actually contains the liquid template, see https://shopify.github.io/liquid/
{% include 'matrix_mul_common.wgsl' %}

// v2 is version of v1 with one thread per 4 rows.

[[stage(compute), workgroup_size({{wg_x}}, {{wg_y}})]]
fn main_v2(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>
) {
    let row_idx = global_id.x * 4u;
    let vec_idx = global_id.y;
    let K = uniforms.K;
    let L = uniforms.L;
    let M = uniforms.M;
    if (row_idx < L - 3u && vec_idx < M) {
        var ret0 = 0.0;
        var ret1 = 0.0;
        var ret2 = 0.0;
        var ret3 = 0.0;
        for (var i = 0u; i < K; i = i + 1u) {
            let src_vec = src_vecs.data[vec_idx * K + i];
            ret0 = ret0 + src_mat.data[row_idx * K + i] * src_vec;
            ret1 = ret1 + src_mat.data[(row_idx + 1u) * K + i] * src_vec;
            ret2 = ret2 + src_mat.data[(row_idx + 2u) * K + i] * src_vec;
            ret3 = ret3 + src_mat.data[(row_idx + 3u) * K + i] * src_vec;
        }
        dst_vecs.data[vec_idx * L + row_idx] = ret0;
        dst_vecs.data[vec_idx * L + row_idx + 1u] = ret1;
        dst_vecs.data[vec_idx * L + row_idx + 2u] = ret2;
        dst_vecs.data[vec_idx * L + row_idx + 3u] = ret3;
        return;
    }
    if (row_idx < L - 2u && vec_idx < M) {
        var ret0 = 0.0;
        var ret1 = 0.0;
        var ret2 = 0.0;
        for (var i = 0u; i < K; i = i + 1u) {
            let src_vec = src_vecs.data[vec_idx * K + i];
            ret0 = ret0 + src_mat.data[row_idx * K + i] * src_vec;
            ret1 = ret1 + src_mat.data[(row_idx + 1u) * K + i] * src_vec;
            ret2 = ret2 + src_mat.data[(row_idx + 2u) * K + i] * src_vec;
        }
        dst_vecs.data[vec_idx * L + row_idx] = ret0;
        dst_vecs.data[vec_idx * L + row_idx + 1u] = ret1;
        dst_vecs.data[vec_idx * L + row_idx + 2u] = ret2;
        return;
    }
    if (row_idx < L - 1u && vec_idx < M) {
        var ret0 = 0.0;
        var ret1 = 0.0;
        for (var i = 0u; i < K; i = i + 1u) {
            let src_vec = src_vecs.data[vec_idx * K + i];
            ret0 = ret0 + src_mat.data[row_idx * K + i] * src_vec;
            ret1 = ret1 + src_mat.data[(row_idx + 1u) * K + i] * src_vec;
        }
        dst_vecs.data[vec_idx * L + row_idx] = ret0;
        dst_vecs.data[vec_idx * L + row_idx + 1u] = ret1;
        return;
    }
    if (row_idx < L && vec_idx < M) {
        var ret0 = 0.0;
        for (var i = 0u; i < K; i = i + 1u) {
            ret0 = ret0 + src_mat.data[row_idx * K + i] * src_vecs.data[vec_idx * K + i];
        }
        dst_vecs.data[vec_idx * L + row_idx] = ret0;
        return;
    }
    if (row_idx > L + {{wg_x}}u * 4u || vec_idx > M + {{wg_y}}u) {
        let unused = atomicAdd(&(dst_vecs.errors), 1);
    }
}

