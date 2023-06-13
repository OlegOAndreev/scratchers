// The file actually contains the liquid template, see https://shopify.github.io/liquid/
{% include 'matrix_mul_common.wgsl' %}

// v2_loop8_unrolled_single_accum is a version of v2_loop8_unrolled which uses single accumulator instead of 8.

[[stage(compute), workgroup_size({{wg_x}}, {{wg_y}})]]
fn main_v2_loop8_unrolled_single_accum(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>
) {
    let row_idx = global_id.x * 4u;
    let vec_idx = global_id.y;
    let K = uniforms.K;
    let L = uniforms.L;
    let M = uniforms.M;
    if (row_idx < L - 3u && vec_idx < M) {
        var src_vec: array<f32, 8>;
        var ret: array<f32, 4>;
        // TODO: This should not be needed, arrays should be automatically initialized to zero. See if this is
        // a bug in naga -> MSL translation.
{% if zero_out %}
        ret[0u] = 0.0;
        ret[1u] = 0.0;
        ret[2u] = 0.0;
        ret[3u] = 0.0;
{% endif %}
        var i = 0u;
        for (; i < K - 7u; i = i + 8u) {
            src_vec[0] = src_vecs.data[vec_idx * K + i];
            src_vec[1] = src_vecs.data[vec_idx * K + i + 1u];
            src_vec[2] = src_vecs.data[vec_idx * K + i + 2u];
            src_vec[3] = src_vecs.data[vec_idx * K + i + 3u];
            src_vec[4] = src_vecs.data[vec_idx * K + i + 4u];
            src_vec[5] = src_vecs.data[vec_idx * K + i + 5u];
            src_vec[6] = src_vecs.data[vec_idx * K + i + 6u];
            src_vec[7] = src_vecs.data[vec_idx * K + i + 7u];
            for (var k = 0u; k < 4u; k = k + 1u) {
                ret[k] = ret[k] + src_mat.data[(row_idx + k) * K + i] * src_vec[0]
                    + src_mat.data[(row_idx + k) * K + i + 1u] * src_vec[1]
                    + src_mat.data[(row_idx + k) * K + i + 2u] * src_vec[2]
                    + src_mat.data[(row_idx + k) * K + i + 3u] * src_vec[3]
                    + src_mat.data[(row_idx + k) * K + i + 4u] * src_vec[4]
                    + src_mat.data[(row_idx + k) * K + i + 5u] * src_vec[5]
                    + src_mat.data[(row_idx + k) * K + i + 6u] * src_vec[6]
                    + src_mat.data[(row_idx + k) * K + i + 7u] * src_vec[7];
            }
        }
        for (var j = 0u; j < K - i; j = j + 1u) {
            let src_vec_e = src_vecs.data[vec_idx * K + i + j];
            for (var k = 0u; k < 4u; k = k + 1u) {
                ret[k] = ret[k] + src_mat.data[(row_idx + k) * K + i + j] * src_vec_e;
            }
        }
        for (var k = 0u; k < 4u; k = k + 1u) {
            dst_vecs.data[vec_idx * L + row_idx + k] = ret[k];
        }
        return;
    }
    if (row_idx < L - 2u && vec_idx < M) {
        var src_vec: array<f32, 8>;
        var ret: array<f32, 3>;
        // TODO: This should not be needed, arrays should be automatically initialized to zero. See if this is
        // a bug in naga -> MSL translation.
{% if zero_out %}
        ret[0u] = 0.0;
        ret[1u] = 0.0;
        ret[2u] = 0.0;
{% endif %}
        var i = 0u;
        for (; i < K - 7u; i = i + 8u) {
            src_vec[0] = src_vecs.data[vec_idx * K + i];
            src_vec[1] = src_vecs.data[vec_idx * K + i + 1u];
            src_vec[2] = src_vecs.data[vec_idx * K + i + 2u];
            src_vec[3] = src_vecs.data[vec_idx * K + i + 3u];
            src_vec[4] = src_vecs.data[vec_idx * K + i + 4u];
            src_vec[5] = src_vecs.data[vec_idx * K + i + 5u];
            src_vec[6] = src_vecs.data[vec_idx * K + i + 6u];
            src_vec[7] = src_vecs.data[vec_idx * K + i + 7u];
            for (var k = 0u; k < 3u; k = k + 1u) {
                ret[k] = ret[k] + src_mat.data[(row_idx + k) * K + i] * src_vec[0]
                    + src_mat.data[(row_idx + k) * K + i + 1u] * src_vec[1]
                    + src_mat.data[(row_idx + k) * K + i + 2u] * src_vec[2]
                    + src_mat.data[(row_idx + k) * K + i + 3u] * src_vec[3]
                    + src_mat.data[(row_idx + k) * K + i + 4u] * src_vec[4]
                    + src_mat.data[(row_idx + k) * K + i + 5u] * src_vec[5]
                    + src_mat.data[(row_idx + k) * K + i + 6u] * src_vec[6]
                    + src_mat.data[(row_idx + k) * K + i + 7u] * src_vec[7];
            }
        }
        for (var j = 0u; j < K - i; j = j + 1u) {
            let src_vec_e = src_vecs.data[vec_idx * K + i + j];
            for (var k = 0u; k < 3u; k = k + 1u) {
                ret[k] = ret[k] + src_mat.data[(row_idx + k) * K + i + j] * src_vec_e;
            }
        }
        for (var k = 0u; k < 4u; k = k + 1u) {
            dst_vecs.data[vec_idx * L + row_idx + k] = ret[k];
        }
        return;
    }
    if (row_idx < L - 1u && vec_idx < M) {
        var src_vec: array<f32, 8>;
        var ret: array<f32, 2>;
        // TODO: This should not be needed, arrays should be automatically initialized to zero. See if this is
        // a bug in naga -> MSL translation.
{% if zero_out %}
        ret[0u] = 0.0;
        ret[1u] = 0.0;
{% endif %}
        var i = 0u;
        for (; i < K - 7u; i = i + 8u) {
            src_vec[0] = src_vecs.data[vec_idx * K + i];
            src_vec[1] = src_vecs.data[vec_idx * K + i + 1u];
            src_vec[2] = src_vecs.data[vec_idx * K + i + 2u];
            src_vec[3] = src_vecs.data[vec_idx * K + i + 3u];
            src_vec[4] = src_vecs.data[vec_idx * K + i + 4u];
            src_vec[5] = src_vecs.data[vec_idx * K + i + 5u];
            src_vec[6] = src_vecs.data[vec_idx * K + i + 6u];
            src_vec[7] = src_vecs.data[vec_idx * K + i + 7u];
            for (var k = 0u; k < 2u; k = k + 1u) {
                ret[k] = ret[k] + src_mat.data[(row_idx + k) * K + i] * src_vec[0]
                    + src_mat.data[(row_idx + k) * K + i + 1u] * src_vec[1]
                    + src_mat.data[(row_idx + k) * K + i + 2u] * src_vec[2]
                    + src_mat.data[(row_idx + k) * K + i + 3u] * src_vec[3]
                    + src_mat.data[(row_idx + k) * K + i + 4u] * src_vec[4]
                    + src_mat.data[(row_idx + k) * K + i + 5u] * src_vec[5]
                    + src_mat.data[(row_idx + k) * K + i + 6u] * src_vec[6]
                    + src_mat.data[(row_idx + k) * K + i + 7u] * src_vec[7];
            }
        }
        for (var j = 0u; j < K - i; j = j + 1u) {
            let src_vec_e = src_vecs.data[vec_idx * K + i + j];
            for (var k = 0u; k < 2u; k = k + 1u) {
                ret[k] = ret[k] + src_mat.data[(row_idx + k) * K + i + j] * src_vec_e;
            }
        }
        for (var k = 0u; k < 2u; k = k + 1u) {
            dst_vecs.data[vec_idx * L + row_idx + k] = ret[k];
        }
        return;
    }
    if (row_idx < L && vec_idx < M) {
        var src_vec: array<f32, 8>;
        var ret: array<f32, 1>;
        // TODO: This should not be needed, arrays should be automatically initialized to zero. See if this is
        // a bug in naga -> MSL translation.
{% if zero_out %}
        ret[0u] = 0.0;
{% endif %}
        var i = 0u;
        for (; i < K - 7u; i = i + 8u) {
            src_vec[0] = src_vecs.data[vec_idx * K + i];
            src_vec[1] = src_vecs.data[vec_idx * K + i + 1u];
            src_vec[2] = src_vecs.data[vec_idx * K + i + 2u];
            src_vec[3] = src_vecs.data[vec_idx * K + i + 3u];
            src_vec[4] = src_vecs.data[vec_idx * K + i + 4u];
            src_vec[5] = src_vecs.data[vec_idx * K + i + 5u];
            src_vec[6] = src_vecs.data[vec_idx * K + i + 6u];
            src_vec[7] = src_vecs.data[vec_idx * K + i + 7u];
            for (var k = 0u; k < 1u; k = k + 1u) {
                ret[k] = ret[k] + src_mat.data[(row_idx + k) * K + i] * src_vec[0]
                    + src_mat.data[(row_idx + k) * K + i + 1u] * src_vec[1]
                    + src_mat.data[(row_idx + k) * K + i + 2u] * src_vec[2]
                    + src_mat.data[(row_idx + k) * K + i + 3u] * src_vec[3]
                    + src_mat.data[(row_idx + k) * K + i + 4u] * src_vec[4]
                    + src_mat.data[(row_idx + k) * K + i + 5u] * src_vec[5]
                    + src_mat.data[(row_idx + k) * K + i + 6u] * src_vec[6]
                    + src_mat.data[(row_idx + k) * K + i + 7u] * src_vec[7];
            }
        }
        for (var j = 0u; j < K - i; j = j + 1u) {
            let src_vec_e = src_vecs.data[vec_idx * K + i + j];
            for (var k = 0u; k < 1u; k = k + 1u) {
                ret[k] = ret[k] + src_mat.data[(row_idx + k) * K + i + j] * src_vec_e;
            }
        }
        for (var k = 0u; k < 1u; k = k + 1u) {
            dst_vecs.data[vec_idx * L + row_idx + k] = ret[k];
        }
        return;
    }
    if (row_idx > L + {{wg_x}}u * 4u || vec_idx > M + {{wg_y}}u) {
        let unused = atomicAdd(&(dst_vecs.errors), 1);
    }
}
