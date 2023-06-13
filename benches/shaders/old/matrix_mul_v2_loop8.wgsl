// The file actually contains the liquid template, see https://shopify.github.io/liquid/
{% include 'matrix_mul_common.wgsl' %}

// v2_loop8 is a version of v2_vec8 which uses internal loop and plain f32 instead of 2x vec4 to read and multiply rows.

[[stage(compute), workgroup_size({{wg_x}}, {{wg_y}})]]
fn main_v2_loop8(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>
) {
    let row_idx = global_id.x * 4u;
    let vec_idx = global_id.y;
    let K = uniforms.K;
    let L = uniforms.L;
    let M = uniforms.M;
    if (row_idx < L - 3u && vec_idx < M) {
        var src_vec: array<f32, 8>;
        var ret: array<array<f32, 8>, 4>;
        // TODO: This should not be needed, arrays should be automatically initialized to zero. See if this is
        // a bug in naga -> MSL translation.
{% if zero_out %}
        for (var k = 0u; k < 4u; k = k + 1u) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                ret[k][j] = 0.0;
            }
        }
{% endif %}
        var i = 0u;
        for (; i < K - 7u; i = i + 8u) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                src_vec[j] = src_vecs.data[vec_idx * K + i + j];
            }
            for (var k = 0u; k < 4u; k = k + 1u) {
                for (var j = 0u; j < 8u; j = j + 1u) {
                    ret[k][j] = ret[k][j] + src_mat.data[(row_idx + k) * K + i + j] * src_vec[j];
                }
            }
        }
        for (var j = 0u; j < K - i; j = j + 1u) {
            let src_vec_e = src_vecs.data[vec_idx * K + i + j];
            for (var k = 0u; k < 4u; k = k + 1u) {
                ret[k][j] = ret[k][j] + src_mat.data[(row_idx + k) * K + i + j] * src_vec_e;
            }
        }
        for (var k = 0u; k < 4u; k = k + 1u) {
            var acc = 0.0;
            for (var j = 0u; j < 8u; j = j + 1u) {
                acc = acc + ret[k][j];
            }
            dst_vecs.data[vec_idx * L + row_idx + k] = acc;
        }
        return;
    }
    if (row_idx < L - 2u && vec_idx < M) {
        var src_vec: array<f32, 8>;
        var ret: array<array<f32, 8>, 3>;
        // TODO: This should not be needed, arrays should be automatically initialized to zero. See if this is
        // a bug in naga -> MSL translation.
{% if zero_out %}
        for (var k = 0u; k < 3u; k = k + 1u) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                ret[k][j] = 0.0;
            }
        }
{% endif %}
        var i = 0u;
        for (; i < K - 7u; i = i + 8u) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                src_vec[j] = src_vecs.data[vec_idx * K + i + j];
            }
            for (var k = 0u; k < 3u; k = k + 1u) {
                for (var j = 0u; j < 8u; j = j + 1u) {
                    ret[k][j] = ret[k][j] + src_mat.data[(row_idx + k) * K + i + j] * src_vec[j];
                }
            }
        }
        for (var j = 0u; j < K - i; j = j + 1u) {
            let src_vec_e = src_vecs.data[vec_idx * K + i + j];
            for (var k = 0u; k < 3u; k = k + 1u) {
                ret[k][j] = ret[k][j] + src_mat.data[(row_idx + k) * K + i + j] * src_vec_e;
            }
        }
        for (var k = 0u; k < 3u; k = k + 1u) {
            var acc = 0.0;
            for (var j = 0u; j < 8u; j = j + 1u) {
                acc = acc + ret[k][j];
            }
            dst_vecs.data[vec_idx * L + row_idx + k] = acc;
        }
        return;
    }
    if (row_idx < L - 1u && vec_idx < M) {
        var src_vec: array<f32, 8>;
        var ret: array<array<f32, 8>, 2>;
        // TODO: This should not be needed, arrays should be automatically initialized to zero. See if this is
        // a bug in naga -> MSL translation.
{% if zero_out %}
        for (var k = 0u; k < 2u; k = k + 1u) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                ret[k][j] = 0.0;
            }
        }
{% endif %}
        var i = 0u;
        for (; i < K - 7u; i = i + 8u) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                src_vec[j] = src_vecs.data[vec_idx * K + i + j];
            }
            for (var k = 0u; k < 2u; k = k + 1u) {
                for (var j = 0u; j < 8u; j = j + 1u) {
                    ret[k][j] = ret[k][j] + src_mat.data[(row_idx + k) * K + i + j] * src_vec[j];
                }
            }
        }
        for (var j = 0u; j < K - i; j = j + 1u) {
            let src_vec_e = src_vecs.data[vec_idx * K + i + j];
            for (var k = 0u; k < 2u; k = k + 1u) {
                ret[k][j] = ret[k][j] + src_mat.data[(row_idx + k) * K + i + j] * src_vec_e;
            }
        }
        for (var k = 0u; k < 2u; k = k + 1u) {
            var acc = 0.0;
            for (var j = 0u; j < 8u; j = j + 1u) {
                acc = acc + ret[k][j];
            }
            dst_vecs.data[vec_idx * L + row_idx + k] = acc;
        }
        return;
    }
    if (row_idx < L && vec_idx < M) {
        var src_vec: array<f32, 8>;
        var ret: array<array<f32, 8>, 1>;
        // TODO: This should not be needed, arrays should be automatically initialized to zero. See if this is
        // a bug in naga -> MSL translation.
{% if zero_out %}
        for (var k = 0u; k < 1u; k = k + 1u) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                ret[k][j] = 0.0;
            }
        }
{% endif %}
        var i = 0u;
        for (; i < K - 7u; i = i + 8u) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                src_vec[j] = src_vecs.data[vec_idx * K + i + j];
            }
            for (var k = 0u; k < 1u; k = k + 1u) {
                for (var j = 0u; j < 8u; j = j + 1u) {
                    ret[k][j] = ret[k][j] + src_mat.data[(row_idx + k) * K + i + j] * src_vec[j];
                }
            }
        }
        for (var j = 0u; j < K - i; j = j + 1u) {
            let src_vec_e = src_vecs.data[vec_idx * K + i + j];
            for (var k = 0u; k < 1u; k = k + 1u) {
                ret[k][j] = ret[k][j] + src_mat.data[(row_idx + k) * K + i + j] * src_vec_e;
            }
        }
        for (var k = 0u; k < 1u; k = k + 1u) {
            var acc = 0.0;
            for (var j = 0u; j < 8u; j = j + 1u) {
                acc = acc + ret[k][j];
            }
            dst_vecs.data[vec_idx * L + row_idx + k] = acc;
        }
        return;
    }
    if (row_idx > L + {{wg_x}}u * 4u || vec_idx > M + {{wg_y}}u) {
        let unused = atomicAdd(&(dst_vecs.errors), 1);
    }
}

