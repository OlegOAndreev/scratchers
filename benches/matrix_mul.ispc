// The most basic version of algorithm: simply multiply row by row, doing reduce_add for each row.
export void ispc_matrix_vec_mul_v1(
    uniform float src_mat[],
    uniform int src_mat_stride,
    uniform float src_vec[],
    uniform float dst_vec[],
    uniform int K,
    uniform int L
) {
    // Help the optimizer: we always pass arrays with length aligned to 4.
    src_mat_stride = src_mat_stride & ~3;
    K = K & ~3;
    for (uniform int l = 0; l < L; l++) {
        varying float accum = 0.0f;
        foreach (k = 0 ... K) {
            accum += src_mat[l * src_mat_stride + k] * src_vec[k];
        }
        dst_vec[l] = reduce_add(accum);
    }
}

const uniform int ROWS_PER_TASK = 32;

// The version of ispc_matrix_vec_mul_v1 which spawns the tasks for each 4 rows.
task void ispc_matrix_vec_mul_v1_launch_rows(
    uniform float src_mat[],
    uniform int src_mat_stride,
    uniform float src_vec[],
    uniform float dst_vec[],
    uniform int K,
    uniform int L
) {
    // Help the optimizer: we always pass arrays with length aligned to 4.
    src_mat_stride = src_mat_stride & ~3;
    K = K & ~3;
    uniform int from = taskIndex * ROWS_PER_TASK;
    uniform int to = min((uniform int)taskIndex * ROWS_PER_TASK + ROWS_PER_TASK, L);
    for (uniform int l = from; l < to; l++) {
        varying float accum = 0.0f;
        foreach (k = 0 ... K) {
            accum += src_mat[l * src_mat_stride + k] * src_vec[k];
        }
        dst_vec[l] = reduce_add(accum);
    }
}

export void ispc_matrix_vec_mul_v1_launch(
    uniform float src_mat[],
    uniform int src_mat_stride,
    uniform float src_vec[],
    uniform float dst_vec[],
    uniform int K,
    uniform int L
) {
    if (L < ROWS_PER_TASK * 4) {
        ispc_matrix_vec_mul_v1(src_mat, src_mat_stride, src_vec, dst_vec, K, L);
    } else {
        uniform int num_tasks = (L + ROWS_PER_TASK + 1) / ROWS_PER_TASK;
        launch[num_tasks] ispc_matrix_vec_mul_v1_launch_rows(src_mat, src_mat_stride, src_vec, dst_vec, K, L);
    }
}

// Try multiplying several rows at once, doing reduce_add for each row.
export void ispc_matrix_vec_mul_v2(
    uniform float src_mat[],
    uniform int src_mat_stride,
    uniform float src_vec[],
    uniform float dst_vec[],
    uniform int K,
    uniform int L
) {
    // Help the optimizer: we always pass arrays with length aligned to 4.
    src_mat_stride = src_mat_stride & ~3;
    K = K & ~3;
    uniform int last_l = (L / 4) * 4;
    for (uniform int l = 0; l < last_l; l += 4) {
        varying float accum0 = 0.0f;
        varying float accum1 = 0.0f;
        varying float accum2 = 0.0f;
        varying float accum3 = 0.0f;
        foreach (k = 0 ... K) {
            accum0 += src_mat[l * src_mat_stride + k] * src_vec[k];
            accum1 += src_mat[(l + 1) * src_mat_stride + k] * src_vec[k];
            accum2 += src_mat[(l + 2) * src_mat_stride + k] * src_vec[k];
            accum3 += src_mat[(l + 3) * src_mat_stride + k] * src_vec[k];
        }
        dst_vec[l] = reduce_add(accum0);
        dst_vec[l + 1] = reduce_add(accum1);
        dst_vec[l + 2] = reduce_add(accum2);
        dst_vec[l + 3] = reduce_add(accum3);
    }

    for (uniform int l = last_l; l < L; l++) {
        varying float accum = 0.0f;
        foreach (k = 0 ... K) {
            accum += src_mat[l * src_mat_stride + k] * src_vec[k];
        }
        dst_vec[l] = reduce_add(accum);
    }
}

// Try multiplying several rows by multiple vectors at once.
export void ispc_matrix_vec_mul_v3(
    uniform float src_mat[],
    uniform int src_mat_stride,
    uniform float src_vec0[],
    uniform float src_vec1[],
    uniform float src_vec2[],
    uniform float src_vec3[],
    uniform float dst_vec0[],
    uniform float dst_vec1[],
    uniform float dst_vec2[],
    uniform float dst_vec3[],
    uniform int K,
    uniform int L
) {
    // Help the optimizer: we always pass arrays with length aligned to 4.
    src_mat_stride = src_mat_stride & ~3;
    K = K & ~3;
    uniform int last_l = (L / 4) * 4;
    for (uniform int l = 0; l < last_l; l += 4) {
        varying float accum0x0 = 0.0f;
        varying float accum1x0 = 0.0f;
        varying float accum2x0 = 0.0f;
        varying float accum3x0 = 0.0f;
        varying float accum0x1 = 0.0f;
        varying float accum1x1 = 0.0f;
        varying float accum2x1 = 0.0f;
        varying float accum3x1 = 0.0f;
        varying float accum0x2 = 0.0f;
        varying float accum1x2 = 0.0f;
        varying float accum2x2 = 0.0f;
        varying float accum3x2 = 0.0f;
        varying float accum0x3 = 0.0f;
        varying float accum1x3 = 0.0f;
        varying float accum2x3 = 0.0f;
        varying float accum3x3 = 0.0f;
        foreach (k = 0 ... K) {
            accum0x0 += src_mat[l * src_mat_stride + k] * src_vec0[k];
            accum1x0 += src_mat[(l + 1) * src_mat_stride + k] * src_vec0[k];
            accum2x0 += src_mat[(l + 2) * src_mat_stride + k] * src_vec0[k];
            accum3x0 += src_mat[(l + 3) * src_mat_stride + k] * src_vec0[k];
            accum0x1 += src_mat[l * src_mat_stride + k] * src_vec1[k];
            accum1x1 += src_mat[(l + 1) * src_mat_stride + k] * src_vec1[k];
            accum2x1 += src_mat[(l + 2) * src_mat_stride + k] * src_vec1[k];
            accum3x1 += src_mat[(l + 3) * src_mat_stride + k] * src_vec1[k];
            accum0x2 += src_mat[l * src_mat_stride + k] * src_vec2[k];
            accum1x2 += src_mat[(l + 1) * src_mat_stride + k] * src_vec2[k];
            accum2x2 += src_mat[(l + 2) * src_mat_stride + k] * src_vec2[k];
            accum3x2 += src_mat[(l + 3) * src_mat_stride + k] * src_vec2[k];
            accum0x3 += src_mat[l * src_mat_stride + k] * src_vec3[k];
            accum1x3 += src_mat[(l + 1) * src_mat_stride + k] * src_vec3[k];
            accum2x3 += src_mat[(l + 2) * src_mat_stride + k] * src_vec3[k];
            accum3x3 += src_mat[(l + 3) * src_mat_stride + k] * src_vec3[k];
        }
        dst_vec0[l] = reduce_add(accum0x0);
        dst_vec0[l + 1] = reduce_add(accum1x0);
        dst_vec0[l + 2] = reduce_add(accum2x0);
        dst_vec0[l + 3] = reduce_add(accum3x0);
        dst_vec1[l] = reduce_add(accum0x1);
        dst_vec1[l + 1] = reduce_add(accum1x1);
        dst_vec1[l + 2] = reduce_add(accum2x1);
        dst_vec1[l + 3] = reduce_add(accum3x1);
        dst_vec2[l] = reduce_add(accum0x2);
        dst_vec2[l + 1] = reduce_add(accum1x2);
        dst_vec2[l + 2] = reduce_add(accum2x2);
        dst_vec2[l + 3] = reduce_add(accum3x2);
        dst_vec3[l] = reduce_add(accum0x3);
        dst_vec3[l + 1] = reduce_add(accum1x3);
        dst_vec3[l + 2] = reduce_add(accum2x3);
        dst_vec3[l + 3] = reduce_add(accum3x3);
    }

    for (uniform int l = last_l; l < L; l++) {
        varying float accum0 = 0.0f;
        varying float accum1 = 0.0f;
        varying float accum2 = 0.0f;
        varying float accum3 = 0.0f;
        foreach (k = 0 ... K) {
            accum0 += src_mat[l * src_mat_stride + k] * src_vec0[k];
            accum1 += src_mat[l * src_mat_stride + k] * src_vec1[k];
            accum2 += src_mat[l * src_mat_stride + k] * src_vec2[k];
            accum3 += src_mat[l * src_mat_stride + k] * src_vec3[k];
        }
        dst_vec0[l] = reduce_add(accum0);
        dst_vec1[l] = reduce_add(accum1);
        dst_vec2[l] = reduce_add(accum2);
        dst_vec3[l] = reduce_add(accum3);
    }
}
